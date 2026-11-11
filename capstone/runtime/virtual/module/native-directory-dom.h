/* SPDX-License-Identifier: GPL-2.0-only */
/* Path operations on pinned directory grants. Normal opens never admit a
 * symlink; only the metadata/readlink path below admits a final link inode. */
#include <linux/delayed_call.h>
#include <linux/dirent.h>
#include <linux/namei.h>
#include <linux/stat.h>
#include <asm/stat.h>


struct native_directory_entries {
    struct dir_context ctx;
    char *next;
    int remaining, previous, error;
};

static bool native_directory_entry(struct dir_context *ctx, const char *name,
        int length, loff_t offset, u64 ino, unsigned int type)
{
    struct native_directory_entries *out = container_of(ctx, struct native_directory_entries, ctx);
    struct linux_dirent64 *entry, *previous;
    int bytes;
    /* Match Linux 6.1 verify_dirent_name/filldir64, including returning
     * already completed records if a later record cannot be emitted. */
    if (length <= 0 || length >= PATH_MAX || memchr(name, '/', length)) {
        out->error = -EIO;
        return false;
    }
    bytes = ALIGN(offsetof(struct linux_dirent64, d_name) + length + 1, sizeof(u64));
    out->error = -EINVAL;
    if (bytes > out->remaining || (out->previous && signal_pending(current))) return false;
    entry = (struct linux_dirent64 *)out->next;
    if (out->previous) {
        previous = (struct linux_dirent64 *)(out->next - out->previous);
        previous->d_off = offset;
    }
    memset(entry, 0, bytes);
    entry->d_ino = ino;
    entry->d_reclen = bytes;
    entry->d_type = type;
    memcpy(entry->d_name, name, length);
    out->next += bytes;
    out->remaining -= bytes;
    out->previous = bytes;
    return true;
}

static long native_directory_getdents(struct file *file, void *data, size_t bytes)
{
    struct native_directory_entries out = {
        .ctx.actor = native_directory_entry, .next = data, .remaining = bytes
    };
    bool locked = file->f_mode & FMODE_ATOMIC_POS;
    long rc;
    BUILD_BUG_ON(offsetof(struct linux_dirent64, d_name) != 19);
    if (bytes > INT_MAX) return -EINVAL;
    if (file->f_mode & FMODE_PATH) return -EBADF;
    if (locked) mutex_lock(&file->f_pos_lock);
    rc = iterate_dir(file, &out.ctx);
    if (rc >= 0) rc = out.error;
    if (out.previous) {
        struct linux_dirent64 *last = (struct linux_dirent64 *)(out.next - out.previous);
        last->d_off = out.ctx.pos;
        rc = bytes - out.remaining;
    }
    if (locked) mutex_unlock(&file->f_pos_lock);
    return rc;
}

static bool native_directory_metadata_inode(const struct inode *inode)
{
    if (dom_disk_inode(inode)) return true;
    if (!S_ISLNK(inode->i_mode)) return false;
    switch (inode->i_sb->s_magic) {
    case TMPFS_MAGIC: case EXT4_SUPER_MAGIC: case V9FS_MAGIC: return true;
    default: return false;
    }
}

static struct file *native_directory_metadata_file(struct context *c,
        const struct capstone_delegate_entry *e, const char *name)
{
    struct path parent;
    const struct path *grant;
    struct file *file;
    char *storage, *leaf;
    int rc = dom_parent(c, e, &parent, &storage, &leaf, &grant, name);
    if (rc) return ERR_PTR(rc);
    /* Preserve the existing exact device-pin case without granting aliases. */
    if (!strcmp(leaf, ".")) file = dom_lookup(c, &parent, leaf, grant);
    else {
        file = file_open_root(&parent, leaf, O_PATH | O_NOFOLLOW, 0);
        if (!IS_ERR(file) &&
            (!dom_beneath(&file->f_path, grant) ||
             !native_directory_metadata_inode(file_inode(file)))) {
            fput(file);
            file = ERR_PTR(-EACCES);
        }
    }
    path_put(&parent);
    kfree(storage);
    return file;
}

static long native_directory_change(struct context *c,
        const struct capstone_delegate_entry *e, const char *snapshot,
        const char *target)
{
    struct path parent;
    const struct path *grant;
    struct inode *dir;
    struct dentry *entry;
    char *storage, *leaf, *name;
    size_t length;
    umode_t mode = e->args[2];
    long rc;

    name = kstrdup(snapshot, GFP_KERNEL);
    if (!name) return -ENOMEM;
    /* mkdir/rmdir accept a trailing slash without requiring the final
     * directory to exist before mkdir. Intermediate components still use
     * the ordinary no-symlink grant walk. */
    if (e->nr != CAPSTONE_SYS_symlinkat) {
        length = strlen(name);
        while (length > 1 && name[length - 1] == '/') name[--length] = 0;
    }
    rc = dom_parent(c, e, &parent, &storage, &leaf, &grant, name);
    if (rc) goto free_name;
    dir = d_inode(parent.dentry);
    if (!S_ISDIR(dir->i_mode)) { rc = -ENOTDIR; goto put_parent; }
    if (!dom_disk_inode(dir) || !dom_beneath(&parent, grant)) {
        rc = -EACCES;
        goto put_parent;
    }
    /* dom_parent normalizes a final '..' and an exact grant root to '.'.
     * Never pass that dentry to vfs_rmdir while locking its own inode. */
    if (!strcmp(leaf, ".")) {
        rc = -EEXIST;
        if (e->nr == CAPSTONE_SYS_unlinkat) {
            const char *last = strrchr(name, '/');
            last = last ? last + 1 : name;
            rc = !strcmp(last, "..") ? -ENOTEMPTY :
                 !strcmp(last, ".") ? -EINVAL : -EBUSY;
        }
        goto put_parent;
    }
    rc = mnt_want_write(parent.mnt);
    if (rc) goto put_parent;
    inode_lock_nested(dir, I_MUTEX_PARENT);
    entry = lookup_one(mnt_user_ns(parent.mnt), leaf, parent.dentry, strlen(leaf));
    if (IS_ERR(entry)) rc = PTR_ERR(entry);
    else {
        switch (e->nr) {
        case CAPSTONE_SYS_mkdirat:
            if (d_is_positive(entry)) rc = -EEXIST;
            else {
                /* Linux's mode_strip_umask is local to namei.c. Its LSM
                 * argument leaves ACL-supported filesystems to apply umask. */
                umode_t checked_mode = IS_POSIXACL(dir) ? mode : mode & ~current_umask();
                rc = security_path_mkdir(&parent, entry, checked_mode);
                if (!rc) rc = vfs_mkdir(mnt_user_ns(parent.mnt), dir, entry, mode);
            }
            break;
        case CAPSTONE_SYS_unlinkat:
            if (d_is_negative(entry)) rc = -ENOENT;
            else {
                rc = security_path_rmdir(&parent, entry);
                if (!rc) rc = vfs_rmdir(mnt_user_ns(parent.mnt), dir, entry);
            }
            break;
        default: /* symlinkat: target is inert text, never resolved here. */
            if (d_is_positive(entry)) rc = -EEXIST;
            else {
                rc = security_path_symlink(&parent, entry, target);
                if (!rc) rc = vfs_symlink(mnt_user_ns(parent.mnt), dir, entry, target);
            }
            break;
        }
        dput(entry);
    }
    inode_unlock(dir);
    mnt_drop_write(parent.mnt);
put_parent:
    path_put(&parent);
    kfree(storage);
free_name:
    kfree(name);
    return rc;
}

static long native_directory_link(struct context *c, const struct ns_invocation *call)
{
    struct capstone_delegate_entry e = {.nr = CAPSTONE_SYS_linkat};
    struct file *source;
    struct path parent;
    const struct path *grant;
    struct dentry *entry;
    char *storage, *leaf;
    long rc = -ENOSYS;
    if (call->scalar[4]) return -EINVAL; /* No FOLLOW or EMPTY_PATH authority. */
    e.args[0] = call->scalar[0];
    source = dom_path_file_name(c, &e, call->buffer[1].data);
    if (IS_ERR(source)) return PTR_ERR(source);
    if (!dom_disk_inode(file_inode(source)) || !S_ISREG(file_inode(source)->i_mode)) {
        rc = -EPERM;
        goto put_source;
    }
    e.args[0] = call->scalar[2];
    rc = dom_parent(c, &e, &parent, &storage, &leaf, &grant, call->buffer[3].data);
    if (rc) goto put_source;
    if (!dom_disk_inode(d_inode(parent.dentry)) || !dom_beneath(&parent, grant)) {
        rc = -EACCES;
        goto put_parent;
    }
    if (!S_ISDIR(d_inode(parent.dentry)->i_mode)) { rc = -ENOTDIR; goto put_parent; }
    if (!strcmp(leaf, ".")) { rc = -EEXIST; goto put_parent; }
    rc = mnt_want_write(parent.mnt);
    if (rc) goto put_parent;
    inode_lock_nested(d_inode(parent.dentry), I_MUTEX_PARENT);
    entry = lookup_one(mnt_user_ns(parent.mnt), leaf, parent.dentry, strlen(leaf));
    if (IS_ERR(entry)) rc = PTR_ERR(entry);
    else {
        rc = d_is_positive(entry) ? -EEXIST : capstone_native_path_link(&source->f_path, &parent, entry);
        dput(entry);
    }
    inode_unlock(d_inode(parent.dentry));
    mnt_drop_write(parent.mnt);
put_parent:
    path_put(&parent);
    kfree(storage);
put_source:
    fput(source);
    return rc;
}

static bool native_directory_issue(struct context *c,
        const struct ns_invocation *call, struct file *held, long *result)
{
    const u64 *a = call->scalar;
    u64 nr = call->descriptor->nr;
    struct capstone_delegate_entry e = {.nr = nr};
    struct file *file;
    struct kstat stat;
    long rc = -ENOSYS;

    memcpy(e.args, a, sizeof(e.args));
    switch (nr) {
    case CAPSTONE_SYS_linkat:
        mutex_unlock(&vm_lock);
        rc = native_directory_link(c, call);
        mutex_lock(&vm_lock);
        break;
    case CAPSTONE_SYS_getdents64:
        mutex_unlock(&vm_lock);
        rc = native_directory_getdents(held, call->buffer[1].data, call->buffer[1].length);
        mutex_lock(&vm_lock);
        break;
    case CAPSTONE_SYS_mkdirat:
    case CAPSTONE_SYS_unlinkat:
        if (nr == CAPSTONE_SYS_unlinkat) {
            if (!a[2]) return false; /* Existing ordinary unlink path. */
            if (a[2] != AT_REMOVEDIR) { rc = -EINVAL; break; }
        }
        mutex_unlock(&vm_lock);
        rc = native_directory_change(c, &e, call->buffer[1].data, NULL);
        mutex_lock(&vm_lock);
        break;
    case CAPSTONE_SYS_symlinkat:
        e.args[0] = a[1]; /* dom_parent's directory argument is always arg0. */
        if (!*(const char *)call->buffer[0].data) { rc = -ENOENT; break; }
        mutex_unlock(&vm_lock);
        rc = native_directory_change(c, &e, call->buffer[2].data,
                                      call->buffer[0].data);
        mutex_lock(&vm_lock);
        break;
    case CAPSTONE_SYS_fchmodat:
        mutex_unlock(&vm_lock);
        file = dom_path_file_name(c, &e, call->buffer[1].data);
        if (IS_ERR(file)) rc = PTR_ERR(file);
        else {
            rc = dom_disk_inode(file_inode(file)) ? capstone_native_path_chmod(&file->f_path, a[2]) : -EACCES;
            fput(file);
        }
        mutex_lock(&vm_lock);
        break;
    case CAPSTONE_SYS_truncate:
        if ((s64)a[1] < 0) { rc = -EINVAL; break; }
        e.args[0] = (u64)(s64)AT_FDCWD;
        mutex_unlock(&vm_lock);
        file = dom_path_file_name(c, &e, call->buffer[0].data);
        if (IS_ERR(file)) rc = PTR_ERR(file);
        else {
            rc = dom_disk_inode(file_inode(file)) ? vfs_truncate(&file->f_path, (s64)a[1]) : -EACCES;
            fput(file);
        }
        mutex_lock(&vm_lock);
        break;
    case CAPSTONE_SYS_newfstatat:
        if (!(a[3] & AT_SYMLINK_NOFOLLOW)) return false;
        if (a[3] & ~(u64)(AT_SYMLINK_NOFOLLOW | AT_NO_AUTOMOUNT)) {
            rc = -EINVAL;
            break;
        }
        mutex_unlock(&vm_lock);
        file = native_directory_metadata_file(c, &e, call->buffer[1].data);
        if (IS_ERR(file)) rc = PTR_ERR(file);
        else {
            rc = vfs_getattr(&file->f_path, &stat, STATX_BASIC_STATS, AT_STATX_SYNC_AS_STAT);
            if (!rc) rc = capstone_native_stat(&stat, call->buffer[2].data);
            fput(file);
        }
        mutex_lock(&vm_lock);
        break;
    case CAPSTONE_SYS_readlinkat:
        if (!a[3] || a[3] > INT_MAX) { rc = -EINVAL; break; }
        mutex_unlock(&vm_lock);
        file = native_directory_metadata_file(c, &e, call->buffer[1].data);
        if (IS_ERR(file)) rc = PTR_ERR(file);
        else {
            DEFINE_DELAYED_CALL(done);
            const char *link = vfs_get_link(file->f_path.dentry, &done);
            if (IS_ERR(link)) rc = PTR_ERR(link);
            else if (!link) rc = -EIO;
            else {
                touch_atime(&file->f_path);
                rc = strnlen(link, a[3]);
                memcpy(call->buffer[2].data, link, rc);
            }
            do_delayed_call(&done);
            fput(file);
        }
        mutex_lock(&vm_lock);
        break;
    default: return false;
    }
    *result = rc;
    return true;
}
