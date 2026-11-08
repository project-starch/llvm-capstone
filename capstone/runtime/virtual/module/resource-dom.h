/* SPDX-License-Identifier: GPL-2.0-only */
#include <linux/security.h>
/* Native exec's resource boundary. This is an intentionally smaller ABI than
 * the launcher: the initial cwd is a directory grant, not ambient path access.
 * Linux 6.1 exports no in-kernel openat2 entry point to modules. Walk one pinned
 * O_PATH component at a time instead: '..' stays inside its original grant,
 * with no symlinks (including magic links) or mount transitions. Never open a
 * device before checking its inode. */
#include <linux/fs_struct.h>
static char *dom_shell;
module_param(dom_shell, charp, 0400);
MODULE_PARM_DESC(dom_shell, "trusted pinned native shell executable for /bin/sh (unset by default)");
static bool dom_null;
module_param(dom_null, bool, 0400);
MODULE_PARM_DESC(dom_null, "explicit pinned /dev/null grant (default off)");
static bool dom_local_tmp;
module_param(dom_local_tmp, bool, 0400);
MODULE_PARM_DESC(dom_local_tmp,
    "alias /tmp to private .capstone-tmp inside the initial cwd grant (default off)");
/* MEASUREMENT ONLY (the native route on C.5a's model, where one native task is live at a time): an exec
 * whose cwd lies beneath this directory takes it as its grant root and keeps the cwd as its working
 * directory, as a native chdir inside the grant would leave it. Today's board perl stage execs each test
 * from t/ with -I../lib; with the cwd as root that path is outside the grant. Unset: the cwd is the root. */
static char *dom_grant_root;
module_param(dom_grant_root, charp, 0400);
MODULE_PARM_DESC(dom_grant_root, "MEASUREMENT ONLY: grant root for execs beneath it (default unset: the cwd)");
static bool dom_wide_grants;
module_param(dom_wide_grants, bool, 0444);
MODULE_PARM_DESC(dom_wide_grants,
    "native exec also pins /tmp, /usr and null/zero/urandom (default off)");
static const char *const dom_dir_names[] = {"/tmp", "/usr"};
static const char *const dom_device_names[] = {"/dev/null", "/dev/zero", "/dev/urandom"};
static const unsigned dom_device_minors[] = {3, 5, 9};

static bool dom_disk_inode(const struct inode *inode)
{
    switch (inode->i_sb->s_magic) {
    case TMPFS_MAGIC: case EXT4_SUPER_MAGIC: case V9FS_MAGIC:
        return S_ISREG(inode->i_mode) || S_ISDIR(inode->i_mode);
    default: return false;
    }
}

static bool dom_stdio_file(const struct file *f)
{
    const struct inode *inode = file_inode(f);
    return (dom_disk_inode(inode) && S_ISREG(inode->i_mode)) ||
           (inode->i_sb->s_magic == PIPEFS_MAGIC && S_ISFIFO(inode->i_mode)) ||
           (S_ISCHR(inode->i_mode) &&
            (inode->i_rdev == MKDEV(1, 3) || inode->i_rdev == MKDEV(5, 1) ||
             inode->i_rdev == MKDEV(4, 64))); /* null/console/first serial terminal */
}

static bool dom_fd_allowed(struct context *c, u64 fd)
{
    return fd < DOM_FDS &&
           test_bit(fd, c->dom_delegate->fds);
}

/* Only the trusted exec policy chooses these roots. No application name is
 * resolved here. Pin /dev first so even the intermediate component cannot
 * traverse a symlink; the exact three character devices get individual
 * grants, never a directory grant to /dev or permission to open aliases. */
static int dom_extra_resources(struct dom_delegate *d)
{
    struct file *f, *dev;
    int rc = 0;
    /* /tmp cannot simultaneously name the entire launch grant and its
     * private child. Preserve physical-root pathname round trips elsewhere. */
    if (dom_local_tmp && (dom_wide_grants || !strcmp(d->root_name, "/tmp"))) return -EINVAL;
    if (dom_local_tmp) {
        struct inode *inode;
        /* Trusted launch policy creates this reserved child before exec.
         * Pin the object once; an application rename or replacement must
         * never redirect the logical /tmp prefix into ambient storage. */
        f = file_open_root(&d->root, ".capstone-tmp",
                           O_PATH | O_NOFOLLOW | O_DIRECTORY, 0);
        if (IS_ERR(f)) return PTR_ERR(f);
        inode = file_inode(f);
        if (f->f_path.mnt != d->root.mnt ||
            !is_subdir(f->f_path.dentry, d->root.dentry) ||
            !dom_disk_inode(inode) || !S_ISDIR(inode->i_mode) ||
            (inode->i_mode & (S_IRWXUGO | S_ISUID | S_ISGID | S_ISVTX)) != 0700 ||
            !uid_eq(i_uid_into_mnt(mnt_user_ns(f->f_path.mnt), inode), current_fsuid()))
            rc = -EACCES;
        if (!rc) { d->dirs[0] = f->f_path; path_get(&d->dirs[0]); }
        fput(f);
        if (rc) return rc;
    }
    if (!dom_wide_grants && !dom_null) return 0;
    for (unsigned i = 0; dom_wide_grants && i < DOM_EXTRA_DIRS; ++i) {
        f = filp_open(dom_dir_names[i], O_PATH | O_NOFOLLOW | O_DIRECTORY, 0);
        if (IS_ERR(f)) return PTR_ERR(f);
        if (!dom_disk_inode(file_inode(f))) rc = -EACCES;
        if (!rc) { d->dirs[i] = f->f_path; path_get(&d->dirs[i]); }
        fput(f);
        if (rc) return rc;
    }
    dev = filp_open("/dev", O_PATH | O_NOFOLLOW | O_DIRECTORY, 0);
    if (IS_ERR(dev)) return PTR_ERR(dev);
    for (unsigned i = 0; i < (dom_wide_grants ? DOM_DEVICES : 1); ++i) {
        f = file_open_root(&dev->f_path, dom_device_names[i] + 5, O_PATH | O_NOFOLLOW, 0);
        if (IS_ERR(f)) { rc = PTR_ERR(f); break; }
        if (f->f_path.mnt != dev->f_path.mnt || !S_ISCHR(file_inode(f)->i_mode) ||
            file_inode(f)->i_rdev != MKDEV(1, dom_device_minors[i])) rc = -EACCES;
        if (!rc) { d->devices[i] = f->f_path; path_get(&d->devices[i]); }
        fput(f);
        if (rc) break;
    }
    fput(dev);
    return rc;
}

static const struct path *dom_device_path(struct dom_delegate *d, const char *name)
{
    for (unsigned i = 0; i < DOM_DEVICES; ++i)
        if (d->devices[i].dentry && !strcmp(name, dom_device_names[i]))
            return &d->devices[i];
    return NULL;
}

static int dom_resources_init(struct context *c)
{
    struct dom_delegate *d;
    char *buf, *name;
    unsigned i;
    int inherited = dom_spawn_resources(c);
    if (inherited) return inherited < 0 ? inherited : 0;
    d = kzalloc(sizeof(*d), GFP_KERNEL);
    if (!d) return -ENOMEM;
    c->dom_delegate = d;
    get_fs_pwd(current->fs, &d->root);
    d->cwd_path = d->root;
    path_get(&d->cwd_path);
    if (dom_grant_root && *dom_grant_root) {
        struct path wide;
        int rc = kern_path(dom_grant_root, LOOKUP_DIRECTORY, &wide);
        if (rc) return rc;
        if (d->cwd_path.mnt == wide.mnt && is_subdir(d->cwd_path.dentry, wide.dentry)) {
            path_put(&d->root);
            d->root = wide;
        } else path_put(&wide);
    }
    if (!dom_disk_inode(d_inode(d->root.dentry))) return -EACCES;
    buf = kmalloc(PATH_MAX, GFP_KERNEL);
    if (!buf) return -ENOMEM;
    name = d_path(&d->root, buf, PATH_MAX);
    if (!IS_ERR(name)) d->root_name = kstrdup(name, GFP_KERNEL);
    kfree(buf);
    if (IS_ERR(name)) return PTR_ERR(name);
    if (!d->root_name) return -ENOMEM;
    /* A whole-filesystem cwd is not a useful confinement grant. */
    if (!strcmp(d->root_name, "/")) return -EACCES;
    for (i = 0; i < 3; ++i) {
        struct file *f = fget(i);
        if (f) {
            if (dom_stdio_file(f)) set_bit(i, d->fds);
            fput(f);
        }
    }
    if (dom_shell && *dom_shell) {
        d->shell = filp_open(dom_shell, O_RDONLY | O_NOFOLLOW | O_CLOEXEC, 0);
        if (IS_ERR(d->shell)) {
            int rc = PTR_ERR(d->shell);
            d->shell = NULL;
            return rc;
        }
        if (!dom_disk_inode(file_inode(d->shell)) || !S_ISREG(file_inode(d->shell)->i_mode))
            return -EACCES;
    }
    return dom_extra_resources(d);
}

static bool dom_beneath(const struct path *path, const struct path *root)
{
    return path->mnt == root->mnt && is_subdir(path->dentry, root->dentry);
}

static const struct path *dom_directory_grant(struct dom_delegate *d,
                                              const struct path *path)
{
    if (dom_beneath(path, &d->root)) return &d->root;
    for (unsigned i = 0; i < DOM_EXTRA_DIRS; ++i)
        /* The local alias remains a descendant of the original grant,
         * unlike independent explicitly granted directories. */
        if (!(dom_local_tmp && i == 0) &&
            d->dirs[i].dentry && dom_beneath(path, &d->dirs[i]))
            return &d->dirs[i];
    return NULL;
}

/* Return a held parent and a single final component in an owned string.
 * Absolute names select an explicit pinned grant by component boundary;
 * this is an authority prefix, not a blacklist or a security canonicalizer.
 * The subsequent component walk supplies the actual resolution guarantee. */
static int dom_parent(struct context *c, const struct capstone_delegate_entry *e,
                      struct path *parent, char **storage, char **leaf,
                      const struct path **grant, const char *snapshot)
{
    struct dom_delegate *d = c->dom_delegate;
    const struct path *root = NULL;
    struct file *f;
    char *name, *at, *part;
    unsigned rename_seq = read_seqbegin(&rename_lock);
    bool dotdot = false;
    long bytes;
    int rc = 0;
    if (snapshot) {
        name = kstrdup(snapshot, GFP_KERNEL);
        if (!name) return -ENOMEM;
    } else {
    bytes = strnlen_user((char __user *)(c->dom_exchange + e->args[1]),
                         min_t(u64, PATH_MAX, c->dom_exchange_bytes - e->args[1]));
    if (bytes <= 0 || bytes > PATH_MAX || bytes > c->dom_exchange_bytes - e->args[1])
        return -EFAULT;
    name = memdup_user((void __user *)(c->dom_exchange + e->args[1]), bytes);
    if (IS_ERR(name)) return PTR_ERR(name);
    }
    at = name;
    if (!*at) { rc = -ENOENT; goto out; }
    if (*at == '/') {
        char *root_buf = kmalloc(PATH_MAX, GFP_KERNEL), *root_name;
        size_t n;
        if (!root_buf) { rc = -ENOMEM; goto out; }
        /* The pinned object is immutable authority; its pathname is not.
         * Resolve absolute names under its current name so getcwd remains
         * usable after a trusted rename of the launch directory. */
        root_name = d_path(&d->root, root_buf, PATH_MAX);
        if (IS_ERR(root_name)) { rc = PTR_ERR(root_name); kfree(root_buf); goto out; }
        n = strlen(root_name);
        if (!strncmp(at, root_name, n) && (!at[n] || at[n] == '/')) root = &d->root;
        kfree(root_buf);
        for (unsigned i = 0; i < DOM_EXTRA_DIRS; ++i) {
            size_t len = strlen(dom_dir_names[i]);
            if (d->dirs[i].dentry && (!root || len > n) &&
                (!(dom_local_tmp && i == 0) || dom_beneath(&d->dirs[i], &d->root)) &&
                !strncmp(at, dom_dir_names[i], len) && (!at[len] || at[len] == '/')) {
                root = &d->dirs[i]; n = len;
            }
        }
        /* Exact device names resolve directly to pinned objects. The final
         * dot lookup below cannot turn a device into a directory authority. */
        if (!root) {
            root = dom_device_path(d, at);
            if (root) n = strlen(at);
        }
        if (!root) {
            /* Metadata lookup does not reveal the existence of objects
             * outside the grant. This also preserves optional search paths
             * (for example a compiler's absent system include directory). */
            rc = (e->nr == CAPSTONE_SYS_newfstatat || e->nr == CAPSTONE_SYS_statx) ?
                 -ENOENT : -EACCES;
            goto out;
        }
        at += n;
        while (*at == '/') ++at;
        *parent = *root; path_get(parent);
    } else if ((int)e->args[0] == AT_FDCWD) {
        *parent = d->cwd_path; path_get(parent);
    } else {
        if (!dom_fd_allowed(c, e->args[0])) { rc = -EBADF; goto out; }
        f = fget(e->args[0]);
        if (!f) { rc = -EBADF; goto out; }
        *parent = f->f_path; path_get(parent); fput(f);
        if (!S_ISDIR(d_inode(parent->dentry)->i_mode)) { rc = -ENOTDIR; goto put; }
    }
    if (!root) root = dom_directory_grant(d, parent);
    if (!root || !dom_beneath(parent, root) ||
        (dom_local_tmp && root == &d->dirs[0] && !dom_beneath(parent, &d->root))) {
        rc = -EACCES; goto put;
    }
    while ((part = strsep(&at, "/")) != NULL) {
        if (!strcmp(part, "..")) {
            struct path up = {.mnt = parent->mnt};
            dotdot = true;
            if (path_equal(parent, root) || parent->dentry == parent->mnt->mnt_root ||
                !dom_beneath(parent, root)) { rc = -EACCES; goto put; }
            rc = inode_permission(mnt_user_ns(parent->mnt), d_inode(parent->dentry), MAY_EXEC);
            if (rc) goto put;
            up.dentry = dget_parent(parent->dentry);
            /* As in namei's scoped handle_dots(), a rename anywhere since
             * this walk began makes parent traversal ambiguous. Never use
             * a lexical prefix or a check made before dget_parent(). */
            if (read_seqretry(&rename_lock, rename_seq)) rc = -EAGAIN;
            else if (!dom_beneath(&up, root)) rc = -EACCES;
            if (rc) { dput(up.dentry); goto put; }
            dput(parent->dentry);
            parent->dentry = up.dentry;
            part = ".";
        }
        if (!at) {
            if (dotdot && read_seqretry(&rename_lock, rename_seq)) { rc = -EAGAIN; goto put; }
            if (!dom_beneath(parent, root) ||
                (dom_local_tmp && root == &d->dirs[0] && !dom_beneath(parent, &d->root))) {
                rc = -EACCES; goto put;
            }
            *leaf = *part ? part : ".";
            *storage = name;
            *grant = root;
            return 0;
        }
        if (!*part || !strcmp(part, ".")) continue;
        f = file_open_root(parent, part, O_PATH | O_NOFOLLOW | O_DIRECTORY, 0);
        if (IS_ERR(f)) { rc = PTR_ERR(f); goto put; }
        if (f->f_path.mnt != parent->mnt || !dom_disk_inode(file_inode(f))) rc = -EACCES;
        if (!rc) { path_put(parent); *parent = f->f_path; path_get(parent); }
        fput(f);
        if (rc) goto put;
    }
    rc = -ENOENT;
put:
    path_put(parent);
out:
    kfree(name);
    return rc;
}

static struct file *dom_lookup(struct context *c, const struct path *parent,
                               const char *leaf, const struct path *grant)
{
    struct file *f;
    /* A direct device grant has no pathname traversal and no aliases. */
    for (unsigned i = 0; i < DOM_DEVICES; ++i) {
        const struct path *p = &c->dom_delegate->devices[i];
        if (p->dentry && path_equal(parent, p) && !strcmp(leaf, "."))
            return dentry_open(p, O_PATH, current_cred());
    }
    f = file_open_root(parent, leaf, O_PATH | O_NOFOLLOW, 0);
    if (!IS_ERR(f) && (!dom_beneath(&f->f_path, grant) ||
                      !dom_disk_inode(file_inode(f)))) {
        fput(f); return ERR_PTR(-EACCES);
    }
    return f;
}

static struct file *dom_path_file_name(struct context *c, const struct capstone_delegate_entry *e, const char *snapshot)
{
    struct path parent;
    const struct path *grant;
    struct file *f;
    char *storage, *leaf;
    int rc = dom_parent(c, e, &parent, &storage, &leaf, &grant, snapshot);
    if (rc) return ERR_PTR(rc);
    f = dom_lookup(c, &parent, leaf, grant);
    path_put(&parent); kfree(storage);
    return f;
}

static long dom_path_operation_name(struct context *c, const struct capstone_delegate_entry *e, const char *snapshot)
{
    struct path parent;
    const struct path *grant;
    struct file *pin, *f;
    char *storage, *leaf;
    long rc;
    unsigned flags = e->args[2];
    unsigned allowed = O_ACCMODE | O_CREAT | O_EXCL | O_TRUNC | O_APPEND |
                       O_NONBLOCK | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC |
                       O_LARGEFILE | O_SYNC | O_DSYNC | O_NOCTTY;
    if (e->nr == CAPSTONE_SYS_openat && (e->args[2] & ~((u64)allowed))) return -EINVAL;
    if (e->nr == CAPSTONE_SYS_openat && (flags & O_ACCMODE) == O_ACCMODE) return -EINVAL;
    if (e->nr == CAPSTONE_SYS_unlinkat && flags) return -ENOSYS;
    rc = dom_parent(c, e, &parent, &storage, &leaf, &grant, snapshot);
    if (rc) return rc;
    if (e->nr == CAPSTONE_SYS_unlinkat) {
        struct dentry *victim;
        if (!S_ISDIR(d_inode(parent.dentry)->i_mode)) { rc = -EACCES; goto out; }
        rc = mnt_want_write(parent.mnt);
        if (rc) goto out;
        inode_lock(d_inode(parent.dentry));
        victim = lookup_one_len(leaf, parent.dentry, strlen(leaf));
        if (IS_ERR(victim)) rc = PTR_ERR(victim);
        else {
            if (d_is_negative(victim)) rc = -ENOENT;
            else rc = security_path_unlink(&parent, victim);
            if (!rc) rc = vfs_unlink(mnt_user_ns(parent.mnt), d_inode(parent.dentry), victim, NULL);
            dput(victim);
        }
        inode_unlock(d_inode(parent.dentry));
        mnt_drop_write(parent.mnt);
        goto out;
    }
    pin = dom_lookup(c, &parent, leaf, grant);
    if (IS_ERR(pin)) {
        rc = PTR_ERR(pin);
        if (rc != -ENOENT || !(flags & O_CREAT)) goto out;
        /* O_EXCL is essential: a racing special file must never be opened.
         * Creation can only make a regular file on the admitted filesystem. */
        f = file_open_root(&parent, leaf, flags | O_EXCL | O_NOFOLLOW,
                           (e->args[3] & 0777) & ~current_umask());
    } else {
        struct inode *inode = file_inode(pin);
        unsigned access = MAY_OPEN;
        if ((flags & (O_CREAT | O_EXCL)) == (O_CREAT | O_EXCL)) {
            fput(pin); rc = -EEXIST; goto out;
        }
        if ((flags & O_DIRECTORY) && !S_ISDIR(file_inode(pin)->i_mode)) {
            fput(pin); rc = -ENOTDIR; goto out;
        }
        if ((flags & O_ACCMODE) != O_WRONLY) access |= MAY_READ;
        if ((flags & O_ACCMODE) != O_RDONLY || (flags & O_TRUNC)) access |= MAY_WRITE;
        if (flags & O_APPEND) access |= MAY_APPEND;
        /* O_PATH deliberately bypasses final-inode access checks, and
         * dentry_open does not perform may_open. Preserve Linux's permission
         * and append-only restrictions on the very same pinned inode. */
        rc = inode_permission(mnt_user_ns(pin->f_path.mnt), inode, access);
        if (!rc && S_ISDIR(inode->i_mode) && ((access & MAY_WRITE) || (flags & O_CREAT))) rc = -EISDIR;
        if (!rc && IS_APPEND(inode) &&
            (((access & MAY_WRITE) && !(flags & O_APPEND)) || (flags & O_TRUNC))) rc = -EPERM;
        if (rc) { fput(pin); goto out; }
        f = dentry_open(&pin->f_path, flags & ~(O_CREAT | O_EXCL | O_TRUNC), current_cred());
        fput(pin);
        if (!IS_ERR(f) && (flags & O_TRUNC) && S_ISREG(file_inode(f)->i_mode)) {
            rc = (f->f_mode & FMODE_WRITE) ? vfs_truncate(&f->f_path, 0) : -EACCES;
            if (rc) { fput(f); goto out; }
        }
    }
    if (IS_ERR(f)) { rc = PTR_ERR(f); goto out; }
    rc = get_unused_fd_flags(flags & O_CLOEXEC);
    if (rc >= DOM_FDS) { put_unused_fd(rc); rc = -EMFILE; }
    if (rc < 0) fput(f);
    else { set_bit(rc, c->dom_delegate->fds); fd_install(rc, f); }
out:
    path_put(&parent); kfree(storage);
    return rc;
}

static long dom_chdir_name(struct context *c, const char *name)
{
    struct capstone_delegate_entry e = {.nr = CAPSTONE_SYS_chdir};
    struct dom_delegate *d = c->dom_delegate;
    struct path old;
    struct file *f;
    int rc;
    e.args[0] = (u64)(s64)AT_FDCWD;
    f = dom_path_file_name(c, &e, name);
    if (IS_ERR(f)) return PTR_ERR(f);
    rc = -ENOTDIR;
    if (!S_ISDIR(file_inode(f)->i_mode)) goto out;
    rc = inode_permission(mnt_user_ns(f->f_path.mnt), file_inode(f), MAY_EXEC | MAY_CHDIR);
    if (rc) goto out;
    /* Nothing fallible follows publication. Linux's fs cwd is kept in sync
     * so a future native spawn inherits the same directory. The immutable
     * root grant never changes here. */
    old = d->cwd_path;
    d->cwd_path = f->f_path;
    path_get(&d->cwd_path);
    set_fs_pwd(current->fs, &d->cwd_path);
    path_put(&old);
out:
    fput(f);
    return rc;
}

static long dom_getcwd(struct context *c, void *buffer, size_t bytes)
{
    char *storage, *name;
    unsigned rename_seq;
    long rc;
    if (!bytes) return -ERANGE;
    if (d_unlinked(c->dom_delegate->cwd_path.dentry)) return -ENOENT;
    storage = kmalloc(PATH_MAX, GFP_KERNEL);
    if (!storage) return -ENOMEM;
    rename_seq = read_seqbegin(&rename_lock);
    /* A trusted process may have moved cwd outside its grant. Do not expose
     * the external pathname, including a move racing the rendering below. */
    if (!dom_directory_grant(c->dom_delegate, &c->dom_delegate->cwd_path)) {
        rc = -ENOENT;
        goto out;
    }
    name = d_path(&c->dom_delegate->cwd_path, storage, PATH_MAX);
    if (IS_ERR(name)) rc = PTR_ERR(name);
    else if (read_seqretry(&rename_lock, rename_seq)) rc = -EAGAIN;
    else {
        rc = strlen(name) + 1;
        if (rc > bytes) rc = -ERANGE;
        else memcpy(buffer, name, rc);
    }
out:
    kfree(storage);
    return rc;
}
