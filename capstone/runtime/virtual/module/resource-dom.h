/* SPDX-License-Identifier: GPL-2.0-only */
/* Native exec's resource boundary. This is an intentionally smaller ABI than
 * the launcher: the initial cwd is a directory grant, not ambient path access.
 * Linux 6.1 exports no in-kernel openat2 entry point to modules. Walk one pinned
 * O_PATH component at a time instead: no '..', symlinks (including magic links)
 * or mount transitions. Never open a device before checking its inode. */
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
            (inode->i_rdev == MKDEV(1, 3) || inode->i_rdev == MKDEV(5, 1))); /* null/console */
}

static bool dom_fd_allowed(struct context *c, u64 fd)
{
    return fd < DOM_FDS && fd != c->dom_fd &&
           test_bit(fd, c->dom_delegate->fds);
}

static int dom_resources_init(struct context *c)
{
    struct dom_delegate *d;
    char *buf, *name;
    unsigned i;
    d = kzalloc(sizeof(*d), GFP_KERNEL);
    if (!d) return -ENOMEM;
    c->dom_delegate = d;
    d->path_fd = -1;
    get_fs_pwd(current->fs, &d->root);
    if (!dom_disk_inode(d_inode(d->root.dentry))) return -EACCES;
    buf = kmalloc(PATH_MAX, GFP_KERNEL);
    if (!buf) return -ENOMEM;
    name = d_path(&d->root, buf, PATH_MAX);
    if (!IS_ERR(name)) d->cwd = kstrdup(name, GFP_KERNEL);
    kfree(buf);
    if (IS_ERR(name)) return PTR_ERR(name);
    if (!d->cwd) return -ENOMEM;
    /* A whole-filesystem cwd is not a useful confinement grant. */
    if (!strcmp(d->cwd, "/")) return -EACCES;
    for (i = 0; i < 3; ++i) {
        struct file *f = fget(i);
        if (f) {
            if (dom_stdio_file(f)) set_bit(i, d->fds);
            fput(f);
        }
    }
    return 0;
}

/* Return a held parent and a single final component in an owned string.
 * Absolute names are accepted only beneath the initial cwd's spelling;
 * this is an authority prefix, not a blacklist or a security canonicalizer.
 * The subsequent component walk supplies the actual resolution guarantee. */
static int dom_parent(struct context *c, const struct capstone_delegate_entry *e,
                      struct path *parent, char **storage, char **leaf)
{
    struct dom_delegate *d = c->dom_delegate;
    struct file *f;
    char *name, *at, *part;
    long bytes;
    int rc = 0;
    bytes = strnlen_user((char __user *)(c->dom_exchange + e->args[1]),
                         min_t(u64, PATH_MAX, c->dom_exchange_bytes - e->args[1]));
    if (bytes <= 0 || bytes > PATH_MAX || bytes > c->dom_exchange_bytes - e->args[1])
        return -EFAULT;
    name = memdup_user((void __user *)(c->dom_exchange + e->args[1]), bytes);
    if (IS_ERR(name)) return PTR_ERR(name);
    at = name;
    if (!*at) { rc = -ENOENT; goto out; }
    if (*at == '/') {
        size_t n = strlen(d->cwd);
        if (strncmp(at, d->cwd, n) || (at[n] && at[n] != '/')) {
            /* Metadata lookup does not reveal the existence of objects
             * outside the grant. This also preserves optional search paths
             * (for example a compiler's absent system include directory). */
            rc = (e->nr == CAPSTONE_SYS_newfstatat || e->nr == CAPSTONE_SYS_statx) ?
                 -ENOENT : -EACCES;
            goto out;
        }
        at += n;
        while (*at == '/') ++at;
        *parent = d->root; path_get(parent);
    } else if ((int)e->args[0] == AT_FDCWD) {
        *parent = d->root; path_get(parent);
    } else {
        if (!dom_fd_allowed(c, e->args[0])) { rc = -EBADF; goto out; }
        f = fget(e->args[0]);
        if (!f) { rc = -EBADF; goto out; }
        *parent = f->f_path; path_get(parent); fput(f);
        if (!S_ISDIR(d_inode(parent->dentry)->i_mode)) { rc = -ENOTDIR; goto put; }
    }
    while ((part = strsep(&at, "/")) != NULL) {
        if (!strcmp(part, "..")) { rc = -EACCES; goto put; }
        if (!at) {
            *leaf = *part ? part : ".";
            *storage = name;
            return 0;
        }
        if (!*part || !strcmp(part, ".")) continue;
        f = file_open_root(parent, part, O_PATH | O_NOFOLLOW | O_DIRECTORY, 0);
        if (IS_ERR(f)) { rc = PTR_ERR(f); goto put; }
        if (f->f_path.mnt != d->root.mnt || !dom_disk_inode(file_inode(f))) rc = -EACCES;
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

static struct file *dom_lookup(struct context *c, const struct path *parent, const char *leaf)
{
    struct file *f = file_open_root(parent, leaf, O_PATH | O_NOFOLLOW, 0);
    if (!IS_ERR(f) && (f->f_path.mnt != c->dom_delegate->root.mnt ||
                      !dom_disk_inode(file_inode(f)))) {
        fput(f); return ERR_PTR(-EACCES);
    }
    return f;
}

static struct file *dom_path_file(struct context *c, const struct capstone_delegate_entry *e)
{
    struct path parent;
    struct file *f;
    char *storage, *leaf;
    int rc = dom_parent(c, e, &parent, &storage, &leaf);
    if (rc) return ERR_PTR(rc);
    f = dom_lookup(c, &parent, leaf);
    path_put(&parent); kfree(storage);
    return f;
}

static long dom_path_operation(struct context *c, const struct capstone_delegate_entry *e)
{
    struct path parent;
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
    rc = dom_parent(c, e, &parent, &storage, &leaf);
    if (rc) return rc;
    if (e->nr == CAPSTONE_SYS_unlinkat) {
        struct dentry *victim;
        rc = mnt_want_write(parent.mnt);
        if (rc) goto out;
        inode_lock(d_inode(parent.dentry));
        victim = lookup_one_len(leaf, parent.dentry, strlen(leaf));
        if (IS_ERR(victim)) rc = PTR_ERR(victim);
        else {
            rc = vfs_unlink(mnt_user_ns(parent.mnt), d_inode(parent.dentry), victim, NULL);
            dput(victim);
        }
        inode_unlock(d_inode(parent.dentry));
        mnt_drop_write(parent.mnt);
        goto out;
    }
    pin = dom_lookup(c, &parent, leaf);
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
        if (!IS_ERR(f) && (flags & O_TRUNC)) {
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
