/* SPDX-License-Identifier: GPL-2.0-only */
/* Native process services. The kernel owns bootstrap/exec/reaping; this
 * frontend snapshots requests and carries only explicitly admitted resources. */
#ifdef CV_TASK_BACKEND
#include <linux/capstone_native_spawn.h>
#include <linux/resource.h>
#include <capstone/spawn.h>
#include "spawn-wire.c"

struct native_spawn_request {
    struct context *shadow; /* resource-only context; never a protected context */
    void *block;
    char *argv[CAPSTONE_SPAWN_STRINGS + 1];
    char *envp[CAPSTONE_SPAWN_STRINGS + 1];
    const char *paths[CAPSTONE_SPAWN_ACTIONS];
    struct capstone_spawn_view view;
};

static void native_spawn_release(void *data)
{
    struct native_spawn_request *r = data;
    if (r->shadow) {
        dom_delegate_free(r->shadow);
        kfree(r->shadow);
    }
    kvfree(r->block);
    kfree(r);
}

static int native_spawn_clone_resources(struct native_spawn_request *r, struct context *c)
{
    struct dom_delegate *src = c->dom_delegate, *d;
    r->shadow = kzalloc(sizeof(*r->shadow), GFP_KERNEL);
    if (!r->shadow) return -ENOMEM;
    d = kzalloc(sizeof(*d), GFP_KERNEL);
    if (!d) return -ENOMEM;
    r->shadow->dom_delegate = d;
    d->root = src->root; path_get(&d->root);
    d->cwd_path = src->cwd_path; path_get(&d->cwd_path);
    for (unsigned i = 0; i < DOM_EXTRA_DIRS; ++i)
        if (src->dirs[i].dentry) { d->dirs[i] = src->dirs[i]; path_get(&d->dirs[i]); }
    for (unsigned i = 0; i < DOM_DEVICES; ++i)
        if (src->devices[i].dentry) { d->devices[i] = src->devices[i]; path_get(&d->devices[i]); }
    if (src->shell) d->shell = get_file(src->shell);
    bitmap_copy(d->fds, src->fds, DOM_FDS);
    d->root_name = kstrdup(src->root_name, GFP_KERNEL);
    return d->root_name ? 0 : -ENOMEM;
}

/* Called only by native binfmt admission in the authenticated bootstrap.
 * Transfer ownership once; exec failure cleanup uses the normal context path. */
static int dom_spawn_resources(struct context *c)
{
    struct native_spawn_request *r = capstone_native_spawn_data();
    struct dom_delegate *d;
    if (!r) return 0;
    if (!native_task || !r->shadow || !r->shadow->dom_delegate) return -EPERM;
    d = r->shadow->dom_delegate;
    r->shadow->dom_delegate = NULL;
    c->dom_delegate = d;
    for (unsigned i = 0; i < DOM_FDS; ++i) {
        struct file *f;
        if (!test_bit(i, d->fds)) continue;
        f = fget(i);
        if (f) fput(f);
        else clear_bit(i, d->fds); /* exec has applied CLOEXEC */
    }
    return 1;
}

static int native_spawn_dup(struct context *c, unsigned src, unsigned dst, unsigned flags)
{
    struct file *f;
    int rc;
    if (dst >= DOM_FDS || !dom_fd_allowed(c, src)) return -EBADF;
    f = fget(src);
    if (!f) return -EBADF;
    if (src == dst) { set_close_on_exec(dst, flags & O_CLOEXEC); rc = dst; }
    else rc = replace_fd(dst, f, flags);
    fput(f);
    if (rc >= 0) set_bit(dst, c->dom_delegate->fds);
    return rc < 0 ? rc : 0;
}

static int native_spawn_exec(void *data)
{
    struct native_spawn_request *r = data;
    struct context *c = r->shadow;
    struct dom_delegate *d = c->dom_delegate;
    struct capstone_delegate_entry e = {.nr = CAPSTONE_SYS_openat};
    struct file *image;
    int rc;
    /* Child files/fs are private. No unadmitted inherited descriptor may be
     * usable by an action, survive exec, or prolong an ambient pipe lifetime. */
    __close_range(DOM_FDS, ~0U, 0);
    for (unsigned i = 0; i < DOM_FDS; ++i)
        if (!test_bit(i, d->fds)) close_fd(i);
    set_fs_pwd(current->fs, &d->cwd_path);
    if (r->view.flags & CAPSTONE_SPAWN_SETSIGDEF) {
        for (unsigned sig = 1; sig <= 64; ++sig) {
            if (sig == SIGKILL || sig == SIGSTOP || !(r->view.sigdefault & (1ULL << (sig - 1)))) continue;
            rc = capstone_native_signal_action(sig, 0, 0);
            if (rc) return rc;
        }
    }
    if (r->view.flags & CAPSTONE_SPAWN_SETSIGMASK) {
        sigset_t mask;
        sigemptyset(&mask);
        mask.sig[0] = r->view.sigmask;
        sigdelsetmask(&mask, sigmask(SIGKILL) | sigmask(SIGSTOP));
        set_current_blocked(&mask);
    }
    for (unsigned i = 0; i < r->view.actions; ++i) {
        struct capstone_spawn_action a;
        memcpy(&a, &r->view.action[i], sizeof(a)); /* wire may be unaligned */
        if (a.fd >= DOM_FDS || a.srcfd >= DOM_FDS) return -EBADF;
        switch (a.cmd) {
        case CAPSTONE_SPAWN_CLOSE:
            if (!dom_fd_allowed(c, a.fd)) return -EBADF;
            rc = close_fd(a.fd);
            if (!rc) clear_bit(a.fd, d->fds);
            break;
        case CAPSTONE_SPAWN_DUP2:
            rc = native_spawn_dup(c, a.srcfd, a.fd, 0);
            break;
        case CAPSTONE_SPAWN_OPEN: {
            long fd;
            /* POSIX open actions close the destination before opening. */
            if (dom_fd_allowed(c, a.fd)) { close_fd(a.fd); clear_bit(a.fd, d->fds); }
            e.args[0] = (u64)(s64)AT_FDCWD;
            e.args[2] = a.oflag; e.args[3] = a.mode;
            fd = dom_path_operation_name(c, &e, r->paths[i]);
            if (fd < 0) return fd;
            rc = fd == a.fd ? 0 : native_spawn_dup(c, fd, a.fd, a.oflag & O_CLOEXEC);
            if (fd != a.fd) { close_fd(fd); clear_bit(fd, d->fds); }
            break;
        }
        case CAPSTONE_SPAWN_CHDIR:
            rc = dom_chdir_name(c, r->paths[i]);
            break;
        case CAPSTONE_SPAWN_FCHDIR: {
            struct file *f;
            struct path old;
            if (!dom_fd_allowed(c, a.fd)) return -EBADF;
            f = fget(a.fd);
            if (!f) return -EBADF;
            rc = !S_ISDIR(file_inode(f)->i_mode) ? -ENOTDIR :
                 !dom_directory_grant(d, &f->f_path) ? -EACCES :
                 inode_permission(cv_mnt_idmap(f->f_path.mnt), file_inode(f), MAY_EXEC | MAY_CHDIR);
            if (!rc) {
                old = d->cwd_path; d->cwd_path = f->f_path; path_get(&d->cwd_path);
                set_fs_pwd(current->fs, &d->cwd_path); path_put(&old);
            }
            fput(f);
            break;
        }
        default: return -EINVAL;
        }
        if (rc) return rc;
    }
    e.args[0] = (u64)(s64)AT_FDCWD;
    if (!strcmp(r->view.path, "/bin/sh") && d->shell)
        image = get_file(d->shell);
    else image = dom_path_file_name(c, &e, r->view.path);
    if (IS_ERR(image)) return PTR_ERR(image);
    rc = capstone_native_exec_file(image, (const char *const *)r->argv,
                                    (const char *const *)r->envp);
    fput(image);
    return rc;
}

static const struct capstone_native_spawn_ops native_spawn_ops = {
    .exec = native_spawn_exec, .release = native_spawn_release,
};

static long native_spawn_call(struct context *c, const struct cn_capref *ref, u64 offset, u64 bytes)
{
    struct native_spawn_request *r;
    struct cu_result copied;
    struct pid *pid = NULL;
    unsigned slot;
    long rc;
    if (!native_task || current->seccomp.mode != SECCOMP_MODE_DISABLED) return -EOPNOTSUPP;
    if (bytes < sizeof(struct capstone_spawn_header) || bytes > CAPSTONE_SPAWN_BYTES) return -EINVAL;
    /* SIGCHLD=SIG_IGN/SA_NOCLDWAIT may have autoreaped earlier children.
     * Drop only detached PID objects; live tasks and zombies keep their slot. */
    for (slot = 0; slot < DOM_CHILDREN; ++slot) {
        struct pid *old = c->dom_delegate->children[slot];
        bool detached;
        if (!old) continue;
        rcu_read_lock();
        detached = !pid_task(old, PIDTYPE_PID);
        rcu_read_unlock();
        if (detached) { c->dom_delegate->children[slot] = NULL; put_pid(old); }
    }
    for (slot = 0; slot < DOM_CHILDREN; ++slot)
        if (!c->dom_delegate->children[slot]) break;
    if (slot == DOM_CHILDREN) return -EAGAIN;
    r = kzalloc(sizeof(*r), GFP_KERNEL);
    if (!r) return -ENOMEM;
    r->block = kvmalloc(bytes, GFP_KERNEL);
    if (!r->block) { rc = -ENOMEM; goto fail; }
    copied = cv_checked_copy(c, c->main_thread, ref, offset, r->block, bytes, CU_FROM_DOMAIN);
    if (copied.cause || copied.bytes_done != bytes) { rc = -EFAULT; goto fail; }
    rc = -capstone_spawn_unpack(r->block, bytes, r->argv, ARRAY_SIZE(r->argv),
             r->envp, ARRAY_SIZE(r->envp), r->paths, ARRAY_SIZE(r->paths), &r->view);
    if (rc) goto fail;
    if (!r->view.argc) { rc = -EINVAL; goto fail; }
    if (r->view.flags & ~(CAPSTONE_SPAWN_SETSIGDEF | CAPSTONE_SPAWN_SETSIGMASK)) {
        rc = -EOPNOTSUPP; goto fail;
    }
    rc = native_spawn_clone_resources(r, c);
    if (rc) goto fail;
    mutex_unlock(&vm_lock);
    rc = capstone_native_spawn(&native_spawn_ops, r, &pid); /* consumes r */
    mutex_lock(&vm_lock);
    if (rc > 0) c->dom_delegate->children[slot] = pid;
    return rc;
fail:
    native_spawn_release(r);
    return rc;
}

static void native_spawn_cancel(struct context *c, long number)
{
    for (unsigned i = 0; i < DOM_CHILDREN; ++i) {
        struct pid *pid = c->dom_delegate->children[i];
        if (!pid || pid_vnr(pid) != number) continue;
        c->dom_delegate->children[i] = NULL;
        mutex_unlock(&vm_lock);
        capstone_native_cancel(pid);
        put_pid(pid);
        mutex_lock(&vm_lock);
        return;
    }
}

static long native_wait_call(struct context *c, s64 number, u64 options,
                              const struct cn_capref *status_ref, u64 status_offset,
                              const struct cn_capref *usage_ref, u64 usage_offset)
{
    struct rusage usage = {};
    int status = 0;
    struct pid *pid;
    struct cu_result copied;
    unsigned slot;
    long rc;
    if (!native_task) return -EOPNOTSUPP;
    if ((number <= 0 && number != -1) || number > INT_MAX) return -ECHILD;
    if (options & ~(u64)WNOHANG) return -EINVAL;
    for (slot = 0; slot < DOM_CHILDREN; ++slot) {
        pid = c->dom_delegate->children[slot];
        if (pid && (number == -1 || pid_vnr(pid) == number)) break;
    }
    if (slot == DOM_CHILDREN) return -ECHILD;
    if (status_ref && (rc = cn_authorize(c->native_context, &c->native_context->owner,
                                        status_ref, status_offset, sizeof(status), 1))) return rc;
    if (usage_ref && (rc = cn_authorize(c->native_context, &c->native_context->owner,
                                       usage_ref, usage_offset, sizeof(usage), 1))) return rc;
    if (number == -1) {
        DEFINE_WAIT_FUNC(wait, woken_wake_function);
        u64 gone = 0;
        mutex_unlock(&vm_lock);
        add_wait_queue(&current->signal->wait_chldexit, &wait);
        for (;;) {
            bool alive = false;
            rc = 0;
            for (slot = 0; slot < DOM_CHILDREN; ++slot) {
                pid = c->dom_delegate->children[slot];
                if (!pid || (gone & (1ULL << slot))) continue;
                rc = capstone_native_wait(pid, WNOHANG, &status, &usage);
                if (rc == -ECHILD) { gone |= 1ULL << slot; rc = 0; continue; }
                alive = true;
                if (rc) break;
            }
            if (rc || !alive || (options & WNOHANG)) {
                if (!rc && !alive) rc = -ECHILD;
                break;
            }
            if (signal_pending(current)) { rc = -EINTR; break; }
            wait_woken(&wait, TASK_INTERRUPTIBLE, MAX_SCHEDULE_TIMEOUT);
        }
        remove_wait_queue(&current->signal->wait_chldexit, &wait);
        mutex_lock(&vm_lock);
        if (rc > 0) gone |= 1ULL << slot;
        for (slot = 0; slot < DOM_CHILDREN; ++slot) if (gone & (1ULL << slot)) {
            put_pid(c->dom_delegate->children[slot]);
            c->dom_delegate->children[slot] = NULL;
        }
    } else {
        mutex_unlock(&vm_lock);
        rc = capstone_native_wait(pid, options, &status, &usage);
        mutex_lock(&vm_lock);
        if (rc > 0 || rc == -ECHILD) {
            c->dom_delegate->children[slot] = NULL;
            put_pid(pid);
        }
    }
    if (rc <= 0) return rc;
    /* As Linux wait4, reap precedes copy-out: an output fault consumes status. */
    if (status_ref) {
        copied = cv_checked_copy(c, c->main_thread, status_ref, status_offset, &status, sizeof(status), CU_TO_DOMAIN);
        if (copied.cause || copied.bytes_done != sizeof(status)) return -EFAULT;
    }
    if (usage_ref) {
        copied = cv_checked_copy(c, c->main_thread, usage_ref, usage_offset, &usage, sizeof(usage), CU_TO_DOMAIN);
        if (copied.cause || copied.bytes_done != sizeof(usage)) return -EFAULT;
    }
    return rc;
}
#else
static int dom_spawn_resources(struct context *c) { return 0; }
#endif
