/* SPDX-License-Identifier: GPL-2.0-only */
/* Path, metadata, limit, signal mask and directory calls on kernel buffers. */
#include <linux/stat.h>
#include <linux/resource.h>
#include <linux/sysinfo.h>
#include <asm/stat.h>

/* One path policy and stat layout for both stat calls. vm_lock held. */
static long native_stat_issue(struct context *c, u64 nr, const u64 *args,
                              struct file *held, const char *name, struct stat *out)
{
    struct capstone_delegate_entry e = {0};
    struct kstat st;
    struct file *file;
    long rc;
    if (nr == CAPSTONE_SYS_newfstatat &&
        (args[3] & ~(u64)(AT_SYMLINK_NOFOLLOW | AT_NO_AUTOMOUNT))) return -EINVAL;
    mutex_unlock(&vm_lock);
    if (nr == CAPSTONE_SYS_newfstatat) {
        e.nr = nr; memcpy(e.args, args, sizeof(e.args));
        file = dom_path_file_name(c, &e, name);
    } else { file = held; get_file(file); }
    if (IS_ERR(file)) rc = PTR_ERR(file);
    else {
        rc = vfs_getattr(&file->f_path, &st, STATX_BASIC_STATS, AT_STATX_SYNC_AS_STAT);
        if (!rc) rc = capstone_native_stat(&st, out);
        fput(file);
    }
    mutex_lock(&vm_lock);
    return rc;
}

/* Ordinary unlink and open share the confined resolver and fd publication.
 * vm_lock held; a failed open can already have created or truncated a file. */
static long native_path_issue(struct context *c, u64 nr, const u64 *args,
                              const char *name)
{
    struct capstone_delegate_entry e = {0};
    long rc;
    e.nr = nr; memcpy(e.args, args, sizeof(e.args));
    mutex_unlock(&vm_lock);
    rc = dom_path_operation_name(c, &e, name);
    mutex_lock(&vm_lock);
    if (rc <= -ERESTARTSYS && rc >= -ERESTART_RESTARTBLOCK) rc = -EINTR;
    return rc;
}

static bool native_gcc_issue(struct context *c, const struct ns_invocation *call,
                             struct file *held, long *result)
{
    const u64 *a = call->scalar;
    u64 nr = call->descriptor->nr;
    long rc = -ENOSYS;
    sigset_t old, set;
    switch (nr) {
    case CAPSTONE_SYS_openat:
    case CAPSTONE_SYS_unlinkat:
        rc = native_path_issue(c, nr, a, call->buffer[1].data);
        break;
    case CAPSTONE_SYS_newfstatat:
    case CAPSTONE_SYS_fstat:
        rc = native_stat_issue(c, nr, a, held, call->buffer[1].data,
                               call->buffer[nr == CAPSTONE_SYS_fstat ? 1 : 2].data);
        break;
    case CAPSTONE_SYS_sysinfo:
        rc = capstone_native_sysinfo(call->buffer[0].data);
        break;
    case CAPSTONE_SYS_prlimit64:
        if ((pid_t)a[0] != 0 && (pid_t)a[0] != task_tgid_vnr(current)) { rc = -EPERM; break; }
        /* Same read-only limit policy as the existing launcher/broker. */
        if (call->buffer[2].data) { rc = -EPERM; break; }
        {
            struct rlimit limit;
            rc = capstone_native_getrlimit(a[1], &limit);
            if (!rc && call->buffer[3].data) memcpy(call->buffer[3].data, &limit, sizeof(limit));
        }
        break;
    case CAPSTONE_SYS_rt_sigprocmask:
        if (a[3] != sizeof(sigset_t)) { rc = -EINVAL; break; }
        old = current->blocked;
        rc = 0;
        if (call->buffer[1].data) {
            memcpy(&set, call->buffer[1].data, sizeof(set));
            sigdelsetmask(&set, sigmask(SIGKILL) | sigmask(SIGSTOP));
            rc = sigprocmask(a[0], &set, NULL);
        }
        if (!rc && call->buffer[2].data) memcpy(call->buffer[2].data, &old, sizeof(old));
        break;
    case CAPSTONE_SYS_getcwd:
        mutex_unlock(&vm_lock);
        rc = dom_getcwd(c, call->buffer[0].data, a[1]);
        mutex_lock(&vm_lock);
        break;
    case CAPSTONE_SYS_chdir:
        mutex_unlock(&vm_lock);
        rc = dom_chdir_name(c, call->buffer[0].data);
        mutex_lock(&vm_lock);
        break;
    default: return false;
    }
    *result = rc;
    return true;
}
