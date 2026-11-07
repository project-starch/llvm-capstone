/* SPDX-License-Identifier: GPL-2.0-only */
/* Preliminary native workload coverage. No userspace syscall fallback. */
#include <linux/stat.h>
#include <linux/resource.h>
#include <linux/sysinfo.h>
#include <asm/stat.h>

static unsigned long native_refused_calls;
static bool native_refused_trace;
module_param(native_refused_calls, ulong, 0444);
module_param(native_refused_trace, bool, 0444);
static long native_refused(u64 nr)
{
    ++native_refused_calls;
    if (native_refused_trace) pr_info("native refused nr=%llu\n", nr);
    return -ENOSYS;
}

/* Both transports use one path-policy and stat-layout provider. vm_lock held. */
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

static bool native_gcc_issue(struct context *c, const struct ns_invocation *call,
                             struct file *held, long *result)
{
    const u64 *a = call->scalar;
    u64 nr = call->descriptor->nr;
    long rc = -ENOSYS;
    struct capstone_delegate_entry e = {0};
    sigset_t old, set;
    switch (nr) {
    case CAPSTONE_SYS_openat:
    case CAPSTONE_SYS_unlinkat:
        e.nr = nr; memcpy(e.args, a, sizeof(e.args));
        mutex_unlock(&vm_lock);
        rc = dom_path_operation_name(c, &e, call->buffer[1].data);
        mutex_lock(&vm_lock);
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

/* Exchange ABI vectors contain offsets, never serialized capability authority.
 * Snapshot the wire array once; all payload references retain the authenticated
 * exchange capability and request epoch. One Linux vector operation preserves
 * pipe atomicity and file-position semantics. */
static long native_gcc_vector(struct context *c, const struct capstone_delegate_entry *e,
                               const struct cn_capref *exchange)
{
    struct iovec *wire = NULL;
    struct kvec *vec = NULL;
    struct file *file = NULL;
    struct iov_iter iter;
    struct cu_result copied;
    struct cn_owner owner = {c->main_thread->id, (unsigned long)c->mm, c->table_pa, 1};
    unsigned long i, count = e->args[2];
    size_t total = 0, left;
    bool write = e->nr == CAPSTONE_SYS_writev || e->nr == CAPSTONE_SYS_pwritev;
    bool positional = e->nr == CAPSTONE_SYS_preadv || e->nr == CAPSTONE_SYS_pwritev;
    bool locked = false;
    loff_t pos;
    long rc;
    rc = ns_session_admit(c->native_session, exchange->context_epoch);
    if (rc) return rc;
    /* Positional vectors need their split-offset ABI separately qualified. */
    if (positional) return -ENOSYS;
    if (e->flags != 2 || count > 1024) return -EINVAL;
    if (e->args[0] > INT_MAX || !dom_fd_allowed(c, e->args[0])) return -EBADF;
    file = fget(e->args[0]);
    if (!file || file->private_data == c) { rc = -EBADF; goto out; }
    if (!(file->f_mode & (write ? FMODE_WRITE : FMODE_READ))) { rc = -EBADF; goto out; }
    if (!(file->f_mode & (write ? FMODE_CAN_WRITE : FMODE_CAN_READ))) { rc = -EINVAL; goto out; }
    wire = kcalloc(count ?: 1, sizeof(*wire), GFP_KERNEL);
    vec = kcalloc(count ?: 1, sizeof(*vec), GFP_KERNEL);
    if (!wire || !vec) { rc = -ENOMEM; goto out; }
    copied = cv_checked_copy(c, c->main_thread, exchange, e->args[1], wire,
                             count * sizeof(*wire), CU_FROM_DOMAIN);
    if (copied.cause) { rc = -EFAULT; goto out; }
    for (i = 0; i < count; ++i) {
        size_t bytes = wire[i].iov_len;
        unsigned long offset = (unsigned long)wire[i].iov_base;
        if (bytes > NS_MAX_BYTES - total) { rc = -E2BIG; goto out; }
        total += bytes;
        if (!dom_within(offset, bytes, c->dom_exchange_bytes) ||
            cn_authorize(c->native_context, &owner, exchange, offset, bytes, !write)) {
            rc = -EFAULT; goto out;
        }
        vec[i].iov_len = bytes;
        vec[i].iov_base = kvzalloc(bytes ?: 1, GFP_KERNEL);
        if (!vec[i].iov_base) { rc = -ENOMEM; goto out; }
        if (write && bytes) {
            copied = cv_checked_copy(c, c->main_thread, exchange, offset,
                                     vec[i].iov_base, bytes, CU_FROM_DOMAIN);
            if (copied.cause) { rc = -EFAULT; goto out; }
        }
    }
    iov_iter_kvec(&iter, write ? WRITE : READ, vec, count, total);
    mutex_unlock(&vm_lock);
    if (file->f_mode & FMODE_ATOMIC_POS) { mutex_lock(&file->f_pos_lock); locked = true; }
    pos = file->f_pos;
    if (write) file_start_write(file);
    rc = write ? vfs_iter_write(file, &iter, &pos, 0) : vfs_iter_read(file, &iter, &pos, 0);
    if (write) file_end_write(file);
    if (rc >= 0 && !(file->f_mode & FMODE_STREAM)) file->f_pos = pos;
    if (locked) mutex_unlock(&file->f_pos_lock);
    mutex_lock(&vm_lock);
    if (rc <= -ERESTARTSYS && rc >= -ERESTART_RESTARTBLOCK) rc = -EINTR;
    if (rc > 0 && !write) {
        left = rc;
        for (i = 0; i < count && left; ++i) {
            size_t bytes = min(left, vec[i].iov_len);
            copied = cv_checked_copy(c, c->main_thread, exchange,
                        (unsigned long)wire[i].iov_base, vec[i].iov_base, bytes, CU_TO_DOMAIN);
            if (copied.cause) {
                size_t published = (size_t)rc - left + copied.bytes_done;
                rc = published ? (long)published : -EFAULT;
                break;
            }
            left -= bytes;
        }
    }
out:
    if (vec) for (i = 0; i < count; ++i) kvfree(vec[i].iov_base);
    kfree(vec); kfree(wire);
    if (file) fput(file);
    return rc;
}
