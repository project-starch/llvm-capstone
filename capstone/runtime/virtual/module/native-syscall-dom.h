/* SPDX-License-Identifier: GPL-2.0-only */
/* The issue side of the system calls the kernel implements for a protected program: the
 * allow-list, the descriptor admission and the Linux operations on kernel buffers. Plan v5
 * step 5 binds them to the trap route, where a binder copies each argument through the
 * program's reference before sublet_syscall_issue and back after it. Until then nothing
 * calls them. No program address reaches a Linux API. */
#include <linux/seccomp.h>
#include <linux/timekeeping.h>
#include <capstone/native-syscall.h>
#include "native-syscall-table.c"

/* vm_lock held; preserve Linux's shared open-file-description position lock. */
static long native_seek_issue(struct file *file, u64 offset, u64 whence)
{
    long rc;
    /* Reject malformed 64-bit selectors before narrowing to Linux's type. */
    if (whence > SEEK_MAX) return -EINVAL;
    mutex_unlock(&vm_lock);
    if (file->f_mode & FMODE_ATOMIC_POS) mutex_lock(&file->f_pos_lock);
    rc = vfs_llseek(file, (s64)offset, whence);
    if (rc <= -ERESTARTSYS && rc >= -ERESTART_RESTARTBLOCK) rc = -EINTR;
    if (file->f_mode & FMODE_ATOMIC_POS) mutex_unlock(&file->f_pos_lock);
    mutex_lock(&vm_lock);
    return rc;
}
static bool native_supported(u64 nr)
{
    switch (nr) {
    case CAPSTONE_SYS_read: case CAPSTONE_SYS_write:
    case CAPSTONE_SYS_pread64: case CAPSTONE_SYS_pwrite64:
    case CAPSTONE_SYS_getdents64:
    case CAPSTONE_SYS_close: case CAPSTONE_SYS_lseek:
    case CAPSTONE_SYS_dup: case CAPSTONE_SYS_dup3:
    case CAPSTONE_NR_FCNTL_LOCK: case CAPSTONE_SYS_fcntl:
    case CAPSTONE_SYS_fsync: case CAPSTONE_SYS_fdatasync:
    case CAPSTONE_SYS_ftruncate: case CAPSTONE_SYS_unlinkat:
    case CAPSTONE_SYS_fchmod: case CAPSTONE_SYS_fchown:
    case CAPSTONE_SYS_getpid: case CAPSTONE_SYS_getppid: case CAPSTONE_SYS_gettid:
    case CAPSTONE_SYS_getuid: case CAPSTONE_SYS_geteuid:
    case CAPSTONE_SYS_getgid: case CAPSTONE_SYS_getegid:
    case CAPSTONE_SYS_clock_gettime:
    case CAPSTONE_SYS_sched_yield:
    case CAPSTONE_SYS_nanosleep: case CAPSTONE_SYS_clock_nanosleep: case CAPSTONE_SYS_pselect6:
    case CAPSTONE_SYS_mkdirat: case CAPSTONE_SYS_fchmodat: case CAPSTONE_SYS_symlinkat:
    case CAPSTONE_SYS_readlinkat: case CAPSTONE_SYS_truncate: case CAPSTONE_SYS_linkat:
    case CAPSTONE_SYS_openat: case CAPSTONE_SYS_newfstatat: case CAPSTONE_SYS_fstat:
    case CAPSTONE_SYS_sysinfo: case CAPSTONE_SYS_prlimit64:
    case CAPSTONE_NR_SIGACTION:
    case CAPSTONE_SYS_rt_sigprocmask: case CAPSTONE_SYS_getcwd: case CAPSTONE_SYS_chdir:
        return true;
    default: return false;
    }
}
static int native_acquire(void *opaque, const struct ns_invocation *call, void **held)
{
    struct context *c = opaque;
    struct file *file;
    u64 nr = call->descriptor->nr;
    /* There is no module API for inherited filter evaluation with the domain's
     * register ABI. Refuse filtered tasks rather than skip their policy. */
    if (current->seccomp.mode != SECCOMP_MODE_DISABLED) return -EOPNOTSUPP;
    if (!native_supported(nr)) return -ENOSYS;
    if (nr == CAPSTONE_SYS_openat || nr == CAPSTONE_SYS_newfstatat || nr == CAPSTONE_SYS_unlinkat ||
        nr == CAPSTONE_SYS_mkdirat || nr == CAPSTONE_SYS_fchmodat || nr == CAPSTONE_SYS_symlinkat ||
        nr == CAPSTONE_SYS_readlinkat || nr == CAPSTONE_SYS_truncate || nr == CAPSTONE_SYS_linkat) return 0;
    if (!call->descriptor->fd_mask) return 0;
    if (call->scalar[0] > INT_MAX || !dom_fd_allowed(c, call->scalar[0])) return -EBADF;
    file = fget(call->scalar[0]);
    if (!file) return -EBADF;
    if (file->private_data == c || !file->f_op) { fput(file); return -EBADF; }
    *held = file;
    return 0;
}
/* Value of a call without arguments that Linux answers from the task alone, or false. */
static bool native_task_value(u64 nr, long *value)
{
    switch (nr) {
    case CAPSTONE_SYS_getpid: *value = task_tgid_vnr(current); return true;
    case CAPSTONE_SYS_getppid: *value = task_ppid_nr_ns(current, task_active_pid_ns(current)); return true;
    case CAPSTONE_SYS_gettid: *value = task_pid_vnr(current); return true;
    case CAPSTONE_SYS_getuid: *value = from_kuid_munged(current_user_ns(), current_uid()); return true;
    case CAPSTONE_SYS_geteuid: *value = from_kuid_munged(current_user_ns(), current_euid()); return true;
    case CAPSTONE_SYS_getgid: *value = from_kgid_munged(current_user_ns(), current_gid()); return true;
    case CAPSTONE_SYS_getegid: *value = from_kgid_munged(current_user_ns(), current_egid()); return true;
    default: return false;
    }
}
/* read, write, pread64, pwrite64 on a held file and a kernel buffer; vm_lock is dropped. */
static long native_rw(struct file *file, u64 nr, u64 offset, void *data, size_t length)
{
    bool positional = nr == CAPSTONE_SYS_pread64 || nr == CAPSTONE_SYS_pwrite64, locked = false;
    bool read = nr == CAPSTONE_SYS_read || nr == CAPSTONE_SYS_pread64;
    loff_t pos, *ppos;
    long rc;
    if (positional && (s64)offset < 0) return -EINVAL;
    if (positional && !(file->f_mode & (nr == CAPSTONE_SYS_pread64 ? FMODE_PREAD : FMODE_PWRITE)))
        return -ESPIPE;
    /* The descriptor's own open mode decides, as in Linux: kernel_write() warns (and a
     * panic_on_warn kernel stops) when it is handed a file that was not opened for writing. */
    if (!(file->f_mode & (read ? FMODE_READ : FMODE_WRITE))) return -EBADF;
    if (!(file->f_mode & (read ? FMODE_CAN_READ : FMODE_CAN_WRITE))) return -EINVAL;
    mutex_unlock(&vm_lock);
    if (!positional && (file->f_mode & FMODE_ATOMIC_POS)) {
        mutex_lock(&file->f_pos_lock); locked = true;
    }
    pos = positional ? offset : file->f_pos;
    ppos = !positional && (file->f_mode & FMODE_STREAM) ? NULL : &pos;
    if (read) rc = kernel_read(file, data, length, ppos);
    else {
        file_start_write(file);
        rc = kernel_write(file, data, length, ppos);
        file_end_write(file);
    }
    if (!positional && ppos && rc >= 0) file->f_pos = pos;
    if (locked) mutex_unlock(&file->f_pos_lock);
    mutex_lock(&vm_lock);
    return rc;
}
#include "native-gcc-dom.h"
#include "native-files-dom.h"
#include "native-directory-dom.h"
#include "native-time-dom.h"

static struct ns_completion native_issue(void *opaque, const struct ns_invocation *call,
                                          void *held)
{
    struct context *c = opaque;
    struct file *file = held;
    const u64 *a = call->scalar;
    u64 nr = call->descriptor->nr;
    long rc = -ENOSYS;
    struct timespec64 ts;
    if (native_task_value(nr, &rc)) goto done;
    if (native_directory_issue(c, call, file, &rc)) goto done;
    if (native_time_issue(c, call, &rc)) goto done;
    if (native_gcc_issue(c, call, file, &rc)) goto done;
    if (native_files_issue(c, call, file, &rc)) goto done;
    switch (nr) {
    case CAPSTONE_SYS_dup: {
        mutex_unlock(&vm_lock);
        rc = capstone_native_dupfd(0, file, 0, DOM_FDS);
        mutex_lock(&vm_lock);
        if (rc >= 0) set_bit(rc, c->dom_delegate->fds);
        break;
    }
    case CAPSTONE_SYS_dup3: {
        struct file *dst;
        if (a[0] == a[1] || (a[2] & ~(u64)O_CLOEXEC)) { rc = -EINVAL; break; }
        if (a[1] >= DOM_FDS) { rc = -EBADF; break; }
        dst = fget(a[1]);
        if (dst) {
            fput(dst);
            if (!dom_fd_allowed(c, a[1])) { rc = -EBADF; break; }
        }
        mutex_unlock(&vm_lock);
        rc = replace_fd(a[1], file, a[2]);
        mutex_lock(&vm_lock);
        if (rc >= 0) {
            set_bit(a[1], c->dom_delegate->fds);
            rc = a[1];
        }
        break;
    }
    case CAPSTONE_NR_SIGACTION:
        if (!a[0] || a[0] > 64 || a[1] > CAPSTONE_SIGNAL_CAUGHT || a[2] > UINT_MAX) {
            rc = -EINVAL; break;
        }
        mutex_unlock(&vm_lock);
        rc = capstone_native_signal_action(a[0], a[1], a[2]);
        mutex_lock(&vm_lock);
        break;
    case CAPSTONE_SYS_sched_yield:
        mutex_unlock(&vm_lock); yield(); mutex_lock(&vm_lock); rc = 0; break;
    case CAPSTONE_SYS_clock_gettime:
        BUILD_BUG_ON(sizeof(struct timespec64) != 16);
        BUILD_BUG_ON(offsetof(struct timespec64, tv_sec) != 0);
        BUILD_BUG_ON(offsetof(struct timespec64, tv_nsec) != 8);
        if (a[0] != CLOCK_REALTIME && a[0] != CLOCK_MONOTONIC) { rc = -EINVAL; break; }
        if (a[0] == CLOCK_REALTIME) ktime_get_real_ts64(&ts);
        else ktime_get_ts64(&ts);
        if (call->buffer[1].data) memcpy(call->buffer[1].data, &ts, sizeof(ts));
        rc = 0; break;
    case CAPSTONE_SYS_close:
        rc = native_close_issue(c, a[0]);
        break;
    case CAPSTONE_SYS_lseek:
        rc = native_seek_issue(file, a[1], a[2]);
        break;
    case CAPSTONE_SYS_read: case CAPSTONE_SYS_write:
    case CAPSTONE_SYS_pread64: case CAPSTONE_SYS_pwrite64:
        rc = native_rw(file, nr, a[3], call->buffer[1].data, call->buffer[1].length);
        break;
    }
done:
    if (rc <= -ERESTARTSYS && rc >= -ERESTART_RESTARTBLOCK) rc = -EINTR;
    /* No automatic replay: Linux may already have consumed input. */
    return (struct ns_completion){rc, 1, 0};
}
static void native_release(void *opaque, void *held)
{ (void)opaque; if (held) fput(held); }


/* One call on kernel buffers: admit the descriptor, issue, release. Caller holds vm_lock,
 * which the Linux operations drop while they block. */
long sublet_syscall_issue(struct context *c, const struct ns_invocation *call)
{
    void *held = NULL;
    long rc = native_acquire(c, call, &held);

    if (rc) return rc;
    rc = native_issue(c, call, held).value;
    native_release(c, held);
    return rc;
}

#include "native-spawn-dom.h"
