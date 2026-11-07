/* SPDX-License-Identifier: GPL-2.0-only */
/* Included after the resource broker. No domain address reaches Linux APIs. */
#include <linux/seccomp.h>
#include <linux/timekeeping.h>
#include <capstone/native-syscall.h>
#include "native-syscall-table.c"
#include "native-syscall-core.c"

/* Per native context, allocated once with its session: the owner it was created for, the
 * bounce buffer and the copy state of the specialised calls. Nothing is allocated per call. */
#define NATIVE_FAST_MAX 65536U
struct cn_context {
    struct context *c;
    struct cv_thread *t;
    struct cn_owner owner;
    void *bounce;
    struct cv_copy_state copy;
};
static bool native_dispatch;
/* Diagnostic-only inventory; disabled in measured runs. */
static bool native_route_trace;
module_param(native_route_trace, bool, 0644);
static void native_route_record(const char *route, u64 nr, u64 count, long result)
{
    if (native_route_trace)
        pr_info("native_route route=%s nr=%llu count=%llu result=%ld\n", route, nr, count, result);
}
module_param(native_dispatch, bool, 0444);
MODULE_PARM_DESC(native_dispatch, "checked in-kernel native syscall loop (default off)");
static bool native_fast = true;
static unsigned long native_fast_hits;
module_param(native_fast_hits, ulong, 0444);
MODULE_PARM_DESC(native_fast_hits, "calls completed by the specialised paths (evidence for the test drivers)");
module_param(native_fast, bool, 0444);
MODULE_PARM_DESC(native_fast, "specialised paths for scalar, read and write calls inside native_dispatch (default on)");

static int native_owner(struct cn_context *n, const struct cn_owner *owner)
{
    if (!n || !owner || n->c->mm != current->mm ||
        n->c->dom_pid != task_pid_nr(current) || n->t != n->c->main_thread ||
        owner->thread_id != n->t->id || owner->address_space_id != (unsigned long)n->c->mm ||
        owner->namespace_id != n->c->table_pa || owner->namespace_epoch != 1 ||
        n->t->user_frame || !n->t->started || n->t->terminal || n->t->event != 3)
        return -EPERM;
    return 0;
}
int cn_get_epoch(struct cn_context *n, const struct cn_owner *owner, cn_u64 *epoch)
{
    int rc = native_owner(n, owner);
    if (rc) return rc;
    *epoch = READ_ONCE(n->t->frame[82]);
    return *epoch ? 0 : -ESTALE;
}
int cn_get_capref(struct cn_context *n, const struct cn_owner *owner,
                  unsigned reg, struct cn_capref *ref)
{
    cn_u64 epoch;
    int rc = cn_get_epoch(n, owner, &epoch);
    if (rc) return rc;
    if (!reg || reg >= 32) return -EINVAL;
    *ref = (struct cn_capref){n->t->id, epoch, reg, 0};
    return 0;
}
int cn_snapshot(struct cn_context *n, const struct cn_owner *owner, struct cn_frame *f)
{
    unsigned i;
    cn_u64 epoch, kinds;
    int rc = cn_get_epoch(n, owner, &epoch);
    if (rc) return rc;
    /* QEMU publishes this classification at the escape into kernel-owned RAM.
     * It is only a shape snapshot. CSCHECK authenticates each actual access. */
    kinds = READ_ONCE(n->t->frame[83]);
    memset(f, 0, sizeof(*f));
    for (i = 10; i <= 17; ++i) {
        f->x[i].payload[0] = n->t->frame[72 + i - 10];
        f->x[i].payload[1] = n->t->frame[8 + 2 * i + 1];
        f->x[i].flags = ((kinds >> i) & 1 ? CN_REG_TAG : 0) |
                       ((kinds >> (i + 32)) & 1 ? CN_REG_RAW_TAG : 0);
    }
    return 0;
}
/* Authorization finishes before copy/issue. The owning native task cannot
 * re-enter this session while scheduled out; slot transport does not dispatch
 * services, and the test racer handles only WAIT/MAP/UNMAP/COLLECT.
 * Keep scratch in the kzalloc-backed cn_context: its descriptor is consumed
 * through virt_to_phys, so neither a VMAP_STACK nor kvzalloc context suffices. */
static int native_authorize_reuse(struct cn_context *n, const struct cn_capref *ref,
                                  u64 offset, u64 bytes, bool write)
{
    struct cv_copy_state *s = &n->copy;
    u64 address;
    u32 cause;
    cv_copy_state_init(s, n->c, n->t, write ? CU_TO_DOMAIN : CU_FROM_DOMAIN, bytes);
    cv_copy_enter(s);
    cause = cv_copy_check(s, ref, offset, bytes,
                          write ? CU_TO_DOMAIN : CU_FROM_DOMAIN, &address);
    cv_copy_drain(s);
    cv_copy_leave(s);
    return cause ? -EFAULT : 0;
}
int cn_authorize(struct cn_context *n, const struct cn_owner *owner,
                  const struct cn_capref *ref, cn_u64 offset, cn_u64 bytes, int write)
{
    int rc = native_owner(n, owner);
    if (rc) return rc;
    return native_authorize_reuse(n, ref, offset, bytes, write != 0);
}
static struct ns_copy_result native_copy(void *opaque, struct cn_context *n,
        const struct cn_owner *owner, const struct cn_capref *ref, cn_u64 offset,
        void *buffer, size_t bytes, enum ns_direction direction)
{
    struct cu_result r = {0, CU_REFUSED};
    (void)opaque;
    /* Preserve the allocating wrapper's task-lifetime check across sleeping
     * copies. This session's synchronous calls cannot overlap n->copy. */
    if (!native_owner(n, owner) && riscv_capstone_owns(n->c))
        r = cv_checked_copy_reuse(&n->copy, n->c, n->t, ref, offset, buffer, bytes,
                                  direction == NS_TO_DOMAIN ? CU_TO_DOMAIN : CU_FROM_DOMAIN);
    return (struct ns_copy_result){r.bytes_done, r.cause};
}
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
#ifdef CV_TASK_BACKEND
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
#endif
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

static int native_session(struct context *c)
{
    static const struct ns_ops ops = {
        .copy = native_copy, .acquire = native_acquire,
        .issue = native_issue, .release = native_release
    };
    struct cn_owner owner = {c->main_thread->id, (unsigned long)c->mm, c->table_pa, 1};
    int rc;
    if (c->native_session) return 0;
    c->native_context = kzalloc(sizeof(*c->native_context), GFP_KERNEL);
    if (!c->native_context) return -ENOMEM;
    c->native_context->c = c; c->native_context->t = c->main_thread; c->native_context->owner = owner;
    rc = ns_session_create(c->native_context, &owner, &ops, c, &c->native_session);
    if (rc) { kfree(c->native_context); c->native_context = NULL; }
    return rc;
}
/* Final context teardown, after synchronous service/copy use has ended.
 * The enclosing context is destroyed next; this is not a reset operation. */
static void native_session_destroy(struct context *c)
{
    ns_session_free(c->native_session);
    if (c->native_context) kfree(c->native_context->bounce);
    kfree(c->native_context);
}
static int native_meta_reply(struct context *c, const struct cn_capref *meta, long value)
{
    u64 words[3] = {value, 0, CAPSTONE_ROUND_DONE};
    u32 zero = 0;
    struct cu_result r = cv_checked_copy(c, c->main_thread, meta,
        offsetof(struct capstone_delegate_entry, result), words, sizeof(words), CU_TO_DOMAIN);
    if (r.cause) return -EFAULT;
    r = cv_checked_copy(c, c->main_thread, meta, CAPSTONE_SIGNAL_OFFSET +
        offsetof(struct capstone_signal_block, count), &zero, sizeof(zero), CU_TO_DOMAIN);
    c->dom_step.result = 0;
    return r.cause ? -EFAULT : 0;
}

/* ---- Specialised paths: the same calls, the same checks and the same results as the shared
 * dispatcher, without its per-call snapshot. They take effect only when every register of the
 * escape has the plain shape of that call; anything else (extra tags, nonzero upper words,
 * unusual sizes, a descriptor that would be refused) returns false BEFORE the epoch is
 * admitted or any effect happens, and the shared dispatcher then decides and reports. ---- */
#define NF_ARGS 0x3fc00ULL  /* x10..x17: a0..a5, the call number, the selector */

static __always_inline bool nf_regs_plain(struct cv_thread *t, u64 kinds, u64 pointer_regs)
{
    const unsigned long *f = t->frame;
    u64 hi = 0;
    if ((kinds >> 32) & NF_ARGS) return false;               /* raw-move cargo */
    if ((kinds & NF_ARGS) != pointer_regs) return false;     /* exactly the expected tags */
    /* Untagged upper words are zero: x10..x17 live at frame[8 + 2 * reg + 1]. */
#define NF_HI(reg) (((pointer_regs >> (reg)) & 1) ? 0 : READ_ONCE(f[8 + 2 * (reg) + 1]))
    hi = NF_HI(10) | NF_HI(11) | NF_HI(12) | NF_HI(13) | NF_HI(14) | NF_HI(15) | NF_HI(16) | NF_HI(17);
#undef NF_HI
    return !hi;
}
#include "native-process-dom.h"
#include "native-spawn-dom.h"

/* Closing is a single admitted operation; even an error must never replay it. */
static bool native_close_fast(struct context *c, struct cv_step *step, u64 kinds)
{
    struct cv_thread *t = c->main_thread;
    struct file *file;
    long rc;
    if (step->args[6] != CAPSTONE_SYS_close || !nf_regs_plain(t, kinds, 0) ||
        step->args[0] > INT_MAX || !dom_fd_allowed(c, step->args[0])) return false;
    file = fget(step->args[0]);
    if (!file) return false;
    if (file->private_data == c || !file->f_op) { fput(file); return false; }
    rc = ns_session_admit(c->native_session, READ_ONCE(t->frame[82]));
    if (rc) rc = -ESTALE;
    else rc = native_close_issue(c, step->args[0]);
    fput(file);
    step->result = rc;
    ++native_fast_hits;
    return true;
}

/* Scalar fd operations retain the same authority and once-only admission. */
static bool native_fd_fast(struct context *c, struct cv_step *step, u64 kinds)
{
    struct cv_thread *t = c->main_thread;
    struct file *file;
    long rc;
    if (!native_fd_supported(step->args[6]) || !nf_regs_plain(t, kinds, 0) ||
        step->args[0] > INT_MAX || !dom_fd_allowed(c, step->args[0])) return false;
    file = fget(step->args[0]);
    if (!file) return false;
    if (file->private_data == c || !file->f_op) { fput(file); return false; }
    rc = ns_session_admit(c->native_session, READ_ONCE(t->frame[82]));
    if (rc) rc = -ESTALE;
    else rc = native_fd_issue(file, step->args[6], step->args[1], step->args[2]);
    fput(file);
    step->result = rc;
    ++native_fast_hits;
    return true;
}

/* Scalar-only service: no capability buffer or request snapshot is needed. */
static bool native_seek_fast(struct context *c, struct cv_step *step, u64 kinds)
{
    struct cv_thread *t = c->main_thread;
    struct file *file;
    long rc;
    if (step->args[6] != CAPSTONE_SYS_lseek || !nf_regs_plain(t, kinds, 0) ||
        step->args[0] > INT_MAX || !dom_fd_allowed(c, step->args[0])) return false;
    file = fget(step->args[0]);
    if (!file) return false;
    if (file->private_data == c || !file->f_op) { fput(file); return false; }
    rc = ns_session_admit(c->native_session, READ_ONCE(t->frame[82]));
    if (rc) rc = -ESTALE;
    else rc = native_seek_issue(file, step->args[1], step->args[2]);
    fput(file);
    step->result = rc;
    ++native_fast_hits;
    return true;
}

/* Snapshot through the first NUL only, including tightly bounded strings.
 * The caller owns *name on every return and has already admitted this epoch. */
static int native_snapshot_name(struct context *c, u64 epoch, char **name)
{
    struct cn_context *n = c->native_context;
    struct cn_capref path = {c->main_thread->id, epoch, 11, 0};
    size_t i;
    *name = kmalloc(NS_MAX_PATH, GFP_KERNEL);
    if (!*name) return -ENOMEM;
    for (i = 0; i < NS_MAX_PATH; ++i) {
        struct ns_copy_result copied = native_copy(c, n, &n->owner, &path,
                                                   i, *name + i, 1, NS_FROM_DOMAIN);
        if (copied.cause || copied.bytes_done != 1) return -EFAULT;
        if (!(*name)[i]) return 0;
    }
    return -ENAMETOOLONG;
}

/* Nonzero unlink flags retain the generic directory-service dispatch. */
static bool native_path_fast(struct context *c, struct cv_step *step, u64 kinds)
{
    struct cv_thread *t = c->main_thread;
    u64 nr = step->args[6], epoch = READ_ONCE(t->frame[82]);
    char *name = NULL;
    long rc;
    if (nr != CAPSTONE_SYS_openat && nr != CAPSTONE_SYS_unlinkat) return false;
    if (nr == CAPSTONE_SYS_unlinkat && step->args[2]) return false;
    if (!nf_regs_plain(t, kinds, 1ULL << 11)) return false;
    rc = ns_session_admit(c->native_session, epoch);
    if (rc) rc = -ESTALE;
    else {
        rc = native_snapshot_name(c, epoch, &name);
        if (!rc) rc = native_path_issue(c, nr, step->args, name);
    }
    /* Even an error can follow creation/truncation: never replay or fall back. */
    kfree(name);
    step->result = rc;
    ++native_fast_hits;
    return true;
}

/* Fixed stat output and a bounded, checked pathname snapshot. Never read past
 * the first NUL: a valid string may end exactly at a capability/page boundary. */
static bool native_stat_fast(struct context *c, struct cv_step *step, u64 kinds)
{
    struct cn_context *n = c->native_context;
    struct cv_thread *t = c->main_thread;
    u64 nr = step->args[6], epoch = READ_ONCE(t->frame[82]);
    bool at = nr == CAPSTONE_SYS_newfstatat;
    struct cn_capref output;
    struct ns_copy_result copied;
    struct stat wire;
    struct file *file = NULL;
    char *name = NULL;
    long rc;
    BUILD_BUG_ON(sizeof(wire) != 128);
    if (nr != CAPSTONE_SYS_fstat && !at) return false;
    if (!nf_regs_plain(t, kinds, at ? ((1ULL << 11) | (1ULL << 12)) : (1ULL << 11))) return false;
    if (at && (step->args[3] & ~(u64)(AT_SYMLINK_NOFOLLOW | AT_NO_AUTOMOUNT))) return false;
    if (!at) {
        if (step->args[0] > INT_MAX || !dom_fd_allowed(c, step->args[0])) return false;
        file = fget(step->args[0]);
        if (!file) return false;
        if (file->private_data == c || !file->f_op) { fput(file); return false; }
    }
    rc = ns_session_admit(c->native_session, epoch);
    if (rc) { rc = -ESTALE; goto done; }
    output = (struct cn_capref){t->id, epoch, at ? 12 : 11, 0};
    if (at) {
        rc = native_snapshot_name(c, epoch, &name);
        if (rc) goto done;
    }
    if (native_authorize_reuse(n, &output, 0, sizeof(wire), true)) { rc = -EFAULT; goto done; }
    rc = native_stat_issue(c, nr, step->args, file, name, &wire);
    if (!rc) {
        copied = native_copy(c, n, &n->owner, &output, 0, &wire, sizeof(wire), NS_TO_DOMAIN);
        if (copied.cause || copied.bytes_done != sizeof(wire)) rc = -EFAULT;
    }
done:
    if (file) fput(file);
    kfree(name);
    step->result = rc;
    ++native_fast_hits;
    return true;
}

/* A fixed-size lock request needs no generic request snapshot or bounce allocation.
 * Shape/policy rejection precedes epoch admission; after admission no fallback. */
static bool native_lock_fast(struct context *c, struct cv_step *step, u64 kinds)
{
    struct cn_context *n = c->native_context;
    struct cv_thread *t = c->main_thread;
    struct native_wire_flock wire;
    struct flock lock = {0};
    struct cn_capref ref;
    struct ns_copy_result copied;
    struct file *file;
    u64 cmd = step->args[1], epoch = READ_ONCE(t->frame[82]);
    long rc;
    if (step->args[6] != CAPSTONE_NR_FCNTL_LOCK ||
        (cmd != F_GETLK && cmd != F_SETLK && cmd != F_SETLKW) ||
        !nf_regs_plain(t, kinds, 1ULL << 12) || step->args[0] > INT_MAX ||
        !dom_fd_allowed(c, step->args[0])) return false;
    file = fget(step->args[0]);
    if (!file) return false;
    if (file->private_data == c || !file->f_op) { fput(file); return false; }
    rc = ns_session_admit(c->native_session, epoch);
    if (rc) { rc = -ESTALE; goto done; }
    ref = (struct cn_capref){t->id, epoch, 12, 0};
    if (native_authorize_reuse(n, &ref, 0, sizeof(wire), false) ||
        (cmd == F_GETLK && native_authorize_reuse(n, &ref, 0, sizeof(wire), true))) {
        rc = -EFAULT; goto done;
    }
    copied = native_copy(c, n, &n->owner, &ref, 0, &wire, sizeof(wire), NS_FROM_DOMAIN);
    if (copied.cause || copied.bytes_done != sizeof(wire)) { rc = -EFAULT; goto done; }
    lock.l_type = wire.type; lock.l_whence = wire.whence;
    lock.l_start = wire.start; lock.l_len = wire.length; lock.l_pid = wire.pid;
    mutex_unlock(&vm_lock);
    rc = capstone_native_fcntl(step->args[0], file, cmd, 0, &lock, DOM_FDS);
    mutex_lock(&vm_lock);
    if (rc <= -ERESTARTSYS && rc >= -ERESTART_RESTARTBLOCK) rc = -EINTR;
    if (!rc && cmd == F_GETLK) {
        memset(&wire, 0, sizeof(wire));
        wire.type = lock.l_type; wire.whence = lock.l_whence;
        wire.start = lock.l_start; wire.length = lock.l_len; wire.pid = lock.l_pid;
        copied = native_copy(c, n, &n->owner, &ref, 0, &wire, sizeof(wire), NS_TO_DOMAIN);
        if (copied.cause || copied.bytes_done != sizeof(wire)) rc = -EFAULT;
    }
done:
    fput(file);
    step->result = rc;
    ++native_fast_hits;
    return true;
}
/* Returns true when the call was completed (step->result set). */
static bool native_fast_try(struct context *c, struct cv_step *step)
{
    struct cn_context *n = c->native_context;
    struct cv_thread *t = c->main_thread;
    u64 nr = step->args[6], kinds, epoch;
    struct cn_capref ref;
    struct cu_result r;
    struct file *file;
    size_t length;
    long rc, value;
    bool write;

    if (!native_fast || !n || native_owner(n, &n->owner) ||
        current->seccomp.mode != SECCOMP_MODE_DISABLED) return false;
    kinds = READ_ONCE(t->frame[83]);
    if (native_close_fast(c, step, kinds)) return true;
    if (native_fd_fast(c, step, kinds)) return true;
    if (native_seek_fast(c, step, kinds)) return true;
    if (native_path_fast(c, step, kinds)) return true;
    if (native_stat_fast(c, step, kinds)) return true;
    if (native_lock_fast(c, step, kinds)) return true;
    if (native_task_value(nr, &value)) {
        if (!nf_regs_plain(t, kinds, 0)) return false;
        if (ns_session_admit(c->native_session, READ_ONCE(t->frame[82]))) value = -ESTALE;
        step->result = value;
        ++native_fast_hits;
        return true;
    }
    switch (nr) {
    case CAPSTONE_SYS_read: case CAPSTONE_SYS_pread64: write = false; break;
    case CAPSTONE_SYS_write: case CAPSTONE_SYS_pwrite64: write = true; break;
    default: return false;
    }
    length = step->args[2];
    if (!length || length > NATIVE_FAST_MAX || step->args[0] > INT_MAX ||
        !nf_regs_plain(t, kinds, 1ULL << 11) || !dom_fd_allowed(c, step->args[0]) ||
        ((nr == CAPSTONE_SYS_pread64 || nr == CAPSTONE_SYS_pwrite64) && (s64)step->args[3] < 0))
        return false;
    file = fget(step->args[0]);
    if (!file) return false;
    if (file->private_data == c || !file->f_op ||
        ((nr == CAPSTONE_SYS_pread64 || nr == CAPSTONE_SYS_pwrite64) &&
         !(file->f_mode & (write ? FMODE_PWRITE : FMODE_PREAD)))) { fput(file); return false; }
    if (!n->bounce) {
        n->bounce = kmalloc(NATIVE_FAST_MAX, GFP_KERNEL);
        if (!n->bounce) { fput(file); return false; }
    }
    epoch = READ_ONCE(t->frame[82]);
    if (ns_session_admit(c->native_session, epoch)) { fput(file); step->result = -ESTALE; return true; }
    ref = (struct cn_capref){t->id, epoch, 11, 0};
    /* Preflight only: the copy below re-authenticates every chunk. */
    if (native_authorize_reuse(n, &ref, 0, length, !write)) { value = -EFAULT; goto out; }
    if (write) {
        r = cv_checked_copy_reuse(&n->copy, c, t, &ref, 0, n->bounce, length, CU_FROM_DOMAIN);
        if (r.cause || r.bytes_done != length) { value = -EFAULT; goto out; }
    }
    rc = native_rw(file, nr, step->args[3], n->bounce, length);
    if (rc <= -ERESTARTSYS && rc >= -ERESTART_RESTARTBLOCK) rc = -EINTR;
    value = rc;
    if (rc > 0 && (size_t)rc > length) value = -EPROTO;
    else if (rc > 0 && !write) {
        r = cv_checked_copy_reuse(&n->copy, c, t, &ref, 0, n->bounce, rc, CU_TO_DOMAIN);
        if (r.bytes_done > (size_t)rc || (!r.cause && r.bytes_done != (size_t)rc)) value = -EPROTO;
        else if (r.cause) value = r.bytes_done ? (long)r.bytes_done : -EFAULT;
    }
out:
    fput(file);
    step->result = value;
    ++native_fast_hits;
    return true;
}
/* The register-ABI service: a6 is the call, a0..a5 keep their tags. */
static long native_service_direct_impl(struct context *c, struct cv_step *step)
{
    struct ns_result result = {0};
    long rc;
    step->reply = 1;
    if (!checked_copy) return -EOPNOTSUPP;
    if (!c->native_session) {
        rc = native_session(c);
        if (rc) return rc;
    }
    if (step->args[6] == CAPSTONE_SYS_pipe2) return native_pipe_direct(c, step);
#ifdef CV_TASK_BACKEND
    if (step->args[6] == CAPSTONE_NR_SPAWN || step->args[6] == CAPSTONE_SYS_wait4) {
        struct cv_thread *t = c->main_thread;
        struct cn_capref ref = {t->id, READ_ONCE(t->frame[82]), 10, 0}, status = ref, usage = ref;
        u64 tags = READ_ONCE(t->frame[83]);
        rc = ns_session_admit(c->native_session, ref.context_epoch);
        if (!rc && step->args[6] == CAPSTONE_NR_SPAWN) {
            if (!nf_regs_plain(t, tags, 1ULL << 10)) rc = -EINVAL;
            else rc = native_spawn_call(c, &ref, 0, step->args[1]);
        } else if (!rc) {
            u64 expected = tags & ((1ULL << 11) | (1ULL << 13));
            status.frame_slot = 11; usage.frame_slot = 13;
            if (!nf_regs_plain(t, tags, expected) ||
                (!(expected & (1ULL << 11)) && step->args[1]) ||
                (!(expected & (1ULL << 13)) && step->args[3])) rc = -EINVAL;
            else rc = native_wait_call(c, step->args[0], step->args[2],
                expected & (1ULL << 11) ? &status : NULL, 0,
                expected & (1ULL << 13) ? &usage : NULL, 0);
        }
        step->result = rc;
        return 0;
    }
#endif
    if (!native_supported(step->args[6])) { step->result = native_refused(step->args[6]); return 0; }
    if (native_fast_try(c, step)) return 0;
    rc = ns_dispatch(c->native_session, READ_ONCE(c->main_thread->frame[82]), &result);
    step->result = rc ? rc : result.value;
    return 0;
}
static long native_service_direct(struct context *c, struct cv_step *step)
{
    long rc = native_service_direct_impl(c, step);
    native_route_record("register", step->args[6], step->args[2], rc ? rc : step->result);
    return rc;
}
static int native_exchange_reply(struct context *c, const struct cn_capref *meta,
                                 const struct capstone_delegate_entry *e, long value)
{
    native_route_record("exchange", e->nr, e->args[2], value);
    return native_meta_reply(c, meta, value);
}
/* The exchange-buffer service of the SDK's delegate calls. */
static long native_service(struct context *c, struct cv_step *step)
{
    struct ns_result result = {0};
    struct capstone_delegate_entry e;
    struct ns_request request = {0};
    struct cn_capref meta = {c->main_thread->id, READ_ONCE(c->main_thread->frame[82]), 10, 0};
    struct cn_capref exchange = meta;
    struct cu_result copied;
    size_t lengths[6];
    unsigned i;
    long rc;
    step->reply = 1;
    if (!checked_copy) return -EOPNOTSUPP;
    rc = native_session(c);
    if (rc) return rc;
    exchange.frame_slot = 11;
    copied = cv_checked_copy(c, c->main_thread, &meta, 0, &e, sizeof(e), CU_FROM_DOMAIN);
    if (copied.cause) return -EFAULT;
    rc = capstone_delegate_validate(&e, NULL, c->dom_exchange_bytes, lengths);
    if (rc) return native_exchange_reply(c, &meta, &e, -rc);
    if (e.nr == CAPSTONE_SYS_exit_group) return (e.args[0] & 255) + 1;
    if (e.nr == DOM_HELLO || e.nr == CAPSTONE_NR_SIGPOLL)
        return native_exchange_reply(c, &meta, &e, 0);
    if (e.nr == CAPSTONE_SYS_pipe2) return native_pipe_exchange(c, &meta, &exchange, &e);
#ifdef CV_TASK_BACKEND
    if (e.nr == CAPSTONE_NR_SPAWN || e.nr == CAPSTONE_SYS_wait4) {
        long reply;
        rc = ns_session_admit(c->native_session, meta.context_epoch);
        if (!rc && e.nr == CAPSTONE_NR_SPAWN)
            rc = native_spawn_call(c, &exchange, e.args[0], e.args[1]);
        else if (!rc)
            rc = native_wait_call(c, e.args[0], e.args[2],
                e.args[1] ? &exchange : NULL, e.args[1],
                e.args[3] ? &exchange : NULL, e.args[3]);
        reply = native_exchange_reply(c, &meta, &e, rc);
        if (reply && e.nr == CAPSTONE_NR_SPAWN && rc > 0) native_spawn_cancel(c, rc);
        return reply;
    }
#endif
    if (dom_vector(e.nr)) {
        rc = native_gcc_vector(c, &e, &exchange);
        return native_exchange_reply(c, &meta, &e, rc);
    }
    if (!native_supported(e.nr)) return native_exchange_reply(c, &meta, &e, native_refused(e.nr));
    request.nr = e.nr;
    for (i = 0; i < 6; ++i) {
        request.scalar[i] = e.args[i];
        request.refs[i] = exchange;
        if (e.flags & (1ULL << i)) {
            request.tagged |= 1U << i;
            request.offsets[i] = e.args[i];
        } else if (!e.args[i]) request.nulls |= 1U << i;
    }
    rc = ns_dispatch_request(c->native_session, meta.context_epoch, &request, &result);
    return native_exchange_reply(c, &meta, &e, rc ? rc : result.value);
}

/* Built-in kernel helper; its typed definition is required at link time. */
static long native_protect(struct context *c, struct cv_step *s)
{
    unsigned long base = s->args[0], bytes = s->args[1], prot = s->args[2];
    struct arena *a = arena_at(c, base);
    if (!bytes || ((base | bytes) & ~PAGE_MASK) ||
        (prot & ~(PROT_READ | PROT_WRITE | PROT_EXEC)) ||
        !a || !a->dom_dynamic || a->dom_metadata || a->copy_holds || a->copy_retired ||
        !dom_within(base - a->address, bytes, a->bytes)) return -EINVAL;
    /* Permission changes preserve the grant, object identity and mapping.
     * Linux performs aligned subrange splitting and checks VM_MAY* plus LSM.
     * No mapping is removed or replaced, and vm_lock excludes grant retirement. */
    s->result = capstone_native_mprotect(base, bytes, prot);
    return 0;
}
