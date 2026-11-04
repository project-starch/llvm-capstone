/* SPDX-License-Identifier: GPL-2.0-only */
/* Included after the resource broker. No domain address reaches Linux APIs. */
#include <linux/seccomp.h>
#include <linux/timekeeping.h>
#include <capstone/native-syscall.h>
#include "native-syscall-table.c"
#include "native-syscall-core.c"

struct cn_context { struct context *c; struct cv_thread *t; };
static bool native_dispatch;
module_param(native_dispatch, bool, 0444);
MODULE_PARM_DESC(native_dispatch, "checked in-kernel native syscall loop (default off)");

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
int cn_authorize(struct cn_context *n, const struct cn_owner *owner,
                  const struct cn_capref *ref, cn_u64 offset, cn_u64 bytes, int write)
{
    struct cv_copy_state *state;
    cn_u64 address;
    cn_u32 cause;
    int rc = native_owner(n, owner);
    if (rc) return rc;
    state = kzalloc(sizeof(*state), GFP_KERNEL);
    if (!state) return -ENOMEM;
    state->c = n->c; state->t = n->t;
    cv_copy_enter(state);
    cause = cv_copy_check(state, ref, offset, bytes,
                          write ? CU_TO_DOMAIN : CU_FROM_DOMAIN, &address);
    cv_copy_drain(state);
    cv_copy_leave(state);
    kfree(state);
    return cause ? -EFAULT : 0;
}
static struct ns_copy_result native_copy(void *opaque, struct cn_context *n,
        const struct cn_owner *owner, const struct cn_capref *ref, cn_u64 offset,
        void *buffer, size_t bytes, enum ns_direction direction)
{
    struct cu_result r = {0, CU_REFUSED};
    (void)opaque;
    if (!native_owner(n, owner))
        r = cv_checked_copy(n->c, n->t, ref, offset, buffer, bytes,
                            direction == NS_TO_DOMAIN ? CU_TO_DOMAIN : CU_FROM_DOMAIN);
    return (struct ns_copy_result){r.bytes_done, r.cause};
}
static bool native_supported(u64 nr)
{
    switch (nr) {
    case CAPSTONE_SYS_read: case CAPSTONE_SYS_write:
    case CAPSTONE_SYS_pread64: case CAPSTONE_SYS_pwrite64:
    case CAPSTONE_SYS_close: case CAPSTONE_SYS_lseek:
    case CAPSTONE_SYS_getpid: case CAPSTONE_SYS_getppid: case CAPSTONE_SYS_gettid:
    case CAPSTONE_SYS_getuid: case CAPSTONE_SYS_geteuid:
    case CAPSTONE_SYS_getgid: case CAPSTONE_SYS_getegid:
    case CAPSTONE_SYS_clock_gettime:
    case CAPSTONE_SYS_sched_yield:
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
    if (!call->descriptor->fd_mask) return 0;
    if (call->scalar[0] > INT_MAX || !dom_fd_allowed(c, call->scalar[0])) return -EBADF;
    file = fget(call->scalar[0]);
    if (!file) return -EBADF;
    if (file->private_data == c || !file->f_op) { fput(file); return -EBADF; }
    *held = file;
    return 0;
}
static struct ns_completion native_issue(void *opaque, const struct ns_invocation *call,
                                          void *held)
{
    struct context *c = opaque;
    struct file *file = held;
    const u64 *a = call->scalar;
    u64 nr = call->descriptor->nr;
    long rc = -ENOSYS;
    bool positional, locked = false;
    loff_t pos, *ppos;
    struct timespec64 ts;
    switch (nr) {
    case CAPSTONE_SYS_getpid: rc = task_tgid_vnr(current); break;
    case CAPSTONE_SYS_getppid: rc = task_ppid_nr_ns(current, task_active_pid_ns(current)); break;
    case CAPSTONE_SYS_gettid: rc = task_pid_vnr(current); break;
    case CAPSTONE_SYS_getuid: rc = from_kuid_munged(current_user_ns(), current_uid()); break;
    case CAPSTONE_SYS_geteuid: rc = from_kuid_munged(current_user_ns(), current_euid()); break;
    case CAPSTONE_SYS_getgid: rc = from_kgid_munged(current_user_ns(), current_gid()); break;
    case CAPSTONE_SYS_getegid: rc = from_kgid_munged(current_user_ns(), current_egid()); break;
    case CAPSTONE_SYS_sched_yield:
        mutex_unlock(&vm_lock); yield(); mutex_lock(&vm_lock); rc = 0; break;
    case CAPSTONE_SYS_clock_gettime:
        if (a[0] != CLOCK_REALTIME && a[0] != CLOCK_MONOTONIC) { rc = -EINVAL; break; }
        if (a[0] == CLOCK_REALTIME) ktime_get_real_ts64(&ts);
        else ktime_get_ts64(&ts);
        if (call->buffer[1].data) memcpy(call->buffer[1].data, &ts, sizeof(ts));
        rc = 0; break;
    case CAPSTONE_SYS_close:
        clear_bit(a[0], c->dom_delegate->fds);
        mutex_unlock(&vm_lock); rc = close_fd(a[0]); mutex_lock(&vm_lock); break;
    case CAPSTONE_SYS_lseek:
        mutex_unlock(&vm_lock); rc = vfs_llseek(file, a[1], a[2]); mutex_lock(&vm_lock); break;
    case CAPSTONE_SYS_read: case CAPSTONE_SYS_write:
    case CAPSTONE_SYS_pread64: case CAPSTONE_SYS_pwrite64:
        positional = nr == CAPSTONE_SYS_pread64 || nr == CAPSTONE_SYS_pwrite64;
        if (positional && (s64)a[3] < 0) { rc = -EINVAL; break; }
        if (positional && !(file->f_mode & (nr == CAPSTONE_SYS_pread64 ? FMODE_PREAD : FMODE_PWRITE))) {
            rc = -ESPIPE; break;
        }
        mutex_unlock(&vm_lock);
        if (!positional && (file->f_mode & FMODE_ATOMIC_POS)) {
            mutex_lock(&file->f_pos_lock); locked = true;
        }
        pos = positional ? a[3] : file->f_pos;
        ppos = !positional && (file->f_mode & FMODE_STREAM) ? NULL : &pos;
        if (nr == CAPSTONE_SYS_read || nr == CAPSTONE_SYS_pread64)
            rc = kernel_read(file, call->buffer[1].data, call->buffer[1].length, ppos);
        else rc = kernel_write(file, call->buffer[1].data, call->buffer[1].length, ppos);
        if (!positional && ppos && rc >= 0) file->f_pos = pos;
        if (locked) mutex_unlock(&file->f_pos_lock);
        mutex_lock(&vm_lock);
        break;
    }
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
    *c->native_context = (struct cn_context){c, c->main_thread};
    rc = ns_session_create(c->native_context, &owner, &ops, c, &c->native_session);
    if (rc) { kfree(c->native_context); c->native_context = NULL; }
    return rc;
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
static long native_service(struct context *c, struct cv_step *step)
{
    struct ns_result result = {0};
    struct capstone_delegate_entry e;
    struct ns_request request = {0};
    struct cn_capref meta = {c->main_thread->id, c->main_thread->frame[82], 10, 0};
    struct cn_capref exchange = meta;
    struct cu_result copied;
    size_t lengths[6];
    unsigned i;
    long rc;
    bool direct = step->args[7] == NS_SERVICE_NATIVE;
    step->reply = 1;
    if (!checked_copy) return -EOPNOTSUPP;
    rc = native_session(c);
    if (rc) return rc;
    if (direct) {
        if (!native_supported(step->args[6])) { step->result = -ENOSYS; return 0; }
        rc = ns_dispatch(c->native_session, meta.context_epoch, &result);
        step->result = rc ? rc : result.value;
        return 0;
    }
    exchange.frame_slot = 11;
    copied = cv_checked_copy(c, c->main_thread, &meta, 0, &e, sizeof(e), CU_FROM_DOMAIN);
    if (copied.cause) return -EFAULT;
    rc = capstone_delegate_validate(&e, NULL, c->dom_exchange_bytes, lengths);
    if (rc) return native_meta_reply(c, &meta, -rc);
    if (e.nr == CAPSTONE_SYS_exit_group) return (e.args[0] & 255) + 1;
    if (e.nr == DOM_HELLO || e.nr == CAPSTONE_NR_SIGPOLL)
        return native_meta_reply(c, &meta, 0);
    if (!native_supported(e.nr)) return native_meta_reply(c, &meta, -ENOSYS);
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
    return native_meta_reply(c, &meta, rc ? rc : result.value);
}

/* Optional kernel extension; a stock kernel still loads the default-off module. */
extern int capstone_native_mprotect(unsigned long, size_t, unsigned long);
static long native_protect(struct context *c, struct cv_step *s)
{
    typeof(&capstone_native_mprotect) protect;
    unsigned long base = s->args[0], bytes = s->args[1], prot = s->args[2];
    struct arena *a = arena_at(c, base);
    if (!bytes || ((base | bytes) & ~PAGE_MASK) ||
        (prot & ~(PROT_READ | PROT_WRITE | PROT_EXEC)) ||
        !a || !a->dom_dynamic || a->copy_holds || a->copy_retired ||
        !dom_within(base - a->address, bytes, a->bytes)) return -EINVAL;
    /* Permission changes preserve the grant, object identity and mapping.
     * Linux performs aligned subrange splitting and checks VM_MAY* plus LSM.
     * No mapping is removed or replaced, and vm_lock excludes grant retirement. */
    protect = symbol_get(capstone_native_mprotect);
    if (!protect) return -EOPNOTSUPP;
    s->result = protect(base, bytes, prot);
    symbol_put(capstone_native_mprotect);
    return 0;
}
