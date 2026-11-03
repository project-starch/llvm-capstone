/* SPDX-License-Identifier: GPL-2.0-only */
/* The native trampoline never reads transport. It receives only a syscall
 * prepared here, over private untagged bounce memory, then returns a result.
 * The fixed transport grants cannot retire, and only dom_pid can step this
 * context; that is the lifetime hold across a sleeping syscall. */
#include <asm/ioctls.h>
#include "delegate-wire.c"
#include "delegate-policy.h"

struct dom_delegate {
    struct capstone_delegate_entry request;
    const struct capstone_delegate_shape *shape;
    size_t lengths[CAPSTONE_DELEGATE_ARGS];
    unsigned long bounce, vectors;
    struct iovec *wire;
    char *scratch;
    bool pending;
};

/* Closed day-one set: all have existing delegated shapes and belong to the
 * launcher's seccomp allowlist. Pointer-retaining, signal-handover, nested
 * message and process/job-control operations are deliberately absent. */
static bool dom_forwarded(u64 nr)
{
    switch (nr) {
#define CALL(name) case CAPSTONE_SYS_##name:
    CALL(getcwd) CALL(dup) CALL(dup3) CALL(fcntl) CALL(ioctl)
    CALL(mkdirat) CALL(unlinkat) CALL(ftruncate) CALL(faccessat)
    CALL(symlinkat) CALL(sync_file_range) CALL(flock) CALL(fchmodat)
    CALL(chdir) CALL(openat) CALL(close) CALL(pipe2) CALL(getdents64)
    CALL(lseek) CALL(pread64) CALL(pwrite64)
    CALL(readv) CALL(writev) CALL(preadv) CALL(pwritev)
    CALL(readlinkat) CALL(newfstatat) CALL(fstat) CALL(fsync) CALL(fdatasync)
    CALL(utimensat) CALL(renameat2) CALL(clock_gettime) CALL(gettimeofday)
    CALL(times) CALL(getpid) CALL(getppid) CALL(getuid) CALL(geteuid)
    CALL(getgid) CALL(getegid) CALL(gettid) CALL(umask) CALL(uname)
    CALL(sysinfo) CALL(prlimit64) CALL(getrandom) CALL(sched_yield)
    CALL(getpgid) CALL(getsid) CALL(linkat) CALL(mknodat) CALL(statfs)
    CALL(fstatfs) CALL(statx) CALL(truncate) CALL(fallocate) CALL(fchdir)
    CALL(fchmod) CALL(fchown) CALL(fchownat) CALL(faccessat2)
    CALL(sendfile) CALL(copy_file_range) CALL(readahead) CALL(fadvise64)
    CALL(sync) CALL(syncfs) CALL(memfd_create) CALL(clock_getres)
    CALL(getgroups) CALL(getrusage) CALL(getpriority) CALL(setpriority)
    CALL(sched_getaffinity) CALL(sched_setaffinity)
    CALL(sched_get_priority_max) CALL(sched_get_priority_min)
    CALL(sched_rr_get_interval) CALL(getresuid) CALL(getresgid)
    CALL(rt_sigprocmask)
    CALL(socket) CALL(socketpair) CALL(bind) CALL(listen) CALL(connect)
    CALL(sendto) CALL(setsockopt) CALL(shutdown) CALL(epoll_create1)
    CALL(epoll_ctl) CALL(eventfd2) CALL(timerfd_create)
    CALL(timerfd_settime) CALL(timerfd_gettime)
#undef CALL
    case CAPSTONE_NR_FCNTL_LOCK: case CAPSTONE_NR_IOCTL_BUF:
        return true;
    default: return false;
    }
}
static bool dom_writes(unsigned kind)
{
    return kind == CAPSTONE_ARG_OUT || kind == CAPSTONE_ARG_OPT_OUT ||
           kind == CAPSTONE_ARG_INOUT || kind == CAPSTONE_ARG_OPT_INOUT;
}
static bool dom_reads(unsigned kind)
{
    return kind == CAPSTONE_ARG_IN || kind == CAPSTONE_ARG_OPT_IN ||
           kind == CAPSTONE_ARG_INOUT || kind == CAPSTONE_ARG_OPT_INOUT;
}
static int dom_touch(struct context *c, unsigned long at, size_t bytes)
{
    struct cv_touch r = {.address = at, .bytes = bytes};
    return bytes ? touch(c, &r) : 0;
}
static int dom_store(struct context *c, unsigned long at, const void *src, size_t bytes)
{
    int rc = dom_touch(c, at, bytes);
    return rc ? rc : copy_to_user((void __user *)at, src, bytes) ? -EFAULT : 0;
}
static int dom_copy(struct dom_delegate *d, unsigned long dst, unsigned long src, size_t bytes)
{
    while (bytes) {
        size_t n = min_t(size_t, bytes, PAGE_SIZE);
        if (copy_from_user(d->scratch, (void __user *)src, n) ||
            copy_to_user((void __user *)dst, d->scratch, n)) return -EFAULT;
        src += n; dst += n; bytes -= n;
    }
    return 0;
}
static long dom_issue(struct context *c, unsigned long nr, const unsigned long a[6])
{
    struct pt_regs *r = current_pt_regs();
    r->a1 = a[0]; r->a2 = a[1]; r->a3 = a[2];
    r->a4 = a[3]; r->a5 = a[4]; r->a6 = a[5]; r->a7 = nr;
    c->dom_delegate->pending = true;
    return DOM_DO_SYSCALL;
}
static bool dom_vector(u64 nr)
{
    return nr == CAPSTONE_SYS_readv || nr == CAPSTONE_SYS_writev ||
           nr == CAPSTONE_SYS_preadv || nr == CAPSTONE_SYS_pwritev;
}
static long dom_prepare_vector(struct context *c, unsigned long a[6])
{
    struct dom_delegate *d = c->dom_delegate;
    const struct capstone_delegate_entry *e = &d->request;
    size_t count = e->args[2], total = 0, i;
    if (count > 1024) return -EINVAL;
    kfree(d->wire); d->wire = NULL;
    if (!count) { a[1] = 0; return dom_issue(c, e->nr, a); }
    d->wire = kmalloc_array(count, sizeof(*d->wire), GFP_KERNEL);
    if (!d->wire) return -ENOMEM;
    if (copy_from_user(d->wire, (void __user *)(c->dom_exchange + e->args[1]),
                       count * sizeof(*d->wire))) return -EFAULT;
    for (i = 0; i < count; ++i) {
        unsigned long offset = (unsigned long)d->wire[i].iov_base;
        size_t bytes = d->wire[i].iov_len;
        if (!dom_within(offset, bytes, c->dom_exchange_bytes)) return -EFAULT;
        if (bytes > SSIZE_MAX - total) return -EINVAL;
        total += bytes;
    }
    if (!d->vectors) {
        unsigned long p = dom_reserve(16384, PAGE_SIZE, PROT_READ | PROT_WRITE);
        if (IS_ERR_VALUE(p)) return p;
        d->vectors = p;
    }
    for (i = 0; i < count; ++i) {
        unsigned long offset = (unsigned long)d->wire[i].iov_base;
        struct iovec v = {.iov_base = (void __user *)(d->bounce + offset),
                          .iov_len = d->wire[i].iov_len};
        /* Match vector_call: initialize each span even on reads, preserving
         * overlap and bytes beyond a short read. The wire array is private. */
        if (dom_copy(d, d->bounce + offset, c->dom_exchange + offset, v.iov_len) ||
            copy_to_user((void __user *)(d->vectors + i * sizeof(v)), &v, sizeof(v)))
            return -EFAULT;
    }
    a[1] = d->vectors;
    return dom_issue(c, e->nr, a);
}
static long dom_prepare(struct context *c, const struct capstone_delegate_entry *e)
{
    struct dom_delegate *d = c->dom_delegate;
    const struct capstone_delegate_shape *s;
    size_t lengths[CAPSTONE_DELEGATE_ARGS];
    unsigned long a[CAPSTONE_DELEGATE_ARGS] = {0};
    unsigned mask, i;
    long rc;
    if (!dom_forwarded(e->nr) && e->nr != CAPSTONE_NR_SIGACTION) return -ENOSYS;
    rc = capstone_delegate_validate(e, NULL, c->dom_exchange_bytes, lengths);
    if (rc) return -rc;
    s = capstone_delegate_shape(e->nr);
    mask = fd_arguments(e->nr);
    for (i = 0; i < CAPSTONE_DELEGATE_ARGS; ++i)
        if ((mask & (1U << i)) && (int)e->args[i] == c->dom_fd) return -EBADF;
    if (!command_ok(e->nr, e->args[1])) return -ENOSYS;
    if (e->nr == CAPSTONE_SYS_prlimit64 && (pid_t)e->args[0] != 0 &&
        (pid_t)e->args[0] != task_tgid_vnr(current)) return -EPERM;
    if ((e->nr == CAPSTONE_SYS_sched_getaffinity || e->nr == CAPSTONE_SYS_sched_setaffinity) &&
        e->args[0] != 0) return -EPERM;
    if (e->nr == CAPSTONE_SYS_sched_rr_get_interval && (pid_t)e->args[0] != 0 &&
        (pid_t)e->args[0] != task_tgid_vnr(current)) return -EPERM;
    if ((e->nr == CAPSTONE_SYS_getpriority || e->nr == CAPSTONE_SYS_setpriority) &&
        (e->args[0] != 0 /* PRIO_PROCESS */ ||
         ((pid_t)e->args[1] != 0 && (pid_t)e->args[1] != task_tgid_vnr(current)))) return -EPERM;
    if (e->nr == CAPSTONE_SYS_rt_sigprocmask && e->args[3] != 8) return -EINVAL;
    if (e->nr == CAPSTONE_NR_SIGACTION) {
        if (e->args[0] < 1 || e->args[0] > 64 || e->args[0] == SIGKILL ||
            e->args[0] == SIGSTOP || e->args[1] > CAPSTONE_SIGNAL_CAUGHT) return -EINVAL;
        if (e->args[1] == CAPSTONE_SIGNAL_CAUGHT) return -ENOSYS;
    }
    if (!d) {
        d = kzalloc(sizeof(*d), GFP_KERNEL);
        if (!d) return -ENOMEM;
        c->dom_delegate = d;
    }
    d->request = *e; d->shape = s;
    memcpy(d->lengths, lengths, sizeof(lengths));
    if ((e->flags || e->nr == CAPSTONE_NR_SIGACTION) && !d->bounce) {
        unsigned long p;
        if (!d->scratch) d->scratch = (char *)__get_free_page(GFP_KERNEL);
        if (!d->scratch) return -ENOMEM;
        /* Strings must not alias output buffers: the launcher reads strings
         * from transport while it writes buffers in bounce memory. Reserve
         * a private slice per argument to preserve that rule, even for an
         * adversarial wire request with overlapping offsets. */
        p = dom_reserve(c->dom_exchange_bytes * (CAPSTONE_DELEGATE_ARGS + 1),
                        PAGE_SIZE, PROT_READ | PROT_WRITE);
        if (IS_ERR_VALUE(p)) return p;
        d->bounce = p;
    }
    if (e->nr == CAPSTONE_NR_SIGACTION) {
        /* RISC-V's kernel sigaction has no restorer. No native handler is
         * installed until caught-signal recording/restart is implemented. */
        unsigned long action[3] = {e->args[1] == CAPSTONE_SIGNAL_IGNORE ? 1 : 0,
                                  e->args[2] & (SA_NOCLDSTOP | SA_NOCLDWAIT), 0};
        if (copy_to_user((void __user *)d->bounce, action, sizeof(action))) return -EFAULT;
        a[0] = e->args[0]; a[1] = d->bounce; a[3] = 8;
        return dom_issue(c, CAPSTONE_SYS_rt_sigaction, a);
    }
    for (i = 0; i < s->argc; ++i) a[i] = e->args[i];
    if (dom_vector(e->nr)) return dom_prepare_vector(c, a);
    for (i = 0; i < s->argc; ++i) {
        unsigned kind = s->args[i].kind;
        size_t bytes = lengths[i];
        if (!((e->flags >> i) & 1)) continue;
        /* No supported row needs a length fetched through another pointer.
         * Adding such a row requires the validator's second, snapshot pass. */
        if (s->args[i].length == CAPSTONE_LEN_WORD) return -ENOSYS;
        a[i] = d->bounce + e->args[i];
        if (kind == CAPSTONE_ARG_STR || kind == CAPSTONE_ARG_OPT_STR) {
            long n = strnlen_user((char __user *)(c->dom_exchange + e->args[i]),
                                  c->dom_exchange_bytes - e->args[i]);
            if (n <= 0 || n > c->dom_exchange_bytes - e->args[i]) return -EFAULT;
            bytes = n;
            a[i] = d->bounce + c->dom_exchange_bytes * (i + 1);
        }
        if (dom_reads(kind) || kind == CAPSTONE_ARG_STR || kind == CAPSTONE_ARG_OPT_STR) {
            rc = dom_copy(d, a[i], c->dom_exchange + e->args[i], bytes);
            if (rc) return rc;
        } else if (clear_user((void __user *)a[i], bytes)) return -EFAULT;
    }
    return dom_issue(c, e->nr == CAPSTONE_NR_FCNTL_LOCK ? CAPSTONE_SYS_fcntl :
                         e->nr == CAPSTONE_NR_IOCTL_BUF ? CAPSTONE_SYS_ioctl : e->nr, a);
}
static long dom_complete(struct context *c, long result)
{
    struct dom_delegate *d = c->dom_delegate;
    const struct capstone_delegate_entry *e = &d->request;
    unsigned i;
    int rc;
    if (dom_vector(e->nr)) {
        bool writing = e->nr == CAPSTONE_SYS_writev || e->nr == CAPSTONE_SYS_pwritev;
        size_t left = result < 0 ? 0 : result;
        for (i = 0; !writing && i < e->args[2] && left; ++i) {
            unsigned long offset = (unsigned long)d->wire[i].iov_base;
            size_t n = min_t(size_t, d->wire[i].iov_len, left);
            rc = dom_touch(c, c->dom_exchange + offset, n);
            if (rc) return rc;
            rc = dom_copy(d, c->dom_exchange + offset, d->bounce + offset, n);
            if (rc) return rc;
            left -= n;
        }
    } else {
        for (i = 0; i < d->shape->argc; ++i) {
            size_t bytes;
            if (!((e->flags >> i) & 1) || !dom_writes(d->shape->args[i].kind)) continue;
            bytes = capstone_delegate_result_bytes(e->nr, i, d->lengths[i], result);
            rc = dom_touch(c, c->dom_exchange + e->args[i], bytes);
            if (rc) return rc;
            rc = dom_copy(d, c->dom_exchange + e->args[i], d->bounce + e->args[i], bytes);
            if (rc) return rc;
        }
    }
    return 0;
}
static void dom_delegate_free(struct context *c)
{
    struct dom_delegate *d = c->dom_delegate;
    if (!d) return;
    kfree(d->wire);
    if (d->scratch) free_page((unsigned long)d->scratch);
    kfree(d); c->dom_delegate = NULL;
}
