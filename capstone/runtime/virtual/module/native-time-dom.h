/* SPDX-License-Identifier: GPL-2.0-only */
#ifdef CV_TASK_BACKEND
#include <linux/capstone_native_time.h>
#include <linux/hrtimer.h>
#include <linux/time_namespace.h>
#include <linux/restart_block.h>

static bool native_time_issue(struct context *c, const struct ns_invocation *call, long *result)
{
    const u64 *a = call->scalar;
    u64 nr = call->descriptor->nr;
    struct timespec64 request, remaining;
    long rc;
    unsigned request_arg, remaining_arg;
    clockid_t clock;
    ktime_t expires;
    enum hrtimer_mode mode;
    if (nr == CAPSTONE_SYS_pselect6) {
        if (a[0] > DOM_FDS) { *result = -EINVAL; return true; }
        for (unsigned arg = 1; arg <= 3; ++arg) {
            const unsigned long *bits = call->buffer[arg].data;
            if (!bits) continue;
            for (unsigned fd = 0; fd < a[0]; ++fd)
                if (test_bit(fd, bits) && !dom_fd_allowed(c, fd)) {
                    *result = -EBADF; return true;
                }
        }
        mutex_unlock(&vm_lock);
        *result = capstone_native_select(a[0], call->buffer[1].data,
            call->buffer[2].data, call->buffer[3].data,
            call->buffer[4].data, call->buffer[5].data);
        mutex_lock(&vm_lock);
        return true;
    }
    if (nr != CAPSTONE_SYS_nanosleep && nr != CAPSTONE_SYS_clock_nanosleep) return false;
    request_arg = nr == CAPSTONE_SYS_nanosleep ? 0 : 2;
    remaining_arg = request_arg + 1;
    clock = nr == CAPSTONE_SYS_nanosleep ? CLOCK_MONOTONIC : a[0];
    mode = nr == CAPSTONE_SYS_nanosleep || !a[1] ? HRTIMER_MODE_REL : HRTIMER_MODE_ABS;
    if (nr == CAPSTONE_SYS_clock_nanosleep &&
        ((a[1] & ~(u64)TIMER_ABSTIME) || (a[0] != CLOCK_REALTIME && a[0] != CLOCK_MONOTONIC))) {
        *result = -EINVAL; return true;
    }
    memcpy(&request, call->buffer[request_arg].data, sizeof(request));
    if (!timespec64_valid(&request)) { *result = -EINVAL; return true; }
    /* Linux's timer machinery, but never its user-pointer remaining-time
     * copy or scalar syscall restart. The dispatcher owns checked copy-out. */
    current->restart_block.nanosleep.type = TT_NONE;
    current->restart_block.nanosleep.rmtp = NULL;
    expires = timespec64_to_ktime(request);
    if (mode == HRTIMER_MODE_ABS && clock == CLOCK_MONOTONIC)
        expires = timens_ktime_to_host(clock, expires);
    mutex_unlock(&vm_lock);
    rc = hrtimer_nanosleep(expires, mode, clock);
    if (rc == -ERESTART_RESTARTBLOCK || rc == -ERESTARTNOHAND) {
        if (mode == HRTIMER_MODE_REL && call->buffer[remaining_arg].data) {
            s64 left = current->restart_block.nanosleep.expires - ktime_get_ns();
            remaining = ktime_to_timespec64(max_t(s64, left, 0));
            memcpy(call->buffer[remaining_arg].data, &remaining, sizeof(remaining));
        }
        rc = -EINTR;
    }
    current->restart_block.fn = do_no_restart_syscall;
    mutex_lock(&vm_lock);
    *result = rc;
    return true;
}
#else
static bool native_time_issue(struct context *c, const struct ns_invocation *call, long *result)
{ return false; }
#endif
