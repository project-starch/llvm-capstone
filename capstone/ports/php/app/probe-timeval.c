/* Does this libc's select() reject a timeval whose tv_usec is out of range, or sleep on it?
 *
 * WHY THIS PROBE EXISTS. CRASH-092's trigger must run to its end for its use-after-free to be
 * reached: the dead object is read when the script's resources are destroyed at shutdown
 * (_zval_ptr_dtor, ASan's own site). Before that, PHP 5.0.0 checks the HTTP socket's liveness
 * (main/streams/xp_socket.c, PHP_STREAM_OPTION_CHECK_LIVENESS) with a `struct timeval tv` of
 * which it sets ONLY tv_sec; tv_usec is whatever the stack held. Linux rejects such a timeval:
 * glibc hands it to the kernel, which answers EINVAL, select returns -1, PHP reads "alive" and
 * goes on, and the script ends -- which is what the corpus's native oracle saw. musl 1.2.5's
 * select (src/select/select.c) folds an out-of-range tv_usec into the seconds instead:
 *
 *     if (us/1000000 > max_time - s) { s = max_time; ... } else { s += us/1000000; ... }
 *
 * so a positive garbage tv_usec becomes a timeout of hours or days, and the script hangs before
 * its defect. Whether it hangs depends on what the stack held, which is why the case hung on
 * some arms and images and completed on others (plan doc, 2026-10-28).
 *
 * The declaration this probe checks: "select() rejects a timeval whose tv_usec is out of range
 * (EINVAL) rather than sleeping on it". Control: a timeval of {0, 0} must return 0 at once, or the
 * probe cannot read timeouts at all. Test: {0, 5000000000} -- five thousand seconds if folded --
 * under a two-second alarm. EINVAL at once means the behaviour is PRESENT (the defect is
 * reachable); a return of 0 after the full wait, or EINTR from the alarm, means it is ABSENT.
 *
 *   TIMEVAL_PROBE control rc=0 ...                  and  out-of-range-usec rejected=1  -> present
 *   TIMEVAL_PROBE out-of-range-usec rejected=0 ...                                    -> absent
 *   anything else (control failed, no DONE line)                                       -> no result
 *
 * No capability builtin, no allocator of PHP's: the question is the libc's alone. */
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/select.h>
#include <sys/time.h>
#include <unistd.h>

static void on_alarm(int sig) { (void)sig; }

static long elapsed_ms(const struct timeval *a, const struct timeval *b)
{
    return (b->tv_sec - a->tv_sec) * 1000L + (b->tv_usec - a->tv_usec) / 1000L;
}

int main(void)
{
    int fds[2], failures = 0, rc, err;
    fd_set rfds;
    struct timeval tv, t0, t1;
    struct sigaction sa;

    if (pipe(fds)) { printf("TIMEVAL_PROBE FAIL pipe: %s\n", strerror(errno)); return 1; }
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_alarm;               /* no SA_RESTART: the alarm must interrupt select */
    sigaction(SIGALRM, &sa, NULL);

    /* control: a zero timeout returns 0 at once */
    FD_ZERO(&rfds); FD_SET(fds[0], &rfds);
    tv.tv_sec = 0; tv.tv_usec = 0;
    gettimeofday(&t0, NULL);
    errno = 0; rc = select(fds[0] + 1, &rfds, NULL, NULL, &tv); err = errno;
    gettimeofday(&t1, NULL);
    printf("TIMEVAL_PROBE control rc=%d errno=%d ms=%ld\n", rc, err, elapsed_ms(&t0, &t1));
    if (rc != 0 || elapsed_ms(&t0, &t1) > 500) {
        printf("TIMEVAL_PROBE FAIL the zero timeout did not return 0 at once; the probe cannot "
               "read timeouts, so its reading of the out-of-range one means nothing\n");
        failures++;
    }

    /* the question: tv_usec far out of range, under a two-second alarm */
    FD_ZERO(&rfds); FD_SET(fds[0], &rfds);
    tv.tv_sec = 0; tv.tv_usec = 5000000000L;
    alarm(2);
    gettimeofday(&t0, NULL);
    errno = 0; rc = select(fds[0] + 1, &rfds, NULL, NULL, &tv); err = errno;
    gettimeofday(&t1, NULL);
    alarm(0);
    printf("TIMEVAL_PROBE out-of-range-usec rejected=%d rc=%d errno=%d (%s) ms=%ld\n",
           (rc == -1 && err == EINVAL) ? 1 : 0, rc, err, err ? strerror(err) : "ok",
           elapsed_ms(&t0, &t1));
    printf("TIMEVAL_PROBE DONE failures=%d\n", failures);
    fflush(stdout);
    return failures ? 1 : 0;
}
