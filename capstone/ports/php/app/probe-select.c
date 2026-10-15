/* Does select() work on this profile? PHP 5.0.0 spins forever if it does not.
 *
 * WHY THIS PROBE EXISTS. CRASH-092 never returns here. It is the one case in the ASAN-blind
 * population that builds its own listener, and the first Phase B measurement lost the ten
 * cases after it alphabetically on two arms because every run of an arm shares one boot.
 * Reading the case does not explain the hang on its own: it sets
 * `ini_set('default_socket_timeout', 1)`, and main/streams/xp_socket.c:726 copies that into
 * the socket's own timeout, so the read of a connection nobody will answer should give up
 * after one second.
 *
 * It does not, and php_sock_stream_wait_for_data (xp_socket.c:95-106) is why:
 *
 *     while(1) {
 *         retval = select(sock->socket + 1, &tfdr, NULL, NULL, ptimeout);
 *         if (retval == 0) sock->timeout_event = 1;
 *         if (retval >= 0) break;
 *     }
 *
 * A select() that returns 0 breaks the loop with the timeout flag set, and a select() that
 * returns a positive count breaks it too. A select() that returns -1 does NOT break it, and
 * nothing in the loop inspects errno or counts attempts, so ONE PERSISTENT ERROR IS AN
 * INFINITE LOOP. That is PHP's own bug -- a later PHP checks for EINTR and gives up on
 * anything else -- but it makes the platform question sharp and worth answering separately:
 * if select() is broken here, every port that waits on a descriptor inherits the hang, not
 * just this case, and the fix is far below PHP.
 *
 * So: four stages, each printing what it got AND errno, because "select returned -1" and
 * "select returned -1 with EFAULT" are different findings. A stage that cannot even set up
 * its descriptors says so and does not let the stages after it report a pass.
 *
 * This probe is deliberately NOT a capability test. It calls no capability builtin and makes
 * no claim about bounds; the whole question is whether an ordinary blocking-I/O primitive
 * survives the trip through CSRUNV, where the kernel is handed pointers that the application
 * holds as capabilities. Failures of that shape are invisible to a compile survey.
 */
#include <errno.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdio.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

static int failures;

static void report(const char *stage, int got, int want, const char *meaning)
{
    if (got == want) {
        printf("SELECT_PROBE ok   %-22s -> %d (%s)\n", stage, got, meaning);
    } else {
        printf("SELECT_PROBE FAIL %-22s -> %d errno=%d (%s), wanted %d (%s)\n",
               stage, got, errno, strerror(errno), want, meaning);
        failures++;
    }
    fflush(stdout);
}

/* Stage 1: an idle descriptor with a timeout must TIME OUT, returning 0. This is the exact
 * call PHP makes and the exact answer PHP needs to escape its loop. */
static void idle_times_out(int fd)
{
    fd_set r;
    struct timeval tv = { .tv_sec = 1, .tv_usec = 0 };
    FD_ZERO(&r);
    FD_SET(fd, &r);
    report("select idle 1s", select(fd + 1, &r, NULL, NULL, &tv), 0, "timed out");
}

/* Stage 2: a descriptor with a byte waiting must be reported READY. A select() that times out
 * on everything would pass stage 1 while being just as broken. */
static void ready_is_seen(int fd, int peer)
{
    fd_set r;
    struct timeval tv = { .tv_sec = 1, .tv_usec = 0 };
    if (write(peer, "x", 1) != 1) {
        printf("SELECT_PROBE FAIL %-22s -> could not write to the peer, errno=%d (%s)\n",
               "setup for ready", errno, strerror(errno));
        failures++;
        return;
    }
    FD_ZERO(&r);
    FD_SET(fd, &r);
    report("select ready", select(fd + 1, &r, NULL, NULL, &tv), 1, "one descriptor ready");
}

/* Stage 3: a NULL timeout is a different argument shape -- PHP takes it whenever the socket
 * timeout is -1 -- so it is worth one call of its own, on a descriptor that is already ready
 * so the probe cannot itself hang. */
static void null_timeout(int fd)
{
    fd_set r;
    FD_ZERO(&r);
    FD_SET(fd, &r);
    report("select NULL timeout", select(fd + 1, &r, NULL, NULL, NULL), 1, "one descriptor ready");
}

/* Stage 4: poll(), for comparison. If select() is broken and poll() is not, the finding is
 * about one syscall's argument marshalling rather than about waiting in general. */
static void poll_idle(int fd)
{
    struct pollfd p = { .fd = fd, .events = POLLIN, .revents = 0 };
    report("poll idle 1000ms", poll(&p, 1, 1000), 0, "timed out");
}

int main(void)
{
    int sv[2];
    char drain;

    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        printf("SELECT_PROBE FAIL socketpair -> errno=%d (%s); no stage can run\n",
               errno, strerror(errno));
        return 2;
    }
    idle_times_out(sv[0]);
    ready_is_seen(sv[0], sv[1]);
    null_timeout(sv[0]);
    if (read(sv[0], &drain, 1) != 1) {
        printf("SELECT_PROBE FAIL drain -> errno=%d (%s); the poll stage would not be idle\n",
               errno, strerror(errno));
        failures++;
    } else {
        poll_idle(sv[0]);
    }
    close(sv[0]);
    close(sv[1]);

    printf("SELECT_PROBE DONE failures=%d\n", failures);
    fflush(stdout);
    /* A non-zero exit, so a harness that only reads the status still learns the answer. */
    return failures ? 1 : 0;
}
