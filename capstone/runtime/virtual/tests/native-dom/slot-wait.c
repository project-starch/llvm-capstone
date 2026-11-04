/* SPDX-License-Identifier: GPL-2.0-only */
/* Plain parent; two native children contend for one physical slot. */
#define _GNU_SOURCE
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static pid_t start(const char *image, int *fd)
{
    int p[2];
    if (pipe(p)) return -1;
    pid_t child = fork();
    if (!child) {
        close(p[0]);
        if (dup2(p[1], 1) < 0) _exit(124);
        close(p[1]);
        execl(image, image, "spin", (char *)0);
        _exit(125);
    }
    close(p[1]);
    *fd = p[0];
    return child;
}

static int ready(int fd, int timeout)
{
    struct pollfd p = {.fd = fd, .events = POLLIN};
    char line[6];
    return poll(&p, 1, timeout) > 0 &&
           read(fd, line, sizeof(line)) == sizeof(line) &&
           !memcmp(line, "ready\n", sizeof(line));
}

static int stop(pid_t child)
{
    int status;
    if (child <= 0) return 0;
    kill(child, SIGTERM);
    return waitpid(child, &status, 0) == child &&
           WIFSIGNALED(status) && WTERMSIG(status) == SIGTERM;
}

int main(int argc, char **argv)
{
    int failed = 0;
    if (argc != 2 || chdir("/tmp")) return 2;
    /* Also bound regressions where the second task cannot be interrupted. */
    alarm(60);
    for (unsigned round = 0; round < 8; ++round) {
        int a_fd = -1, b_fd = -1;
        pid_t a = start(argv[1], &a_fd), b = -1;
        int first = a > 0 && ready(a_fd, 5000);
        if (first) b = start(argv[1], &b_fd);
        int excluded = b > 0 && !ready(b_fd, 50);
        int woke = 0, interrupted = 0;
        if (round & 1) {
            interrupted = stop(b);
            b = -1;
            woke = stop(a);
            a = -1;
        } else {
            interrupted = stop(a);
            a = -1;
            woke = b > 0 && ready(b_fd, 5000);
        }
        int ok = first && excluded && woke && interrupted;
        printf("%s slot-wait round=%u excluded=%d resumed=%d reaped=%d\n",
               ok ? "PASS" : "FAIL", round, excluded, woke, interrupted);
        failed += !ok;
        if (a > 0) failed += !stop(a);
        if (b > 0) failed += !stop(b);
        if (a_fd >= 0) close(a_fd);
        if (b_fd >= 0) close(b_fd);
    }
    return failed ? 1 : 0;
}
