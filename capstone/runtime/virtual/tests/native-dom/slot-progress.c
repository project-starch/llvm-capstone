/* SPDX-License-Identifier: GPL-2.0-only */
/* Plain parent: both native children must progress while the first is alive. */
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
        execl(image, image, "spin-progress", (char *)0);
        _exit(125);
    }
    close(p[1]);
    *fd = p[0];
    return child;
}
static int ready(int fd)
{
    struct pollfd p = {.fd = fd, .events = POLLIN};
    char line[6];
    return poll(&p, 1, 3000) > 0 && read(fd, line, sizeof(line)) == sizeof(line) &&
           !memcmp(line, "ready\n", sizeof(line));
}
static void drain(int fd)
{
    struct pollfd p = {.fd = fd, .events = POLLIN};
    char buffer[256];
    while (poll(&p, 1, 0) > 0 && (p.revents & POLLIN))
        if (read(fd, buffer, sizeof(buffer)) <= 0) break;
}
static int stop(pid_t child)
{
    int status;
    if (child <= 0) return 0;
    kill(child, SIGCONT);
    kill(child, SIGTERM);
    return waitpid(child, &status, 0) == child &&
           WIFSIGNALED(status) && WTERMSIG(status) == SIGTERM;
}
int main(int argc, char **argv)
{
    if (argc != 2 || chdir("/tmp")) return 2;
    alarm(60);
    for (unsigned round = 0; round < 8; ++round) {
        int a_fd = -1, b_fd = -1, status = 0;
        pid_t a = start(argv[1], &a_fd), b = -1;
        int first = a > 0 && ready(a_fd), second = 0, stopped = 0;
        if (first) {
            kill(a, SIGSTOP);
            stopped = waitpid(a, &status, WUNTRACED) == a && WIFSTOPPED(status);
            b = start(argv[1], &b_fd);
            second = b > 0 && ready(b_fd);
            /* No old output may satisfy the resume check. */
            drain(a_fd);
            kill(a, SIGCONT);
            if (second) second = ready(a_fd) && ready(b_fd);
        }
        int reaped = stop(a);
        if (b > 0) reaped &= stop(b);
        if (a_fd >= 0) close(a_fd);
        if (b_fd >= 0) close(b_fd);
        int ok = first && second && stopped && reaped;
        printf("%s slot-progress round=%u first=%d second=%d stopped=%d reaped=%d\n",
               ok ? "PASS" : "FAIL", round, first, second, stopped, reaped);
        if (!ok) return 1;
    }
    return 0;
}
