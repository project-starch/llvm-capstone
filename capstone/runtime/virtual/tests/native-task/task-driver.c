/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
static int failures;
#define CHECK(name, expr) do { int ok = !!(expr); printf("%s %s\n", ok ? "PASS" : "FAIL", name); failures += !ok; } while (0)
static pid_t launch(const char *image, int *input)
{
    int in[2], out[2];
    char byte = 0;
    if (pipe(in) || pipe(out)) exit(2);
    pid_t pid = fork();
    if (!pid) {
        dup2(in[0], 0); dup2(out[1], 1);
        close(in[0]); close(in[1]); close(out[0]); close(out[1]);
        execl(image, image, (char *)0); _exit(99);
    }
    close(in[0]); close(out[1]); *input = in[1];
    CHECK("task readiness", read(out[0], &byte, 1) == 1 && byte == 'R');
    close(out[0]); return pid;
}
static unsigned long user_ticks(pid_t pid)
{
    char path[80], line[4096], *p, *save;
    snprintf(path, sizeof(path), "/proc/%ld/stat", (long)pid);
    FILE *f = fopen(path, "r");
    if (!f || !fgets(line, sizeof(line), f)) exit(3);
    fclose(f); p = strrchr(line, ')'); if (!p) exit(3);
    p = strtok_r(p + 2, " ", &save);
    for (int field = 3; p; ++field, p = strtok_r(NULL, " ", &save))
        if (field == 14) return strtoul(p, NULL, 10);
    exit(3);
}
int main(int argc, char **argv)
{
    if (argc != 3) return 2;
    alarm(30);
    errno = 0;
    CHECK("legacy device absent", access("/dev/capstone-vm", F_OK) == -1 && errno == ENOENT);
    errno = 0;
    CHECK("legacy misc registration absent",
          access("/sys/class/misc/capstone-vm", F_OK) == -1 && errno == ENOENT);
    int input, st;
    struct rusage usage;
    pid_t pid = launch(argv[1], &input);
    errno = 0;
    long traced = ptrace(PTRACE_SEIZE, pid, 0, 0);
    CHECK("ptrace rejected", traced == -1 && errno == EOPNOTSUPP);
    if (!traced) { kill(pid, SIGKILL); waitpid(pid, &st, 0); return 1; }
    char path[80], line[512]; int tramp = 0, fd_count = 0;
    snprintf(path, sizeof(path), "/proc/%ld/maps", (long)pid);
    FILE *f = fopen(path, "r");
    if (!f) return 3;
    while (fgets(line, sizeof(line), f)) tramp |= strstr(line, "capstone-trampoline") != NULL;
    fclose(f); CHECK("no trampoline mapping", !tramp);
    /* Only stdin/stdout/stderr survive this fixture's exec. */
    for (int fd = 3; fd < 32; ++fd) {
        snprintf(path, sizeof(path), "/proc/%ld/fd/%d", (long)pid, fd);
        fd_count += readlink(path, line, sizeof(line)) >= 0;
    }
    CHECK("no control descriptor", fd_count == 0);
    usleep(250000);
    unsigned long before = user_ticks(pid);
    CHECK("compute charged to user", before > 0);
    CHECK("compute SIGSTOP", !kill(pid, SIGSTOP) && waitpid(pid, &st, WUNTRACED) == pid && WIFSTOPPED(st));
    unsigned long stopped = user_ticks(pid); usleep(100000);
    CHECK("stopped task does not execute", user_ticks(pid) == stopped);
    CHECK("compute SIGCONT", !kill(pid, SIGCONT) && waitpid(pid, &st, WCONTINUED) == pid && WIFCONTINUED(st));
    usleep(250000); CHECK("compute resumes", user_ticks(pid) > stopped);
    CHECK("compute SIGKILL", !kill(pid, SIGKILL) && wait4(pid, &st, 0, &usage) == pid && WIFSIGNALED(st) && WTERMSIG(st) == SIGKILL);
    printf("compute usage user=%ld.%06ld system=%ld.%06ld\n", usage.ru_utime.tv_sec, usage.ru_utime.tv_usec, usage.ru_stime.tv_sec, usage.ru_stime.tv_usec);
    close(input);
    for (int n = 0; n < 12; ++n) {
        pid = launch(argv[2], &input);
        CHECK("blocked task SIGKILL/reclaim", !kill(pid, SIGKILL) && waitpid(pid, &st, 0) == pid && WIFSIGNALED(st) && WTERMSIG(st) == SIGKILL);
        close(input);
    }
    printf("native task failures=%d\n", failures);
    return failures != 0;
}
