/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

static int in, out, checks;
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr, "PROCESS_DRIVER_FAIL line=%d errno=%d\n", __LINE__, errno); exit(1); } } while (0)
static pid_t start(const char *binary, const char *mode, unsigned limit, unsigned bound)
{
    int a[2], b[2];
    CHECK(pipe(a) == 0 && pipe(b) == 0);
    pid_t p = fork();
    CHECK(p >= 0);
    if (!p) {
        struct rlimit r = {limit, limit}, core = {0, 0};
        char value[32];
        snprintf(value, sizeof value, "%u", bound);
        if (setrlimit(RLIMIT_NOFILE, &r) || setrlimit(RLIMIT_CORE, &core) ||
            dup2(a[0], 0) < 0 || dup2(b[1], 1) < 0) _exit(125);
        close(a[0]); close(a[1]); close(b[0]); close(b[1]);
        execl(binary, binary, mode, value, NULL);
        _exit(126);
    }
    close(a[0]); close(b[1]); out = a[1]; in = b[0];
    return p;
}
static void receive(char want)
{
    struct pollfd p = {.fd = in, .events = POLLIN};
    char c = 0;
    CHECK(poll(&p, 1, 30000) == 1 && read(in, &c, 1) == 1 && c == want);
}
static void answer(char c) { CHECK(write(out, &c, 1) == 1); }
static void done(pid_t p, int signal)
{
    int status;
    CHECK(waitpid(p, &status, 0) == p);
    CHECK(signal ? WIFSIGNALED(status) && WTERMSIG(status) == signal :
                   WIFEXITED(status) && WEXITSTATUS(status) == 0);
    CHECK(close(in) == 0 && close(out) == 0);
}
static unsigned long files(void)
{
    unsigned long n;
    FILE *f = fopen("/proc/sys/fs/file-nr", "r");
    CHECK(f != NULL && fscanf(f, "%lu", &n) == 1 && fclose(f) == 0);
    return n;
}
static int descriptors(pid_t pid)
{
    char path[80];
    snprintf(path, sizeof path, "/proc/%ld/fd", (long)pid);
    DIR *d = opendir(path);
    CHECK(d != NULL);
    int n = 0;
    struct dirent *e;
    while ((e = readdir(d))) if (e->d_name[0] != '.') ++n;
    CHECK(closedir(d) == 0);
    return n;
}
int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    alarm(180);
    pid_t p = start(argv[1], "basic", 2048, 1024);
    receive('I');
    CHECK(descriptors(p) == 3);
    CHECK(kill(p, SIGUSR1) == 0);
    answer('I'); done(p, 0);
    unsigned long baseline = files();
    p = start(argv[1], "rollback", 2048, 1024); done(p, 0);
    unsigned long after_rollback = files();
    CHECK(after_rollback <= baseline + 16);
    p = start(argv[1], "limit", 32, 32); done(p, 0);
    p = start(argv[1], "limit", 2048, 1024); done(p, 0);
    p = start(argv[1], "default", 2048, 1024);
    receive('D'); CHECK(kill(p, SIGUSR2) == 0); done(p, SIGUSR2);
    p = start(argv[1], "stop-kill", 2048, 1024);
    receive('S');
    int status;
    CHECK(kill(p, SIGSTOP) == 0 && waitpid(p, &status, WUNTRACED) == p && WIFSTOPPED(status));
    struct pollfd blocked = {.fd = in, .events = POLLIN};
    CHECK(poll(&blocked, 1, 50) == 0);
    CHECK(kill(p, SIGCONT) == 0 && waitpid(p, &status, WCONTINUED) == p && WIFCONTINUED(status));
    answer('S'); receive('K'); CHECK(kill(p, SIGKILL) == 0); done(p, SIGKILL);
    /* The first metadata reply succeeds; its second, signal-count copy
     * fails. Repeating beyond the file counter's batching quantum detects
     * leaked pipe-file references even though each task then exits. */
    baseline = files();
    for (int i = 0; i < 64; ++i) {
        p = start(argv[1], "bad-meta", 2048, 1024); done(p, SIGSEGV);
    }
    unsigned long after_metadata = files();
    CHECK(after_metadata <= baseline + 16);
    printf("NATIVE_PROCESS_DRIVER checks=%d payload_files=%lu metadata_files=%lu failures=0\n",
           checks, after_rollback, after_metadata);
    return 0;
}
