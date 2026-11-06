/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

static int input, output, checks;
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr, "SPAWN_DRIVER_FAIL line=%d errno=%d test=%s\n", __LINE__, errno, #x); exit(1); } } while (0)
static void copy_file(const char *from, const char *to)
{
    char buf[65536];
    int src = open(from, O_RDONLY), dst = open(to, O_CREAT | O_EXCL | O_WRONLY, 0700);
    CHECK(src >= 0 && dst >= 0);
    ssize_t n;
    while ((n = read(src, buf, sizeof buf)) > 0) {
        ssize_t at = 0;
        while (at < n) { ssize_t k = write(dst, buf + at, n - at); CHECK(k > 0); at += k; }
    }
    CHECK(n == 0 && close(src) == 0 && close(dst) == 0);
}
static pid_t start(const char *mode, int ambient_child, int ambient_fd)
{
    int a[2], b[2];
    CHECK(pipe(a) == 0 && pipe(b) == 0);
    pid_t p = fork();
    CHECK(p >= 0);
    if (!p) {
        struct rlimit core = {0, 0};
        if (setrlimit(RLIMIT_CORE, &core) || dup2(a[0], 0) < 0 || dup2(b[1], 1) < 0) _exit(125);
        close(a[0]); close(a[1]); close(b[0]); close(b[1]);
        if (ambient_fd && dup2(1, 199) != 199) _exit(124);
        if (ambient_child) {
            pid_t foreign = fork();
            if (foreign < 0) _exit(123);
            if (!foreign) _exit(61);
            siginfo_t info = {0};
            if (waitid(P_PID, foreign, &info, WEXITED | WNOWAIT)) _exit(122);
            uint64_t packet[2] = {'A', (uint64_t)foreign};
            if (write(1, packet, sizeof packet) != sizeof packet) _exit(121);
        }
        execl("./probe.dom", "probe.dom", mode, NULL);
        _exit(126);
    }
    close(a[0]); close(b[1]); output = a[1]; input = b[0];
    return p;
}
static void receive(uint64_t p[2])
{
    struct pollfd fd = {.fd = input, .events = POLLIN};
    size_t at = 0;
    while (at < 2 * sizeof(uint64_t)) {
        CHECK(poll(&fd, 1, 60000) == 1);
        ssize_t n = read(input, (char *)p + at, 2 * sizeof(uint64_t) - at);
        CHECK(n > 0);
        at += n;
    }
}
static pid_t receive_tag(char tag)
{
    uint64_t p[2];
    receive(p);
    CHECK(p[0] == (unsigned char)tag && p[1] > 0 && p[1] <= INT_MAX);
    return p[1];
}
static void answer(char c) { CHECK(write(output, &c, 1) == 1); }
static void status(pid_t p, int code, int sig)
{
    int s = 0;
    CHECK(waitpid(p, &s, 0) == p);
    CHECK(sig ? WIFSIGNALED(s) && WTERMSIG(s) == sig : WIFEXITED(s) && WEXITSTATUS(s) == code);
}
static void close_control(void)
{
    CHECK(close(input) == 0 && close(output) == 0);
}
static void no_children(void)
{
    int s;
    CHECK(waitpid(-1, &s, WNOHANG) == -1 && errno == ECHILD);
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
static void no_task_children(pid_t pid)
{
    char path[100], buf[128];
    snprintf(path, sizeof path, "/proc/%ld/task/%ld/children", (long)pid, (long)pid);
    int fd = open(path, O_RDONLY);
    CHECK(fd >= 0);
    CHECK(read(fd, buf, sizeof buf) == 0 && close(fd) == 0);
}
static void reparenting(void)
{
    pid_t parent = start("wait-parent", 0, 1), child = -1;
    unsigned seen = 0;
    for (unsigned i = 0; i < 2; ++i) {
        uint64_t p[2];
        receive(p);
        CHECK(p[1] > 0 && p[1] <= INT_MAX);
        if (p[0] == 'P') { CHECK(!(seen & 1)); seen |= 1; }
        else { CHECK(p[0] == 'H' && !(seen & 2)); seen |= 2; }
        CHECK(child == -1 || child == (pid_t)p[1]);
        child = p[1];
    }
    CHECK(seen == 3);
    CHECK(descriptors(parent) == 4 && descriptors(child) == 3);
    char path[100];
    snprintf(path, sizeof path, "/proc/%ld/fd/199", (long)child);
    CHECK(access(path, F_OK) == -1 && errno == ENOENT);
    /* H proves exec completed. This is native-task ptrace rejection, not a
     * claim about observing the earlier bootstrap interval. */
    errno = 0;
    CHECK(ptrace(PTRACE_ATTACH, child, NULL, NULL) == -1 && errno == EOPNOTSUPP);
    CHECK(kill(parent, SIGKILL) == 0);
    status(parent, 0, SIGKILL);
    /* Ordinary Linux ownership: killing a waiting parent does not kill its
     * successfully spawned child. The subreaper adopts it and can reap it. */
    answer('E');
    status(child, 23, 0);
    close_control();
    no_children();
}
int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    alarm(240);
    CHECK(prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0);
    char binary[PATH_MAX], scratch[] = "/tmp/native-spawn-lifecycle-XXXXXX";
    CHECK(realpath(argv[1], binary) != NULL);
    CHECK(mkdtemp(scratch) != NULL && chdir(scratch) == 0);
    copy_file(binary, "probe.dom");

    pid_t p = start("negative", 0, 0);
    CHECK(receive_tag('B') == p);
    CHECK(descriptors(p) == 3);
    unsigned long before = files();
    answer('B');
    CHECK(receive_tag('F') == p);
    CHECK(descriptors(p) == 3);
    no_task_children(p);
    unsigned long after = files();
    CHECK(after <= before + 16);
    answer('F'); status(p, 0, 0); close_control(); no_children();

    p = start("wait-any", 1, 0);
    pid_t foreign = receive_tag('A');
    status(p, 0, 0);
    status(foreign, 61, 0); /* native wait-any did not consume ambient child */
    close_control(); no_children();

    p = start("autoreap", 0, 0);
    status(p, 0, 0); close_control(); no_children();
    reparenting();

    before = files();
    for (unsigned i = 0; i < 32; ++i) {
        p = start("bad-meta", 0, 0);
        CHECK(receive_tag('M') == p);
        answer('M');
        status(p, 0, SIGSEGV);
        /* Keep input open: an uncancelled child would stay blocked instead
         * of escaping the check via EOF. Subreaper sees live AND dead leaks. */
        no_children();
        close_control();
    }
    unsigned long metadata_files = files();
    CHECK(metadata_files <= before + 16);
    printf("NATIVE_SPAWN_DRIVER checks=%d failures=0 failed_exec_files=%lu metadata_files=%lu metadata_cases=32 bootstrap_ptrace=untested bootstrap_blocked_cancel=untested scratch=%s\n",
           checks, after, metadata_files, scratch);
    return 0;
}
