#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

static int failures;
static const char *image;
static void check(const char *name, int ok)
{ printf("%s %s\n", ok ? "PASS" : "FAIL", name); failures += !ok; }
static pid_t launch(const char *mode, int in, int out)
{
    pid_t pid = fork();
    if (!pid) {
        if (dup2(in, 0) < 0 || dup2(out, 1) < 0) _exit(124);
        execl(image, image, mode, (char *)0);
        _exit(125);
    }
    return pid;
}
static void simple(const char *mode, int expected, size_t length)
{
    int in = open("/tmp/dom-input", O_RDONLY), out = open("/tmp/dom-output", O_CREAT | O_TRUNC | O_RDWR, 0600);
    int st = -1;
    pid_t pid = launch(mode, in, out);
    int ok = pid > 0 && waitpid(pid, &st, 0) == pid && WIFEXITED(st) && WEXITSTATUS(st) == expected;
    struct stat sb;
    ok = ok && !fstat(out, &sb) && sb.st_size == length;
    if (ok && length) {
        char b[4096]; size_t at = 0; ssize_t n;
        lseek(out, 0, SEEK_SET);
        while ((n = read(out, b, sizeof b)) > 0) {
            for (ssize_t i = 0; i < n; ++i, ++at) {
                char want = !strcmp(mode, "malloc") ? 'a' + at % 26 :
                            !strcmp(mode, "read") ? "test"[at] : "native hello\n"[at];
                if (b[i] != want) ok = 0;
            }
        }
        ok = ok && at == length;
    }
    if (!ok) printf("status=%d output_size=%lld\n", st, (long long)(!fstat(out, &sb) ? sb.st_size : -1));
    check(mode, ok); close(in); close(out);
}
static void signals(void)
{
    int p[2], st = -1, in = open("/dev/null", O_RDONLY);
    char b[6];
    if (pipe(p)) { check("signals-pipe", 0); return; }
    pid_t pid = launch("spin", in, p[1]); close(p[1]); close(in);
    if (pid <= 0 || read(p[0], b, 6) != 6 || memcmp(b, "ready\n", 6)) {
        check("signals-ready", 0); if (pid > 0) kill(pid, SIGKILL); goto done;
    }
    char path[128], data[4096];
    snprintf(path, sizeof path, "/proc/%d/task/%d/children", pid, pid);
    int fd = open(path, O_RDONLY); ssize_t n = fd < 0 ? -1 : read(fd, data, sizeof data);
    if (fd >= 0) close(fd);
    check("no-helper-children", n == 0);
    snprintf(path, sizeof path, "/proc/%d/status", pid);
    fd = open(path, O_RDONLY); n = fd < 0 ? -1 : read(fd, data, sizeof data - 1);
    if (fd >= 0) close(fd);
    if (n >= 0) data[n] = 0;
    check("one-linux-thread", n > 0 && strstr(data, "Threads:\t1\n"));
    check("stop", !kill(pid, SIGSTOP) && waitpid(pid, &st, WUNTRACED) == pid && WIFSTOPPED(st));
    check("continue", !kill(pid, SIGCONT) && waitpid(pid, &st, WCONTINUED) == pid && WIFCONTINUED(st));
    check("terminate-wait", !kill(pid, SIGTERM) && waitpid(pid, &st, 0) == pid && WIFSIGNALED(st) && WTERMSIG(st) == SIGTERM);
    pid = -1;
done:
    if (pid > 0) waitpid(pid, &st, 0);
    close(p[0]);
}
static void refused(const char *suffix, int error)
{
    char path[256]; int st = -1;
    snprintf(path, sizeof path, "%s.%s", image, suffix);
    pid_t pid = fork();
    if (!pid) { execl(path, path, "hello", (char *)0); _exit(errno == error ? 0 : 1); }
    check(suffix, pid > 0 && waitpid(pid, &st, 0) == pid && WIFEXITED(st) && WEXITSTATUS(st) == 0);
}
int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    image = argv[1]; alarm(120); chdir("/tmp"); setenv("DOM_TEST", "present", 1);
    int fd = open("/tmp/dom-input", O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (fd < 0 || write(fd, "test", 4) != 4) return 3;
    close(fd);
    simple("hello", 0, 13); simple("loop", 0, 0); simple("malloc", 0, 1U << 20);
    simple("mmap", 0, 0); simple("read", 0, 4); simple("unsupported", 0, 0);
    simple("protect-refuse", 0, 0); simple("bad-fd", 0, 0);
    simple("startup", 0, 0); simple("exit", 37, 0); signals();
    refused("truncated", ENOEXEC); refused("marker", ENOEXEC);
    refused("overflow", ENOEXEC); refused("wx", ENOEXEC); refused("setuid", EACCES);
    printf("native contract failures=%d\n", failures);
    return failures != 0;
}
