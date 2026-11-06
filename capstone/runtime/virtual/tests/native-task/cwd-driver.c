/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

static int in, out, checks;
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr, "CWD_DRIVER_FAIL line=%d errno=%d\n", __LINE__, errno); exit(1); } } while (0)
static void receive(char want)
{
    struct pollfd p = {.fd = in, .events = POLLIN};
    char c = 0;
    CHECK(poll(&p, 1, 30000) == 1 && read(in, &c, 1) == 1 && c == want);
}
static void answer(char c) { CHECK(write(out, &c, 1) == 1); }
static void marker(const char *path, char byte)
{
    int fd = open(path, O_CREAT | O_EXCL | O_WRONLY, 0600);
    CHECK(fd >= 0 && write(fd, &byte, 1) == 1 && close(fd) == 0);
}
int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    alarm(90);
    char binary[PATH_MAX], temp[] = "/tmp/native-cwd-XXXXXX";
    CHECK(realpath(argv[1], binary) != NULL);
    CHECK(mkdtemp(temp) != NULL && chdir(temp) == 0);
    CHECK(mkdir("grant", 0700) == 0 && mkdir("outside", 0700) == 0);
    CHECK(mkdir("grant/a", 0700) == 0 && mkdir("grant/a/b", 0700) == 0);
    CHECK(mkdir("grant/race", 0700) == 0 && mkdir("grant/mounted", 0700) == 0);
    marker("grant/marker", 'R'); marker("grant/a/marker", 'A'); marker("outside/marker", 'X');
    CHECK(symlink("../outside", "grant/escape") == 0);
    CHECK(mount("outside", "grant/mounted", NULL, MS_BIND, NULL) == 0);
    int a[2], b[2];
    CHECK(pipe(a) == 0 && pipe(b) == 0);
    pid_t child = fork();
    CHECK(child >= 0);
    if (!child) {
        if (chdir("grant") || dup2(a[0], 0) < 0 || dup2(b[1], 1) < 0) _exit(125);
        close(a[0]); close(a[1]); close(b[0]); close(b[1]);
        execl(binary, binary, NULL);
        _exit(126);
    }
    close(a[0]); close(b[1]); out = a[1]; in = b[0];
    receive('R'); CHECK(rename("grant/a", "grant/renamed") == 0); answer('R');
    receive('G'); CHECK(rename("grant", "new-grant") == 0); answer('G');
    receive('O'); CHECK(rename("new-grant/renamed", "outside/moved") == 0); answer('O');
    receive('X');
    unsigned long *moves = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    CHECK(moves != MAP_FAILED);
    pid_t mover = fork();
    CHECK(mover >= 0);
    if (!mover) {
        for (;;) {
            if (rename("new-grant/race", "outside/race") || rename("outside/race", "new-grant/race")) _exit(127);
            ++*(volatile unsigned long *)moves;
        }
    }
    answer('X'); receive('Y');
    CHECK(kill(mover, SIGKILL) == 0);
    int status;
    CHECK(waitpid(mover, &status, 0) == mover && WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL);
    CHECK(*(volatile unsigned long *)moves > 0);
    unsigned long total = *moves;
    CHECK(munmap(moves, 4096) == 0);
    answer('Y');
    CHECK(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    close(in); close(out);
    CHECK(umount("new-grant/mounted") == 0);
    /* Keep fixtures in scratch for inspection, including the untouched sentinel. */
    int fd = open("outside/marker", O_RDONLY);
    char c = 0;
    CHECK(fd >= 0 && read(fd, &c, 1) == 1 && c == 'X' && close(fd) == 0);
    printf("NATIVE_CWD_DRIVER checks=%d rename_pairs=%lu failures=0 scratch=%s\n", checks, total, temp);
    return 0;
}
