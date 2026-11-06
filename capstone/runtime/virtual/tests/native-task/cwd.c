/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int checks, failures;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; fprintf(stderr, "CWD_FAIL line=%d errno=%d\n", __LINE__, errno); } } while (0)
static void exchange(char c)
{
    char answer = 0;
    CHECK(write(1, &c, 1) == 1);
    CHECK(read(0, &answer, 1) == 1 && answer == c);
}
static void contents(const char *name, int want)
{
    char c = 0;
    int fd = open(name, O_RDONLY);
    CHECK(fd >= 0);
    if (fd < 0) return;
    CHECK(read(fd, &c, 1) == 1 && c == want);
    CHECK(close(fd) == 0);
}
static void unchanged(const char *expected)
{
    char cwd[PATH_MAX];
    CHECK(getcwd(cwd, sizeof cwd) != NULL && !strcmp(cwd, expected));
}
static void refused(const char *name, const char *expected)
{
    CHECK(chdir(name) == -1);
    unchanged(expected);
}

#ifdef CWD_RAW
extern long __capstone_native_syscall(long, void *, void *, void *, void *, void *, void *);
static void *scalar(unsigned long n)
{
    void *r;
    __asm__ volatile("movc %0, zero\n\tmv %0, %1" : "=&r"(r) : "r"(n));
    return r;
}
static void raw_checks(void)
{
    char path[] = "a";
    unsigned long base = __builtin_capstone_cap_get_cursor(path);
    void *tiny = __builtin_capstone_cap_shrink(path, base, base + 1);
    CHECK(__capstone_native_syscall(49, scalar(base), 0, 0, 0, 0, 0) == -EFAULT);
    CHECK(__capstone_native_syscall(49, tiny, 0, 0, 0, 0, 0) == -EFAULT);
}
#endif

int main(void)
{
    char original[PATH_MAX], root[PATH_MAX], cwd[PATH_MAX], path[PATH_MAX];
    CHECK(getcwd(original, sizeof original) != NULL);
    strcpy(root, original);
#ifdef CWD_RAW
    raw_checks();
    unchanged(root);
#endif
    errno = 0;
    CHECK(getcwd(cwd, 1) == NULL && errno == ERANGE);
    CHECK(chdir("a//./b/") == 0);
    contents("../marker", 'A');
    contents("../../marker", 'R');
    CHECK(chdir("../..") == 0);
    unchanged(root);
    refused("..", root);
    refused("a/../../outside", root);
    refused("escape", root);
    refused("mounted", root);
    refused("missing", root);
    refused("marker", root);
    refused("", root);
    CHECK(open("a/../../outside/marker", O_RDONLY) == -1);
    CHECK(open("escape/marker", O_RDONLY) == -1);
    CHECK(open("mounted/marker", O_RDONLY) == -1);
    int dir = open("a/b", O_RDONLY | O_DIRECTORY);
    CHECK(dir >= 0);
    int fd = openat(dir, "../marker", O_RDONLY);
    CHECK(fd >= 0);
    if (fd >= 0) {
        char c = 0;
        CHECK(read(fd, &c, 1) == 1 && c == 'A');
        CHECK(close(fd) == 0);
    }
    CHECK(openat(dir, "../../../outside/marker", O_RDONLY) == -1);
    CHECK(close(dir) == 0);

    CHECK(chdir("a/b") == 0);
    exchange('R'); /* supervisor renames a while it is an ancestor of cwd */
    snprintf(path, sizeof path, "%s/renamed/b", root);
    unchanged(path);
    contents("../marker", 'A');
    CHECK(chdir("../..") == 0);
    exchange('G'); /* supervisor renames the immutable grant itself */
    CHECK(getcwd(root, sizeof root) != NULL && strcmp(root, original));
    snprintf(path, sizeof path, "%s/marker", root);
    contents(path, 'R');
    snprintf(path, sizeof path, "%s/renamed/b", root);
    CHECK(chdir(path) == 0);
    exchange('O'); /* cwd's ancestor is now outside the grant */
    CHECK(getcwd(cwd, sizeof cwd) == NULL && errno == ENOENT);
    CHECK(chdir("..") == -1 && errno == EACCES);
    CHECK(open("../marker", O_RDONLY) == -1 && errno == EACCES);
    CHECK(chdir(root) == 0);
    unchanged(root);

    exchange('X'); /* supervisor moves race into/out of the grant */
    for (int i = 0; i < 2000; ++i) {
        int race = open("race/../marker", O_RDONLY);
        if (race >= 0) {
            char c = 0;
            CHECK(read(race, &c, 1) == 1 && c == 'R');
            CHECK(close(race) == 0);
        } else CHECK(errno == EAGAIN || errno == EACCES || errno == ENOENT);
    }
    exchange('Y');
    unchanged(root);
    fprintf(stderr, "NATIVE_CWD checks=%d failures=%d\n", checks, failures);
    return failures != 0;
}
