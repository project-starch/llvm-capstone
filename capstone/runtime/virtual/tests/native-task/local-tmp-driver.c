/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <ftw.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

static int checks, failures;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; fprintf(stderr, "LOCAL_TMP_DRIVER_FAIL line=%d errno=%d test=%s\n", __LINE__, errno, #x); } } while (0)
static void file(const char *name, const char *text)
{
    int fd = open(name, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0 || write(fd, text, strlen(text)) != (ssize_t)strlen(text) || close(fd)) exit(3);
}
static void copy(const char *src, const char *dst)
{
    char buf[8192];
    int in = open(src, O_RDONLY), out = open(dst, O_WRONLY | O_CREAT | O_EXCL, 0700);
    ssize_t n;
    if (in < 0 || out < 0) exit(3);
    while ((n = read(in, buf, sizeof buf)) > 0) if (write(out, buf, n) != n) exit(3);
    if (n < 0 || close(in) || close(out)) exit(3);
}
static void contents(const char *path, const char *want)
{
    char buf[64] = {0};
    int fd = open(path, O_RDONLY);
    CHECK(fd >= 0);
    if (fd < 0) return;
    CHECK(read(fd, buf, sizeof buf) == (ssize_t)strlen(want) && !strcmp(buf, want));
    CHECK(close(fd) == 0);
}
static int remove_entry(const char *name, const struct stat *st, int type, struct FTW *walk)
{
    (void)st; (void)type; (void)walk;
    return remove(name);
}
static void refused_exec(const char *root, const char *image)
{
    int status;
    pid_t child = fork();
    CHECK(child >= 0);
    if (!child) {
        if (chdir(root)) _exit(98);
        execl(image, image, "entry", "/tmp/unused", (char *)0);
        _exit(errno == EACCES || errno == ENOENT || errno == ENOTDIR || errno == EINVAL ? 80 : 99);
    }
    if (child > 0) CHECK(waitpid(child, &status, 0) == child &&
        ((WIFSIGNALED(status) && WTERMSIG(status) == SIGSEGV) ||
         (WIFEXITED(status) && WEXITSTATUS(status) == 80)));
}
static void rejected(const char *base, const char *image, const char *kind)
{
    char root[256], path[512];
    snprintf(root, sizeof root, "%s/%s", base, kind);
    CHECK(mkdir(root, 0700) == 0);
    snprintf(path, sizeof path, "%s/.capstone-tmp", root);
    if (!strcmp(kind, "symlink")) CHECK(symlink(base, path) == 0);
    else if (strcmp(kind, "missing")) {
        CHECK(mkdir(path, !strcmp(kind, "public") ? 0755 : 0700) == 0);
        if (!strcmp(kind, "public")) CHECK(chmod(path, 0755) == 0);
        if (!strcmp(kind, "wrong-owner")) CHECK(chown(path, 1234, 1234) == 0);
        if (!strcmp(kind, "mount")) CHECK(mount("tmpfs", path, "tmpfs", 0, "mode=0700") == 0);
    }
    refused_exec(root, image);
    if (!strcmp(kind, "mount")) CHECK(umount(path) == 0);
}
int main(int argc, char **argv)
{
    if (argc != 3 || (strcmp(argv[2], "on") && strcmp(argv[2], "on-null") && strcmp(argv[2], "off") && strcmp(argv[2], "conflict"))) return 2;
    int enabled = strcmp(argv[2], "off") && strcmp(argv[2], "conflict");
    alarm(45);
    char base[] = "/tmp/local-tmp-XXXXXX", image[PATH_MAX], root[128], path[512];
    char scratch[192], moved[192], outside[128], host[128], mountpoint[256], moved_out[128];
    if (!realpath(argv[1], image) || !mkdtemp(base)) return 3;
    snprintf(root, sizeof root, "%s/root", base); CHECK(mkdir(root, 0700) == 0);
    snprintf(scratch, sizeof scratch, "%s/.capstone-tmp", root); CHECK(mkdir(scratch, 0700) == 0);
    if (!strcmp(argv[2], "conflict")) {
        refused_exec(root, image);
        CHECK(nftw(base, remove_entry, 16, FTW_DEPTH | FTW_PHYS) == 0);
        fprintf(stderr, "NATIVE_LOCAL_TMP_DRIVER mode=conflict checks=%d failures=%d\n", checks, failures);
        return failures != 0;
    }
    snprintf(moved, sizeof moved, "%s/scratch-renamed", root);
    snprintf(moved_out, sizeof moved_out, "%s/scratch-outside", base);
    snprintf(outside, sizeof outside, "%s/outside", base); CHECK(mkdir(outside, 0700) == 0);
    snprintf(host, sizeof host, "/tmp/local-tmp-host-%ld", (long)getpid()); file(host, "host");
    snprintf(path, sizeof path, "%s/marker", outside); file(path, "outside");
    snprintf(path, sizeof path, "%s/root-marker", root); file(path, "root");
    snprintf(path, sizeof path, "%s/subdir", root); CHECK(mkdir(path, 0700) == 0);
    snprintf(path, sizeof path, "%s/probe.dom", root); copy(image, path);
    snprintf(path, sizeof path, "%s/escape", scratch); CHECK(symlink(outside, path) == 0);
    snprintf(mountpoint, sizeof mountpoint, "%s/mounted", scratch); CHECK(mkdir(mountpoint, 0700) == 0);
    CHECK(mount("tmpfs", mountpoint, "tmpfs", 0, "mode=0700") == 0);
    snprintf(path, sizeof path, "%s/marker", mountpoint); file(path, "mounted");
    int in[2], out[2], status;
    CHECK(pipe(in) == 0 && pipe(out) == 0);
    pid_t child = fork(); CHECK(child >= 0);
    if (!child) {
        if (chdir(root) || dup2(in[0], 0) < 0 || dup2(out[1], 1) < 0) _exit(98);
        close(in[0]); close(in[1]); close(out[0]); close(out[1]);
        execl("./probe.dom", "./probe.dom", argv[2], host, (char *)0); _exit(99);
    }
    close(in[0]); close(out[1]);
    if (enabled) {
        char ready = 0;
        CHECK(read(out[0], &ready, 1) == 1 && ready == 'R');
        CHECK(rename(scratch, moved) == 0);
        CHECK(symlink(outside, scratch) == 0);
        CHECK(write(in[1], "R", 1) == 1);
        ready = 0;
        CHECK(read(out[0], &ready, 1) == 1 && ready == 'O');
        CHECK(rename(moved, moved_out) == 0);
        CHECK(write(in[1], "O", 1) == 1);
        snprintf(mountpoint, sizeof mountpoint, "%s/mounted", moved_out);
    }
    close(in[1]); close(out[0]);
    CHECK(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    contents(host, "host");
    snprintf(path, sizeof path, "%s/marker", outside); contents(path, "outside");
    snprintf(path, sizeof path, "%s/after-rename", outside); CHECK(access(path, F_OK) == -1 && errno == ENOENT);
    if (enabled) {
        snprintf(path, sizeof path, "%s/after-rename", moved_out); contents(path, "pinned");
        for (unsigned i = 0; i < 5; ++i) {
            const char *kinds[] = {"missing", "symlink", "public", "wrong-owner", "mount"};
            rejected(base, image, kinds[i]);
        }
        int made = mkdir("/tmp/.capstone-tmp", 0700);
        CHECK(made == 0);
        if (!made) {
            refused_exec("/tmp", image); /* Otherwise valid scratch, ambiguous launch root. */
            CHECK(rmdir("/tmp/.capstone-tmp") == 0);
        }
    }
    CHECK(umount(mountpoint) == 0);
    CHECK(unlink(host) == 0);
    CHECK(nftw(base, remove_entry, 16, FTW_DEPTH | FTW_PHYS) == 0);
    fprintf(stderr, "NATIVE_LOCAL_TMP_DRIVER mode=%s checks=%d failures=%d\n", argv[2], checks, failures);
    return failures != 0;
}
