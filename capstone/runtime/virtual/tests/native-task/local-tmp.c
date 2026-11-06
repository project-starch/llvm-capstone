/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;
static int checks, failures;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; fprintf(stderr, "LOCAL_TMP_FAIL line=%d errno=%d test=%s\n", __LINE__, errno, #x); } } while (0)
static void contents(const char *path, const char *want)
{
    char buf[64] = {0};
    int fd = open(path, O_RDONLY);
    CHECK(fd >= 0);
    if (fd < 0) return;
    CHECK(read(fd, buf, sizeof buf) == (ssize_t)strlen(want) && !strcmp(buf, want));
    CHECK(close(fd) == 0);
}
static void create(const char *path, const char *text)
{
    int fd = open(path, O_RDWR | O_CREAT | O_EXCL, 0600);
    CHECK(fd >= 0);
    if (fd < 0) return;
    CHECK(write(fd, text, strlen(text)) == (ssize_t)strlen(text));
    CHECK(close(fd) == 0);
}
int main(int argc, char **argv)
{
    if (argc != 3) return 2;
    if (!strcmp(argv[1], "entry")) return 42; /* Admission-negative supervisor control. */
    if (!strcmp(argv[1], "child")) {
        create("/tmp/inherited", "child");
        fprintf(stderr, "NATIVE_LOCAL_TMP_CHILD checks=%d failures=%d\n", checks, failures);
        return failures ? 1 : 23;
    }
    char root[PATH_MAX], physical[PATH_MAX], buf[64] = {0};
    struct stat st, other;
    CHECK(getcwd(root, sizeof root) != NULL);
    snprintf(physical, sizeof physical, "%s/root-marker", root);
    contents(physical, "root"); /* Longer original-root prefix wins over /tmp. */
    CHECK(open(argv[2], O_RDONLY) == -1); /* Host /tmp is never the alias target. */
    if (!strcmp(argv[1], "off")) {
        CHECK(tmpfile() == NULL);
        snprintf(physical, sizeof physical, "%s/.capstone-tmp/physical", root);
        create(physical, "physical");
        CHECK(unlink(physical) == 0);
        fprintf(stderr, "NATIVE_LOCAL_TMP mode=off checks=%d failures=%d\n", checks, failures);
        return failures != 0;
    }
    CHECK(open("/tmp/../outside", O_RDONLY) == -1);
    CHECK(open("/tmp/escape/marker", O_RDONLY) == -1);
    CHECK(open("/tmp/mounted/marker", O_RDONLY) == -1);
    int nullfd = open("/dev/null", O_RDWR);
    if (!strcmp(argv[1], "on-null")) { CHECK(nullfd >= 0); if (nullfd >= 0) CHECK(close(nullfd) == 0); }
    else CHECK(nullfd == -1);
    FILE *tmp = tmpfile();
    CHECK(tmp != NULL);
    if (tmp) {
        CHECK(fputs("temporary", tmp) >= 0);
        CHECK(fflush(tmp) == 0);
        CHECK(fstat(fileno(tmp), &st) == 0 && st.st_nlink == 0);
        CHECK(fseek(tmp, 0, SEEK_SET) == 0);
        CHECK(fread(buf, 1, 9, tmp) == 9 && !memcmp(buf, "temporary", 9));
        CHECK(fclose(tmp) == 0);
    }
    int fd = open("/tmp/named", O_RDWR | O_CREAT | O_EXCL, 0600);
    CHECK(fd >= 0);
    if (fd >= 0) {
        CHECK(write(fd, "live", 4) == 4);
        CHECK(unlink("/tmp/named") == 0);
        CHECK(fstat(fd, &st) == 0 && st.st_nlink == 0);
        CHECK(lseek(fd, 0, SEEK_SET) == 0);
        CHECK(read(fd, buf, sizeof buf) == 4 && !memcmp(buf, "live", 4));
        CHECK(close(fd) == 0);
        CHECK(stat("/tmp/named", &st) == -1 && errno == ENOENT);
    }
    create(argv[2], "private"); /* Same absolute spelling creates only a private object. */
    contents(argv[2], "private");
    const char *leaf = strrchr(argv[2], '/');
    snprintf(physical, sizeof physical, "%s/.capstone-tmp/%s", root, leaf + 1);
    CHECK(stat(argv[2], &st) == 0 && stat(physical, &other) == 0 && st.st_ino == other.st_ino);
    CHECK(unlink(argv[2]) == 0);
    CHECK(chdir("subdir") == 0);
    pid_t child = -1;
    char *args[] = {"../probe.dom", "child", argv[2], NULL};
    int error = posix_spawn(&child, "../probe.dom", NULL, NULL, args, environ), status = 0;
    CHECK(error == 0 && child > 0);
    if (!error) CHECK(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 23);
    contents("/tmp/inherited", "child");
    CHECK(unlink("/tmp/inherited") == 0);
    CHECK(chdir("..") == 0);
    char reply = 0;
    CHECK(write(1, "R", 1) == 1); /* Supervisor renames the pinned directory and replaces its old name. */
    CHECK(read(0, &reply, 1) == 1 && reply == 'R');
    create("/tmp/after-rename", "pinned");
    snprintf(physical, sizeof physical, "%s/scratch-renamed/after-rename", root);
    contents(physical, "pinned");
    CHECK(open(".capstone-tmp/marker", O_RDONLY) == -1);
    CHECK(chdir("/tmp") == 0);
    CHECK(getcwd(buf, sizeof buf) != NULL);
    snprintf(physical, sizeof physical, "%s/scratch-renamed", root);
    CHECK(!strcmp(buf, physical));
    int directory = open(".", O_RDONLY | O_DIRECTORY);
    int retained = open("after-rename", O_RDONLY);
    CHECK(directory >= 0 && retained >= 0);
    reply = 0;
    CHECK(write(1, "O", 1) == 1); /* Pin moves outside the original root. */
    CHECK(read(0, &reply, 1) == 1 && reply == 'O');
    CHECK(getcwd(buf, sizeof buf) == NULL && errno == ENOENT);
    CHECK(open("/tmp/after-rename", O_RDONLY) == -1);
    CHECK(open("after-rename", O_RDONLY) == -1);
    if (directory >= 0) {
        CHECK(openat(directory, "after-rename", O_RDONLY) == -1);
        CHECK(close(directory) == 0);
    }
    if (retained >= 0) {
        CHECK(read(retained, buf, sizeof buf) == 6 && !memcmp(buf, "pinned", 6));
        CHECK(close(retained) == 0); /* Already-admitted file authority is not revoked. */
    }
    CHECK(chdir(root) == 0);
    fprintf(stderr, "NATIVE_LOCAL_TMP mode=%s checks=%d failures=%d\n", argv[1], checks, failures);
    return failures != 0;
}
