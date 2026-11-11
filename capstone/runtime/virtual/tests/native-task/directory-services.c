/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <dirent.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

static int checks, failures;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; fprintf(stderr, "DIRECTORY_FAIL line=%d errno=%d test=%s\n", __LINE__, errno, #x); } } while (0)

struct wire_dirent {
    uint64_t ino;
    int64_t offset;
    uint16_t length;
    uint8_t type;
    char name[];
};
static void directory_entries(int fd)
{
    _Alignas(16) unsigned char bytes[32];
    unsigned seen = 0;
    int records = 0;
    _Static_assert(offsetof(struct wire_dirent, name) == 19, "Linux dirent64 layout");
    CHECK(lseek(fd, 0, SEEK_SET) == 0);
    errno = 0;
    CHECK(syscall(SYS_getdents64, fd, bytes, 1) == -1 && errno == EINVAL);
    CHECK(lseek(fd, 0, SEEK_CUR) == 0);
    for (int rounds = 0; rounds < 10; ++rounds) {
        memset(bytes, 0xff, sizeof bytes);
        long n = syscall(SYS_getdents64, fd, bytes, sizeof bytes);
        CHECK(n >= 0 && n <= (long)sizeof bytes);
        if (n <= 0) break;
        struct wire_dirent *entry = (struct wire_dirent *)bytes;
        CHECK(entry->length == n && !(n & 7) && n >= 24);
        if (entry->length != n || n < 24) break;
        CHECK(memchr(entry->name, 0, n - offsetof(struct wire_dirent, name)) != NULL);
        if (!memchr(entry->name, 0, n - offsetof(struct wire_dirent, name))) break;
        const char *names[] = {".", "..", "file", "link", "escape"};
        unsigned bit = 0;
        for (unsigned i = 0; i < 5; ++i) if (!strcmp(entry->name, names[i])) bit = 1U << i;
        CHECK(bit && !(seen & bit));
        seen |= bit;
        CHECK(entry->offset == lseek(fd, 0, SEEK_CUR));
        ++records;
    }
    CHECK(records == 5 && seen == 31);
    CHECK(syscall(SYS_getdents64, fd, bytes, sizeof bytes) == 0);
    DIR *dir = opendir("dir");
    CHECK(dir != NULL);
    if (dir) {
        int count = 0;
        while (readdir(dir)) ++count;
        CHECK(count == 5);
        CHECK(closedir(dir) == 0);
    }
}
int main(void)
{
    struct stat st;
    char buf[128], path[PATH_MAX];
    /* The scalar supervisor supplies umask 0027 before native exec. */
    CHECK(mkdir("dir/", 0777) == 0);
    CHECK(stat("dir", &st) == 0 && (st.st_mode & 0777) == 0750);
    errno = 0; CHECK(mkdir("dir/", 0777) == -1 && errno == EEXIST);
    CHECK(chmod("dir", 0711) == 0);
    CHECK(stat("dir", &st) == 0 && (st.st_mode & 0777) == 0711);
    int dir = open("dir", O_RDONLY | O_DIRECTORY);
    CHECK(dir >= 0);
    CHECK(mkdirat(dir, "nested//", 0700) == 0);
    errno = 0; CHECK(rmdir("dir") == -1 && errno == ENOTEMPTY);
    errno = 0; CHECK(rmdir("dir/.") == -1 && errno == EINVAL);
    errno = 0; CHECK(rmdir("dir/..") == -1 && errno == ENOTEMPTY);
    CHECK(unlinkat(dir, "nested/", AT_REMOVEDIR) == 0);
    CHECK(fchmodat(dir, ".", 0755, 0) == 0);
    CHECK(fstat(dir, &st) == 0 && (st.st_mode & 0777) == 0755);
    int fd = openat(dir, "file", O_RDWR | O_CREAT | O_EXCL, 0600);
    CHECK(fd >= 0);
    CHECK(write(fd, "abcdefgh", 8) == 8);
    CHECK(truncate("dir/file", 3) == 0);
    CHECK(fstat(fd, &st) == 0 && st.st_size == 3);
    errno = 0; CHECK(truncate("dir/file", -1) == -1 && errno == EINVAL);
    CHECK(fstat(fd, &st) == 0 && st.st_size == 3);
    CHECK(truncate("dir/file", 6) == 0);
    CHECK(lseek(fd, 0, SEEK_SET) == 0);
    memset(buf, 1, sizeof buf);
    CHECK(read(fd, buf, sizeof buf) == 6 && !memcmp(buf, "abc\0\0\0", 6));
    CHECK(close(fd) == 0);
    CHECK(symlinkat("file", dir, "link") == 0);
    CHECK(fstatat(dir, "link", &st, AT_SYMLINK_NOFOLLOW) == 0 && S_ISLNK(st.st_mode));
    memset(buf, 'X', sizeof buf);
    CHECK(readlinkat(dir, "link", buf, 2) == 2 && !memcmp(buf, "fiX", 3));
    CHECK(readlink("dir/link", buf, sizeof buf) == 4 && !memcmp(buf, "file", 4));
    CHECK(open("dir/link", O_RDONLY) == -1);
    CHECK(chmod("dir/link", 0777) == -1);
    CHECK(truncate("dir/link", 0) == -1);
    CHECK(stat("dir/file", &st) == 0 && st.st_size == 6 && (st.st_mode & 0777) == 0600);
    CHECK(symlink("../../outside/marker", "dir/escape") == 0);
    CHECK(lstat("dir/escape", &st) == 0 && S_ISLNK(st.st_mode));
    CHECK(readlink("dir/escape", buf, sizeof buf) == 20 && !memcmp(buf, "../../outside/marker", 20));
    CHECK(open("dir/escape", O_RDONLY) == -1);
    CHECK(mkdir("dir/escape/child", 0700) == -1);
    CHECK(mkdir("../outside/nope", 0700) == -1);
    CHECK(truncate("../outside/marker", 0) == -1);
    directory_entries(dir);
    CHECK(linkat(dir, "file", dir, "hard", 0) == 0);
    struct stat linked;
    CHECK(stat("dir/file", &st) == 0 && lstat("dir/hard", &linked) == 0 &&
          st.st_ino == linked.st_ino && st.st_dev == linked.st_dev &&
          st.st_nlink == 2 && linked.st_nlink == 2);
    CHECK(truncate("dir/hard", 3) == 0);
    CHECK(stat("dir/file", &st) == 0 && st.st_size == 3);
    errno = 0; CHECK(link("dir/file", "dir/hard") == -1 && errno == EEXIST);
    CHECK(link("dir", "dir/hard-dir") == -1);
    CHECK(link("dir/link", "dir/hard-symlink") == -1);
    CHECK(linkat(dir, "file", dir, "hard-follow", AT_SYMLINK_FOLLOW) == -1);
    CHECK(link("../outside/marker", "dir/hard-outside") == -1);
    CHECK(link("dir/file", "../outside/hard-escape") == -1);
    CHECK(unlinkat(dir, "hard", 0) == 0);
    CHECK(stat("dir/file", &st) == 0 && st.st_nlink == 1);
    CHECK(getcwd(path, sizeof path) != NULL);
    errno = 0; CHECK(rmdir(path) == -1 && errno == EBUSY);
    CHECK(readlink("dir/file", buf, sizeof buf) == -1 && errno == EINVAL);
    /* musl probes through a private one-byte buffer for a zero-size libc
     * readlink. The raw Linux syscall still rejects zero capacity. */
    memset(buf, 'Z', sizeof buf);
    CHECK(readlink("dir/link", buf, 0) == 0 && buf[0] == 'Z' && buf[sizeof buf - 1] == 'Z');
    errno = 0;
    CHECK(syscall(SYS_readlinkat, dir, "link", buf, 0) == -1 && errno == EINVAL);
    CHECK(unlinkat(dir, "link", 0) == 0);
    CHECK(unlink("dir/escape") == 0);
    CHECK(unlinkat(dir, "file", 0) == 0);
    CHECK(close(dir) == 0);
    CHECK(rmdir("dir///") == 0);
    fprintf(stderr, "NATIVE_DIRECTORY checks=%d failures=%d\n", checks, failures);
    return failures != 0;
}
