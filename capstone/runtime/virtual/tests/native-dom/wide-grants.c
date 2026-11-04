/* SPDX-License-Identifier: GPL-2.0-only */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int failures;
static void check(const char *name, int ok)
{
    printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    failures += !ok;
}
int main(void)
{
    char buf[8] = {0};
    struct stat st;
    int fd = open("/tmp/n2-wide-created", O_CREAT | O_TRUNC | O_RDWR, 0600);
    check("tmp grant create", fd >= 0);
    if (fd >= 0) {
        check("tmp grant write", write(fd, "granted", 7) == 7);
        check("tmp grant pread", pread(fd, buf, 7, 0) == 7 && !memcmp(buf, "granted", 7));
        close(fd);
        check("tmp grant unlink", unlink("/tmp/n2-wide-created") == 0);
    }
    fd = open("/usr/n2-wide-input", O_RDONLY);
    check("usr grant read", fd >= 0 && read(fd, buf, 7) == 7 && !memcmp(buf, "granted", 7));
    if (fd >= 0) close(fd);
    check("usr grant stat", !stat("/usr/n2-wide-input", &st) && st.st_size == 7);
    const char *devices[] = {"/dev/null", "/dev/zero", "/dev/urandom"};
    for (unsigned i = 0; i < 3; ++i) {
        fd = open(devices[i], i ? O_RDONLY : O_WRONLY | O_CREAT | O_TRUNC, 0600);
        check(devices[i], fd >= 0 && (i ? read(fd, buf, 1) : write(fd, "x", 1)) == 1);
        if (i == 1 && fd >= 0) check("zero device value", buf[0] == 0);
        if (fd >= 0) close(fd);
    }
    const char *denied[] = {"/proc/self/mem", "/tmp/n2-proc/self/mem",
        "/tmp/n2-null-alias", "/tmp/n2-memory", "/tmp/n2-null-link",
        "/usr/../proc/self/mem", "/usr-not-granted/file", "/dev/null/../mem",
        "/dev/mem", "/dev/urandom-extra", "/usr/n2-wide-input/"};
    for (unsigned i = 0; i < sizeof(denied) / sizeof(*denied); ++i) {
        fd = open(denied[i], O_RDWR);
        check(denied[i], fd == -1 && (errno == EACCES || errno == ELOOP || errno == ENOTDIR));
        if (fd >= 0) close(fd);
    }
    printf("wide grant failures=%d\n", failures);
    return failures != 0;
}
