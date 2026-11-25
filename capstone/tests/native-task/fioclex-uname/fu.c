/* fu.c: directed test of three calls the native route now serves (ioctl FIOCLEX/FIONCLEX, uname, getrandom) and of
 * the refusals beside them. Each check prints one "FU <name> got=<v> want=<v> <PASS|FAIL>" line; the last line is
 * "FU_DONE <PASS|FAIL> <failures>". Run in a directory under dom_grant_root. */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/random.h>
#include <sys/utsname.h>
#include <termios.h>
#include <unistd.h>

static int fails;
static void check(const char *name, long got, long want)
{
    int ok = got == want;
    fails += !ok;
    printf("FU %s got=%ld want=%ld %s\n", name, got, want, ok ? "PASS" : "FAIL");
}

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    int fd = open("fu.tmp", O_CREAT | O_RDWR | O_TRUNC, 0600);
    check("open", fd >= 0, 1);
    check("getfd-initial", fcntl(fd, F_GETFD), 0);
    check("fioclex", ioctl(fd, FIOCLEX), 0);
    check("getfd-after-fioclex", fcntl(fd, F_GETFD), FD_CLOEXEC);
    check("fionclex", ioctl(fd, FIONCLEX), 0);
    check("getfd-after-fionclex", fcntl(fd, F_GETFD), 0);
    errno = 0;
    check("fioclex-badfd", ioctl(1000, FIOCLEX), -1);
    check("fioclex-badfd-errno", errno, EBADF);
    struct termios t;
    errno = 0;
    check("tcgets-refused", ioctl(fd, TCGETS, &t), -1);           /* a pointer request stays refused */
    check("tcgets-refused-errno", errno, ENOSYS);
    struct utsname u;
    memset(&u, 0x5b, sizeof u);
    int ur = uname(&u);
    check("uname", ur, 0);
    /* bounded compares: on a refusal the struct keeps its 0x5b fill and has no NUL */
    check("uname-sysname-linux", strncmp(u.sysname, "Linux", sizeof u.sysname) == 0, 1);
    check("uname-machine-riscv64", strncmp(u.machine, "riscv64", sizeof u.machine) == 0, 1);
    if (ur == 0)
        printf("FU uname sysname=%.65s release=%.65s machine=%.65s\n", u.sysname, u.release, u.machine);
    unsigned char b[64] = {0}, z[64] = {0};
    check("getrandom-64", getrandom(b, sizeof b, 0), 64);
    check("getrandom-not-all-zero", memcmp(b, z, sizeof b) != 0, 1);
    check("getrandom-nonblock", getrandom(b, 16, GRND_NONBLOCK), 16);
    errno = 0;
    check("getrandom-badflag", getrandom(b, 16, 0x80), -1);
    check("getrandom-badflag-errno", errno, EINVAL);
    close(fd);
    unlink("fu.tmp");
    printf("FU_DONE %s %d\n", fails ? "FAIL" : "PASS", fails);
    return fails != 0;
}
