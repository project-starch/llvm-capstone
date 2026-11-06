#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/uio.h>
#include <unistd.h>
static int checks, failures;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; printf("FAIL line=%d errno=%d\n", __LINE__, errno); } } while (0)
int main(void)
{
    char a[4] = {0}, b[4] = {0}, cwd[4096];
    struct iovec out[] = {{"abc", 3}, {"def", 3}};
    struct iovec in[] = {{a, 3}, {b, 3}};
    struct stat st;
    struct sysinfo info;
    struct rlimit lim;
    sigset_t set, old, now;
    int fd = open("coverage.bin", O_CREAT|O_TRUNC|O_RDWR, 0600);
    CHECK(fd >= 0);
    CHECK(writev(fd, out, 2) == 6);
    CHECK(lseek(fd, 0, SEEK_SET) == 0);
    CHECK(readv(fd, in, 2) == 6);
    CHECK(!strcmp(a,"abc") && !strcmp(b,"def"));
    {
        char x[5] = "SSSS", y[5] = "SSSS";
        struct iovec short_read[] = {{x, 4}, {y, 4}};
        CHECK(lseek(fd, 0, SEEK_SET) == 0);
        CHECK(readv(fd, short_read, 2) == 6);
        CHECK(!memcmp(x, "abcd", 4) && !memcmp(y, "efSS", 4));
    }
    CHECK(fstat(fd, &st) == 0 && st.st_size == 6 && S_ISREG(st.st_mode));
    CHECK(stat("coverage.bin", &st) == 0 && st.st_size == 6);
    CHECK(close(fd) == 0);
    CHECK(writev(fd, out, 2) == -1 && errno == EBADF);
    CHECK(stat("does-not-exist", &st) == -1 && errno == ENOENT);
    CHECK(open("../outside", O_RDONLY) == -1 && errno == EACCES);
    CHECK(getcwd(cwd, sizeof(cwd)) != NULL && cwd[0] == '/');
    CHECK(sysinfo(&info) == 0 && info.totalram && info.mem_unit);
    CHECK(getrlimit(RLIMIT_NOFILE, &lim) == 0 && lim.rlim_cur > 0);
    sigemptyset(&set); sigaddset(&set, SIGUSR1);
    CHECK(sigprocmask(SIG_BLOCK, &set, &old) == 0);
    CHECK(sigprocmask(SIG_BLOCK, NULL, &now) == 0 && sigismember(&now, SIGUSR1));
    CHECK(sigprocmask(SIG_SETMASK, &old, NULL) == 0);
    printf("gcc coverage checks=%d failures=%d\n", checks, failures);
    return failures != 0;
}
