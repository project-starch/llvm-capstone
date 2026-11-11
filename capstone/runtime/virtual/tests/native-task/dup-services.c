/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

static int checks, failures;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; fprintf(stderr, "DUP_FAIL line=%d errno=%d test=%s\n", __LINE__, errno, #x); } } while (0)
static void transfer(int writer, int reader, char want)
{
    char got = 0;
    long rc = write(writer, &want, 1);
    CHECK(rc == 1);
    /* A broken destination grant must fail, not leave the test blocked. */
    if (rc == 1) CHECK(read(reader, &got, 1) == 1 && got == want);
}
int main(void)
{
    int a[2], b[2];
    CHECK(pipe2(a, O_CLOEXEC) == 0);
    CHECK(pipe2(b, O_CLOEXEC) == 0);
    if (failures) return 1;
    CHECK(a[0] < 20 && a[1] < 20 && b[0] < 20 && b[1] < 20);
    CHECK(dup3(a[1], 20, O_CLOEXEC) == 20);
    CHECK(fcntl(20, F_GETFD) == FD_CLOEXEC);
    transfer(20, a[0], 'A');
    errno = 0;
    CHECK(dup3(a[1], a[1], 0) == -1 && errno == EINVAL);
    CHECK(fcntl(a[1], F_GETFD) == FD_CLOEXEC);
    CHECK(dup2(a[1], a[1]) == a[1]);
    CHECK(fcntl(a[1], F_GETFD) == FD_CLOEXEC);
    errno = 0;
    CHECK(dup3(a[1], 20, O_NONBLOCK) == -1 && errno == EINVAL);
    CHECK(fcntl(20, F_GETFD) == FD_CLOEXEC);
    transfer(20, a[0], 'B');
    CHECK(dup3(b[1], 20, 0) == 20);
    CHECK(fcntl(20, F_GETFD) == 0);
    transfer(20, b[0], 'C');
    CHECK(dup2(a[1], 21) == 21);
    CHECK(fcntl(21, F_GETFD) == 0);
    transfer(21, a[0], 'D');
    CHECK(dup2(a[1], 20) == 20);
    CHECK(fcntl(20, F_GETFD) == 0);
    transfer(20, a[0], 'E');
    errno = 0;
    CHECK(dup3(199, 20, 0) == -1 && errno == EBADF);
    transfer(20, a[0], 'F');
    CHECK(close(20) == 0);
    CHECK(close(21) == 0);
    CHECK(close(a[0]) == 0);
    CHECK(close(a[1]) == 0);
    CHECK(close(b[0]) == 0);
    CHECK(close(b[1]) == 0);
    fprintf(stderr, "NATIVE_DUP checks=%d failures=%d\n", checks, failures);
    return failures != 0;
}
