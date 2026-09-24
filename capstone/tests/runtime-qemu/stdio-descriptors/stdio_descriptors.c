/* The standard descriptors of a musl domain, through every syscall that takes a
 * descriptor. stdout and stderr are served by the hostcall and must look like the
 * character devices they act as (not terminals, not seekable); stdin has no
 * service and must say EBADF; a descriptor nobody opened must say EBADF to
 * ioctl too; and a closed stderr must then be closed to everything. Every check
 * prints one line; main returns the number that failed. */
#include <errno.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

static int failures;

static void check(const char *name, int ok, const char *detail)
{
	printf("STDIO-FD %s %s %s\n", ok ? "PASS" : "FAIL", name, detail);
	if (!ok)
		failures++;
}

int main(void)
{
	char detail[96];
	struct stat st;
	int rc;

	for (int fd = 1; fd <= 2; fd++) {
		errno = 0;
		rc = fstat(fd, &st);
		snprintf(detail, sizeof detail, "rc=%d errno=%d mode=%o", rc, errno,
		         rc == 0 ? (unsigned)st.st_mode : 0u);
		check(fd == 1 ? "fstat-stdout" : "fstat-stderr", rc == 0 && S_ISCHR(st.st_mode), detail);
	}
	errno = 0;
	rc = fstat(0, &st);
	snprintf(detail, sizeof detail, "rc=%d errno=%d (want -1, EBADF=%d)", rc, errno, EBADF);
	check("fstat-stdin", rc == -1 && errno == EBADF, detail);

	errno = 0;
	int tty = isatty(1);
	snprintf(detail, sizeof detail, "isatty=%d errno=%d (want 0, ENOTTY=%d)", tty, errno, ENOTTY);
	check("stdout-not-a-tty", tty == 0 && errno == ENOTTY, detail);
	errno = 0;
	tty = isatty(99);
	snprintf(detail, sizeof detail, "isatty=%d errno=%d (want 0, EBADF=%d)", tty, errno, EBADF);
	check("unopened-isatty-ebadf", tty == 0 && errno == EBADF, detail);

	errno = 0;
	off_t off = lseek(1, 0, SEEK_CUR);
	snprintf(detail, sizeof detail, "rc=%lld errno=%d (want -1, ESPIPE=%d)", (long long)off, errno, ESPIPE);
	check("lseek-stdout-espipe", off == -1 && errno == ESPIPE, detail);

	check("write-stdout", write(1, "STDIO-FD write\n", 15) == 15, "");

	/* Last: close stderr, then everything must agree it is gone. */
	errno = 0;
	rc = close(2);
	snprintf(detail, sizeof detail, "rc=%d errno=%d", rc, errno);
	check("close-stderr", rc == 0, detail);
	errno = 0;
	rc = fstat(2, &st);
	int e1 = errno;
	errno = 0;
	ssize_t w = write(2, "x", 1);
	int e2 = errno;
	errno = 0;
	int again = close(2);
	int e3 = errno;
	snprintf(detail, sizeof detail, "fstat=%d/%d write=%zd/%d close=%d/%d (want -1/EBADF=%d each)",
	         rc, e1, w, e2, again, e3, EBADF);
	check("closed-stderr-ebadf", rc == -1 && e1 == EBADF && w == -1 && e2 == EBADF &&
	      again == -1 && e3 == EBADF, detail);

	printf("STDIO-FD-DONE failures=%d\n", failures);
	return failures;
}
