/* fstat on the standard descriptors of a musl domain. stdout and stderr are
 * served by the hostcall and must look like the character devices they act as
 * (not terminals); stdin has no service and must say EBADF. Every check prints
 * one line; main returns the number that failed. */
#include <errno.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

static int failures;

static void check(const char *name, int ok, const char *detail)
{
	printf("STDIO-FSTAT %s %s %s\n", ok ? "PASS" : "FAIL", name, detail);
	if (!ok)
		failures++;
}

int main(void)
{
	char detail[96];
	for (int fd = 1; fd <= 2; fd++) {
		struct stat st;
		errno = 0;
		int rc = fstat(fd, &st);
		snprintf(detail, sizeof detail, "rc=%d errno=%d mode=%o", rc, errno,
		         rc == 0 ? (unsigned)st.st_mode : 0u);
		check(fd == 1 ? "fstat-stdout" : "fstat-stderr", rc == 0 && S_ISCHR(st.st_mode), detail);
	}
	struct stat st0;
	errno = 0;
	int rc0 = fstat(0, &st0);
	snprintf(detail, sizeof detail, "rc=%d errno=%d (want -1, EBADF=%d)", rc0, errno, EBADF);
	check("fstat-stdin", rc0 == -1 && errno == EBADF, detail);
	errno = 0;
	int tty = isatty(1);
	snprintf(detail, sizeof detail, "isatty=%d errno=%d", tty, errno);
	check("stdout-not-a-tty", tty == 0, detail);
	check("write-stdout", write(1, "STDIO-FSTAT write\n", 18) == 18, "");
	printf("STDIO-FSTAT-DONE failures=%d\n", failures);
	return failures;
}
