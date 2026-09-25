/* readlink() in a musl domain, through the hostcall's PATH_READLINK, and
 * realpath() on top of it, which is how a program finds its own path.
 *
 * run.sh stages /mnt/host/rltest with inside.txt and link.txt -> inside.txt.
 * readlink of the link gives the target, of a file EINVAL, of a missing name
 * ENOENT, into a short buffer the first bytes only (no terminator, as
 * readlink(2)); realpath of the link resolves to the file, which is musl's
 * realpath walking the path with readlink on every component; the same
 * relative under chdir. Every check prints one line; the last line counts the
 * failures and main returns that count. */
#define _XOPEN_SOURCE 700
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures;

static void check(const char *name, int ok, const char *detail)
{
	printf("PATH-READLINK %s %s %s\n", ok ? "PASS" : "FAIL", name, detail);
	if (!ok)
		failures++;
}

int main(void)
{
	char detail[200], buf[64];
	long n;

	memset(buf, 'x', sizeof buf);
	errno = 0;
	n = readlink("/mnt/host/rltest/link.txt", buf, sizeof buf - 1);
	if (n >= 0)
		buf[n] = '\0';
	snprintf(detail, sizeof detail, "n=%ld target=%s errno=%d", n, n >= 0 ? buf : "", n < 0 ? errno : 0);
	check("readlink", n == 10 && strcmp(buf, "inside.txt") == 0, detail);

	errno = 0;
	n = readlink("/mnt/host/rltest/inside.txt", buf, sizeof buf);
	snprintf(detail, sizeof detail, "n=%ld errno=%d (want EINVAL=%d)", n, errno, EINVAL);
	check("not-a-symlink", n < 0 && errno == EINVAL, detail);

	errno = 0;
	n = readlink("/mnt/host/rltest/missing.txt", buf, sizeof buf);
	snprintf(detail, sizeof detail, "n=%ld errno=%d (want ENOENT=%d)", n, errno, ENOENT);
	check("missing", n < 0 && errno == ENOENT, detail);

	memset(buf, 'x', sizeof buf);
	n = readlink("/mnt/host/rltest/link.txt", buf, 3);
	snprintf(detail, sizeof detail, "n=%ld first=%.3s fourth=%c (want x: untouched)", n, buf, buf[3]);
	check("short-buffer", n == 3 && memcmp(buf, "ins", 3) == 0 && buf[3] == 'x', detail);

	char *rp = realpath("/mnt/host/rltest/link.txt", NULL);
	snprintf(detail, sizeof detail, "realpath=%s errno=%d", rp ? rp : "(null)", rp ? 0 : errno);
	check("realpath-through-link", rp && strcmp(rp, "/mnt/host/rltest/inside.txt") == 0, detail);
	free(rp);

	rp = realpath("/mnt/host/rltest/../rltest/inside.txt", NULL);
	snprintf(detail, sizeof detail, "realpath=%s errno=%d", rp ? rp : "(null)", rp ? 0 : errno);
	check("realpath-dotdot", rp && strcmp(rp, "/mnt/host/rltest/inside.txt") == 0, detail);
	free(rp);

	int rc = chdir("/mnt/host/rltest");
	memset(buf, 0, sizeof buf);
	n = readlink("link.txt", buf, sizeof buf - 1);
	snprintf(detail, sizeof detail, "chdir=%d n=%ld target=%s", rc, n, buf);
	check("relative-under-cwd", rc == 0 && n == 10 && strcmp(buf, "inside.txt") == 0, detail);

	printf("PATH-READLINK-DONE failures=%d\n", failures);
	return failures;
}
