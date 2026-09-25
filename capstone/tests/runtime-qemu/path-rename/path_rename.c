/* rename() in a musl domain, through the hostcall's PATH_RENAME.
 *
 * Two files on the share: one renamed to a new name, then over an existing
 * one, with the old name checked gone and the new name checked for the moved
 * content each time; a missing source and a missing target directory, which
 * must answer ENOENT; and a renameat2 flag, which the domain refuses with
 * EINVAL because nothing serves it. Every check prints one line; the last line
 * counts the failures and main returns that count, which the host reports as
 * the exit status. */
#define _GNU_SOURCE /* syscall */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

static int failures;

static void check(const char *name, int ok, const char *detail)
{
	printf("PATH-RENAME %s %s %s\n", ok ? "PASS" : "FAIL", name, detail);
	if (!ok)
		failures++;
}

static int create(const char *path, const char *text)
{
	int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	if (fd < 0)
		return -1;
	long n = write(fd, text, strlen(text));
	close(fd);
	return n == (long)strlen(text) ? 0 : -1;
}

/* Reads up to 15 bytes of path into buf, NUL-terminated; "" if it cannot be opened. */
static int slurp(const char *path, char *buf)
{
	buf[0] = '\0';
	int fd = open(path, O_RDONLY);
	if (fd < 0)
		return -errno;
	long n = read(fd, buf, 15);
	close(fd);
	if (n < 0)
		return -1;
	buf[n] = '\0';
	return (int)n;
}

int main(void)
{
	static const char a[] = "/mnt/host/rename-a.dat";
	static const char b[] = "/mnt/host/rename-b.dat";
	static const char c[] = "/mnt/host/rename-c.dat";
	char detail[160], buf[16];
	int rc;

	check("create-a", create(a, "alpha") == 0, "");

	errno = 0;
	rc = rename(a, b);
	snprintf(detail, sizeof detail, "rc=%d errno=%d", rc, rc ? errno : 0);
	check("rename", rc == 0, detail);

	errno = 0;
	rc = open(a, O_RDONLY);
	snprintf(detail, sizeof detail, "fd=%d errno=%d (want ENOENT=%d)", rc, errno, ENOENT);
	check("source-gone", rc < 0 && errno == ENOENT, detail);
	if (rc >= 0)
		close(rc);

	rc = slurp(b, buf);
	snprintf(detail, sizeof detail, "n=%d data=%s", rc, buf);
	check("target-has-data", rc == 5 && strcmp(buf, "alpha") == 0, detail);

	/* Over an existing file: the destination is replaced, the source is gone. */
	check("create-c", create(c, "gamma") == 0, "");
	errno = 0;
	rc = rename(b, c);
	snprintf(detail, sizeof detail, "rc=%d errno=%d", rc, rc ? errno : 0);
	check("rename-over-existing", rc == 0, detail);
	rc = slurp(c, buf);
	snprintf(detail, sizeof detail, "n=%d data=%s (want alpha)", rc, buf);
	check("replaced-content", rc == 5 && strcmp(buf, "alpha") == 0, detail);
	errno = 0;
	rc = open(b, O_RDONLY);
	snprintf(detail, sizeof detail, "fd=%d errno=%d (want ENOENT=%d)", rc, errno, ENOENT);
	check("old-name-gone", rc < 0 && errno == ENOENT, detail);
	if (rc >= 0)
		close(rc);

	errno = 0;
	rc = rename("/mnt/host/rename-missing.dat", b);
	snprintf(detail, sizeof detail, "rc=%d errno=%d (want ENOENT=%d)", rc, errno, ENOENT);
	check("missing-source", rc < 0 && errno == ENOENT, detail);

	errno = 0;
	rc = rename(c, "/mnt/host/no-such-directory/x.dat");
	snprintf(detail, sizeof detail, "rc=%d errno=%d (want ENOENT=%d)", rc, errno, ENOENT);
	check("missing-target-dir", rc < 0 && errno == ENOENT, detail);

	/* renameat2 with RENAME_NOREPLACE (1): nothing serves the flag, so it is
	   refused rather than dropped; the file must still be where it was. */
	errno = 0;
	rc = (int)syscall(SYS_renameat2, AT_FDCWD, c, AT_FDCWD, b, 1);
	snprintf(detail, sizeof detail, "rc=%d errno=%d (want EINVAL=%d)", rc, errno, EINVAL);
	check("flag-refused", rc < 0 && errno == EINVAL, detail);
	rc = slurp(c, buf);
	snprintf(detail, sizeof detail, "n=%d data=%s", rc, buf);
	check("flag-left-file-alone", rc == 5 && strcmp(buf, "alpha") == 0, detail);

	unlink(c);
	printf("PATH-RENAME-DONE failures=%d\n", failures);
	return failures;
}
