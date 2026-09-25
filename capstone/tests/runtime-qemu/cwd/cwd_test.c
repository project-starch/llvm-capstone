/* chdir() and getcwd() in a musl domain: a working directory kept on the
 * domain side, joined onto every relative path before it crosses to the
 * helper.
 *
 * run.sh stages /mnt/host/cwdtest with inside.txt ("inside") and
 * sub/deep.txt ("deep"). Before the first chdir there is no cwd: getcwd says
 * ENOENT and a relative open reaches the helper's own directory, where the
 * file is not. After chdir the relative open, access, rename and unlink all
 * land in the directory, chdir to a relative name and ".." resolve through
 * it, a missing or non-directory target is refused and leaves the cwd alone,
 * and absolute paths still work. Every check prints one line; the last line
 * counts the failures and main returns that count. */
#define _XOPEN_SOURCE 700
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int failures;

static void check(const char *name, int ok, const char *detail)
{
	printf("CWD %s %s %s\n", ok ? "PASS" : "FAIL", name, detail);
	if (!ok)
		failures++;
}

/* Reads up to 15 bytes of path into buf, NUL-terminated; -errno if it cannot be opened. */
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

static int create(const char *path, const char *text)
{
	int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	if (fd < 0)
		return -errno;
	long n = write(fd, text, strlen(text));
	close(fd);
	return n == (long)strlen(text) ? 0 : -1;
}

int main(void)
{
	char detail[200], buf[16], cwd[256];
	int rc;

	errno = 0;
	char *p = getcwd(cwd, sizeof cwd);
	snprintf(detail, sizeof detail, "p=%s errno=%d (want ENOENT=%d)", p ? p : "(null)", errno, ENOENT);
	check("no-cwd-yet", !p && errno == ENOENT, detail);

	rc = slurp("inside.txt", buf);
	snprintf(detail, sizeof detail, "rc=%d (want -ENOENT=%d)", rc, -ENOENT);
	check("relative-before-chdir", rc == -ENOENT, detail);

	errno = 0;
	rc = chdir("/mnt/host/cwdtest");
	snprintf(detail, sizeof detail, "rc=%d errno=%d", rc, rc ? errno : 0);
	check("chdir", rc == 0, detail);

	p = getcwd(cwd, sizeof cwd);
	snprintf(detail, sizeof detail, "cwd=%s", p ? p : "(null)");
	check("getcwd", p && strcmp(p, "/mnt/host/cwdtest") == 0, detail);

	rc = slurp("inside.txt", buf);
	snprintf(detail, sizeof detail, "rc=%d data=%s", rc, buf);
	check("relative-open", rc == 6 && strcmp(buf, "inside") == 0, detail);

	errno = 0;
	rc = access("inside.txt", F_OK);
	snprintf(detail, sizeof detail, "rc=%d errno=%d", rc, rc ? errno : 0);
	check("relative-access", rc == 0, detail);

	errno = 0;
	rc = chdir("sub");
	p = getcwd(cwd, sizeof cwd);
	snprintf(detail, sizeof detail, "rc=%d errno=%d cwd=%s", rc, rc ? errno : 0, p ? p : "(null)");
	check("relative-chdir", rc == 0 && p && strcmp(p, "/mnt/host/cwdtest/sub") == 0, detail);

	rc = slurp("deep.txt", buf);
	snprintf(detail, sizeof detail, "rc=%d data=%s", rc, buf);
	check("open-in-sub", rc == 4 && strcmp(buf, "deep") == 0, detail);

	rc = slurp("../inside.txt", buf);
	snprintf(detail, sizeof detail, "rc=%d data=%s", rc, buf);
	check("dotdot", rc == 6 && strcmp(buf, "inside") == 0, detail);

	errno = 0;
	rc = chdir("/mnt/host/no-such-directory");
	p = getcwd(cwd, sizeof cwd);
	snprintf(detail, sizeof detail, "rc=%d errno=%d (want ENOENT=%d) cwd=%s", rc, errno, ENOENT, p ? p : "(null)");
	check("missing-dir", rc < 0 && errno == ENOENT && p && strcmp(p, "/mnt/host/cwdtest/sub") == 0, detail);

	errno = 0;
	rc = chdir("/mnt/host/cwdtest/inside.txt");
	snprintf(detail, sizeof detail, "rc=%d errno=%d (want ENOTDIR=%d)", rc, errno, ENOTDIR);
	check("not-a-directory", rc < 0 && errno == ENOTDIR, detail);

	/* rename and unlink take relative paths through the same join. */
	rc = create("r1.txt", "one");
	snprintf(detail, sizeof detail, "rc=%d", rc);
	check("relative-create", rc == 0, detail);
	errno = 0;
	rc = rename("r1.txt", "r2.txt");
	int got = slurp("r2.txt", buf);
	snprintf(detail, sizeof detail, "rc=%d errno=%d read=%d data=%s", rc, rc ? errno : 0, got, buf);
	check("relative-rename", rc == 0 && got == 3 && strcmp(buf, "one") == 0, detail);
	errno = 0;
	rc = unlink("r2.txt");
	int gone = access("r2.txt", F_OK) < 0 && errno == ENOENT;
	snprintf(detail, sizeof detail, "rc=%d gone=%d", rc, gone);
	check("relative-unlink", rc == 0 && gone, detail);

	rc = slurp("/mnt/host/cwdtest/inside.txt", buf);
	snprintf(detail, sizeof detail, "rc=%d data=%s", rc, buf);
	check("absolute-after-chdir", rc == 6 && strcmp(buf, "inside") == 0, detail);

	errno = 0;
	p = getcwd(cwd, 8);
	snprintf(detail, sizeof detail, "p=%s errno=%d (want ERANGE=%d)", p ? p : "(null)", errno, ERANGE);
	check("getcwd-erange", !p && errno == ERANGE, detail);

	printf("CWD-DONE failures=%d\n", failures);
	return failures;
}
