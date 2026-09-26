/* dup, FD_CLOEXEC, lstat, symlink, chmod, flock and /dev/tty in a musl domain,
 * the calls mruby-io's tests need from the runtime.
 *
 * On one file on the share: O_CLOEXEC read back through F_GETFD and changed
 * with F_SETFD; dup, dup2 and F_DUPFD_CLOEXEC sharing the file position with
 * the original, and the description surviving the close of either; pipe2's
 * O_CLOEXEC; a symbolic link that lstat sees as a link and stat follows;
 * chmod read back through stat; flock through the helper, where a second open
 * of the same file conflicts with the lock and a dup shares it; /dev/tty
 * refused with ENXIO; and dup of stdout, which is left unserved and so answers
 * ENOSYS. Every check prints one line; the last line counts the failures and
 * main returns that count, which the host reports as the exit status. */
#define _GNU_SOURCE /* dup3, pipe2 */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

static int failures;

static void check(const char *name, int ok, const char *detail)
{
	printf("FD-OPS %s %s %s\n", ok ? "PASS" : "FAIL", name, detail);
	if (!ok)
		failures++;
}

/* One byte from fd, or '?' with errno in *err. */
static char byte(int fd, int *err)
{
	char c;
	errno = 0;
	if (read(fd, &c, 1) != 1) {
		*err = errno;
		return '?';
	}
	*err = 0;
	return c;
}

int main(void)
{
	static const char a[] = "/mnt/host/fdops-a.dat";
	static const char l[] = "/mnt/host/fdops-l.lnk";
	char detail[160];
	int rc, err;

	unlink(l);
	int w = open(a, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	rc = w >= 0 && write(w, "mruby", 5) == 5;
	if (w >= 0)
		close(w);
	check("create", rc, "");

	/* FD_CLOEXEC: kept, and given back. */
	int fd = open(a, O_RDONLY | O_CLOEXEC);
	rc = fcntl(fd, F_GETFD);
	snprintf(detail, sizeof detail, "fd=%d F_GETFD=%d (want %d)", fd, rc, FD_CLOEXEC);
	check("cloexec-open", fd >= 0 && rc == FD_CLOEXEC, detail);
	rc = fcntl(fd, F_SETFD, 0);
	int got = fcntl(fd, F_GETFD);
	snprintf(detail, sizeof detail, "F_SETFD=%d F_GETFD=%d (want 0)", rc, got);
	check("cloexec-cleared", rc == 0 && got == 0, detail);

	/* dup: a new descriptor, one position. */
	errno = 0;
	int d = dup(fd);
	snprintf(detail, sizeof detail, "fd=%d dup=%d errno=%d", fd, d, d < 0 ? errno : 0);
	check("dup", d >= 0 && d != fd, detail);
	char c1 = byte(d, &err), c2 = byte(fd, &err), c3 = byte(d, &err);
	snprintf(detail, sizeof detail, "read %c%c%c (want mru) errno=%d", c1, c2, c3, err);
	check("dup-shares-position", c1 == 'm' && c2 == 'r' && c3 == 'u', detail);
	rc = fcntl(d, F_GETFD);
	snprintf(detail, sizeof detail, "F_GETFD=%d (want 0)", rc);
	check("dup-not-cloexec", rc == 0, detail);

	/* flock: the helper's, per open file description. A second open of the
	   same file is another description and conflicts; the dup shares the lock. */
	errno = 0;
	rc = flock(fd, LOCK_EX | LOCK_NB);
	snprintf(detail, sizeof detail, "rc=%d errno=%d", rc, rc ? errno : 0);
	check("flock", rc == 0, detail);
	int other = open(a, O_RDONLY);
	errno = 0;
	rc = flock(other, LOCK_EX | LOCK_NB);
	snprintf(detail, sizeof detail, "rc=%d errno=%d (want EWOULDBLOCK=%d)", rc, errno, EWOULDBLOCK);
	check("flock-conflicts", other >= 0 && rc < 0 && errno == EWOULDBLOCK, detail);
	errno = 0;
	rc = flock(d, LOCK_EX | LOCK_NB);
	snprintf(detail, sizeof detail, "rc=%d errno=%d", rc, rc ? errno : 0);
	check("flock-dup-shares", rc == 0, detail);
	rc = flock(fd, LOCK_UN);
	int rc2 = flock(other, LOCK_EX | LOCK_NB);
	snprintf(detail, sizeof detail, "unlock=%d relock-other=%d", rc, rc2);
	check("flock-released", rc == 0 && rc2 == 0, detail);
	if (other >= 0)
		close(other);

	/* Closing the dup leaves the original's description open, and back. */
	rc = close(d);
	c1 = byte(fd, &err);
	snprintf(detail, sizeof detail, "close=%d then read %c (want b) errno=%d", rc, c1, err);
	check("close-dup-keeps-original", rc == 0 && c1 == 'b', detail);

	/* dup2 onto a chosen number, and F_DUPFD_CLOEXEC at or above one. */
	int want = fd + 5;
	errno = 0;
	rc = dup2(fd, want);
	snprintf(detail, sizeof detail, "dup2=%d (want %d) errno=%d", rc, want, rc < 0 ? errno : 0);
	check("dup2", rc == want, detail);
	int e = fcntl(fd, F_DUPFD_CLOEXEC, 20);
	got = e >= 0 ? fcntl(e, F_GETFD) : -1;
	snprintf(detail, sizeof detail, "fd=%d F_GETFD=%d", e, got);
	check("dupfd-cloexec", e >= 20 && got == FD_CLOEXEC, detail);
	close(fd);
	c1 = byte(want, &err);
	snprintf(detail, sizeof detail, "read %c (want y) errno=%d", c1, err);
	check("close-original-keeps-dup", c1 == 'y', detail);
	close(want);
	close(e);

	/* pipe2's O_CLOEXEC. */
	int p[2] = { -1, -1 };
	rc = pipe2(p, O_CLOEXEC);
	snprintf(detail, sizeof detail, "rc=%d r=%d w=%d", rc, rc ? -1 : fcntl(p[0], F_GETFD),
	         rc ? -1 : fcntl(p[1], F_GETFD));
	check("pipe-cloexec", rc == 0 && fcntl(p[0], F_GETFD) == FD_CLOEXEC &&
	                      fcntl(p[1], F_GETFD) == FD_CLOEXEC, detail);
	if (rc == 0) {
		close(p[0]);
		close(p[1]);
	}

	/* symlink, and lstat against stat. */
	errno = 0;
	rc = symlink("fdops-a.dat", l);
	snprintf(detail, sizeof detail, "rc=%d errno=%d", rc, rc ? errno : 0);
	check("symlink", rc == 0, detail);
	struct stat st;
	errno = 0;
	rc = lstat(l, &st);
	snprintf(detail, sizeof detail, "rc=%d errno=%d mode=%o", rc, rc ? errno : 0,
	         rc ? 0 : (unsigned)st.st_mode);
	check("lstat-sees-link", rc == 0 && S_ISLNK(st.st_mode), detail);
	rc = stat(l, &st);
	snprintf(detail, sizeof detail, "rc=%d mode=%o size=%lld", rc, rc ? 0 : (unsigned)st.st_mode,
	         rc ? -1LL : (long long)st.st_size);
	check("stat-follows-link", rc == 0 && S_ISREG(st.st_mode) && st.st_size == 5, detail);
	errno = 0;
	rc = lstat("/mnt/host/fdops-missing", &st);
	snprintf(detail, sizeof detail, "rc=%d errno=%d (want ENOENT=%d)", rc, errno, ENOENT);
	check("lstat-missing", rc < 0 && errno == ENOENT, detail);

	/* chmod, read back. */
	errno = 0;
	rc = chmod(a, 0600);
	got = stat(a, &st) == 0 ? (int)(st.st_mode & 07777) : -1;
	snprintf(detail, sizeof detail, "rc=%d errno=%d mode=%o (want 600)", rc, rc ? errno : 0, got);
	check("chmod", rc == 0 && got == 0600, detail);
	rc = chmod(a, 0644);
	got = stat(a, &st) == 0 ? (int)(st.st_mode & 07777) : -1;
	snprintf(detail, sizeof detail, "rc=%d mode=%o (want 644)", rc, got);
	check("chmod-back", rc == 0 && got == 0644, detail);

	/* No controlling terminal. */
	errno = 0;
	rc = open("/dev/tty", O_RDWR);
	snprintf(detail, sizeof detail, "fd=%d errno=%d (want ENXIO=%d)", rc, errno, ENXIO);
	check("dev-tty", rc < 0 && errno == ENXIO, detail);
	if (rc >= 0)
		close(rc);

	/* stdout has no description to share: unserved, said so. */
	errno = 0;
	rc = dup(1);
	snprintf(detail, sizeof detail, "rc=%d errno=%d (want ENOSYS=%d)", rc, errno, ENOSYS);
	check("dup-stdout-unserved", rc < 0 && errno == ENOSYS, detail);

	unlink(l);
	unlink(a);
	printf("FD-OPS-DONE failures=%d\n", failures);
	return failures;
}
