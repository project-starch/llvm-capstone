/* mkdir() and rmdir() in a musl domain, through the hostcall's PATH_MKDIR
 * and PATH_DELETE's directory flag.
 *
 * A directory made on the share and seen as one (opendir), made again
 * (EEXIST), made under a missing parent (ENOENT); a file inside it makes
 * rmdir ENOTEMPTY, rmdir of the file is ENOTDIR and unlink of the directory
 * EISDIR; emptied, the directory goes and access says ENOENT; a relative
 * mkdir and rmdir under chdir. Every check prints one line; the last line
 * counts the failures and main returns that count. */
#define _XOPEN_SOURCE 700
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

static int failures;

static void check(const char *name, int ok, const char *detail)
{
	printf("MKDIR-RMDIR %s %s %s\n", ok ? "PASS" : "FAIL", name, detail);
	if (!ok)
		failures++;
}

int main(void)
{
	static const char d[] = "/mnt/host/mkdirtest";
	static const char f[] = "/mnt/host/mkdirtest/inside.txt";
	char detail[160];
	int rc;

	errno = 0;
	rc = mkdir(d, 0755);
	snprintf(detail, sizeof detail, "rc=%d errno=%d", rc, rc ? errno : 0);
	check("mkdir", rc == 0, detail);

	DIR *dir = opendir(d);
	snprintf(detail, sizeof detail, "opendir=%s errno=%d", dir ? "ok" : "(null)", dir ? 0 : errno);
	check("is-a-directory", dir != 0, detail);
	if (dir)
		closedir(dir);

	errno = 0;
	rc = mkdir(d, 0755);
	snprintf(detail, sizeof detail, "rc=%d errno=%d (want EEXIST=%d)", rc, errno, EEXIST);
	check("mkdir-existing", rc < 0 && errno == EEXIST, detail);

	errno = 0;
	rc = mkdir("/mnt/host/no-such-parent/child", 0755);
	snprintf(detail, sizeof detail, "rc=%d errno=%d (want ENOENT=%d)", rc, errno, ENOENT);
	check("mkdir-missing-parent", rc < 0 && errno == ENOENT, detail);

	int fd = open(f, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	if (fd >= 0) {
		write(fd, "x", 1);
		close(fd);
	}
	errno = 0;
	rc = rmdir(d);
	snprintf(detail, sizeof detail, "fd=%d rc=%d errno=%d (want ENOTEMPTY=%d)", fd, rc, errno, ENOTEMPTY);
	check("rmdir-not-empty", fd >= 0 && rc < 0 && errno == ENOTEMPTY, detail);

	errno = 0;
	rc = rmdir(f);
	snprintf(detail, sizeof detail, "rc=%d errno=%d (want ENOTDIR=%d)", rc, errno, ENOTDIR);
	check("rmdir-a-file", rc < 0 && errno == ENOTDIR, detail);

	errno = 0;
	rc = unlink(d);
	snprintf(detail, sizeof detail, "rc=%d errno=%d (want EISDIR=%d)", rc, errno, EISDIR);
	check("unlink-a-directory", rc < 0 && errno == EISDIR, detail);

	rc = unlink(f);
	int rc2 = rmdir(d);
	errno = 0;
	int gone = access(d, F_OK) < 0 && errno == ENOENT;
	snprintf(detail, sizeof detail, "unlink=%d rmdir=%d gone=%d", rc, rc2, gone);
	check("rmdir-empty", rc == 0 && rc2 == 0 && gone, detail);

	/* Relative to the domain's cwd, through the same join as open and rename. */
	rc = chdir("/mnt/host");
	rc2 = mkdir("mkdir-rel", 0755);
	dir = opendir("/mnt/host/mkdir-rel");
	if (dir)
		closedir(dir);
	int rc3 = rmdir("mkdir-rel");
	errno = 0;
	gone = access("/mnt/host/mkdir-rel", F_OK) < 0 && errno == ENOENT;
	snprintf(detail, sizeof detail, "chdir=%d mkdir=%d seen=%d rmdir=%d gone=%d", rc, rc2, dir != 0, rc3, gone);
	check("relative-mkdir-rmdir", rc == 0 && rc2 == 0 && dir && rc3 == 0 && gone, detail);

	printf("MKDIR-RMDIR-DONE failures=%d\n", failures);
	return failures;
}
