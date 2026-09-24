/* Directory listing in a musl domain, through the hostcall's DIR_READ.
 *
 * run.sh stages /mnt/host/dirtest with three files and a subdirectory, and
 * /mnt/host/manyfiles with 300 files -- enough that musl's 2 KiB readdir buffer
 * needs several DIR_READ rounds, so the cookie has to carry the listing across
 * them. Every check prints one line; the last line counts the failures and
 * main returns that count, which the host reports as the exit status. */
#define _XOPEN_SOURCE 700 /* telldir, seekdir */
#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

static int failures;

static void check(const char *name, int ok, const char *detail)
{
	printf("DIR-READ %s %s %s\n", ok ? "PASS" : "FAIL", name, detail);
	if (!ok)
		failures++;
}

/* Reads the rest of d; counts entries and remembers which of the names were seen. */
static int drain(DIR *d, const char *const *names, int *seen, int n)
{
	int count = 0;
	struct dirent *e;
	errno = 0;
	while ((e = readdir(d))) {
		count++;
		for (int i = 0; i < n; i++)
			if (!strcmp(e->d_name, names[i]))
				seen[i]++;
	}
	return errno ? -errno : count;
}

int main(void)
{
	static const char *const names[] = { ".", "..", "alpha", "beta", "gamma", "sub" };
	enum { N = sizeof names / sizeof names[0] };
	char detail[128];

	DIR *d = opendir("/mnt/host/dirtest");
	snprintf(detail, sizeof detail, "errno=%d", d ? 0 : errno);
	check("opendir", d != 0, detail);
	if (d) {
		int seen[N] = { 0 };
		int count = drain(d, names, seen, N);
		int each_once = 1;
		for (int i = 0; i < N; i++)
			each_once &= seen[i] == 1;
		snprintf(detail, sizeof detail, "count=%d (want 6)", count);
		check("list", count == N && each_once, detail);

		rewinddir(d);
		int again[N] = { 0 };
		count = drain(d, names, again, N);
		snprintf(detail, sizeof detail, "count=%d (want 6)", count);
		check("rewinddir", count == N, detail);

		/* telldir after two entries, read on, seekdir back: the third entry again. */
		rewinddir(d);
		readdir(d);
		readdir(d);
		long pos = telldir(d);
		struct dirent *third = readdir(d);
		char name3[256] = "";
		if (third)
			snprintf(name3, sizeof name3, "%s", third->d_name);
		while (readdir(d))
			;
		seekdir(d, pos);
		struct dirent *again3 = readdir(d);
		snprintf(detail, sizeof detail, "first=%s again=%s", name3,
		         again3 ? again3->d_name : "(none)");
		check("seekdir", third && again3 && !strcmp(name3, again3->d_name), detail);
		closedir(d);
	}

	d = opendir("/mnt/host/manyfiles");
	if (d) {
		int count = drain(d, names, (int[N]){ 0 }, N);
		snprintf(detail, sizeof detail, "count=%d (want 302)", count);
		check("many", count == 302, detail);
		closedir(d);
	} else {
		snprintf(detail, sizeof detail, "opendir errno=%d", errno);
		check("many", 0, detail);
	}

	errno = 0;
	d = opendir("/mnt/host/no-such-directory");
	snprintf(detail, sizeof detail, "errno=%d (want ENOENT=%d)", errno, ENOENT);
	check("missing", !d && errno == ENOENT, detail);

	errno = 0;
	d = opendir("/mnt/host/dirtest/alpha");
	snprintf(detail, sizeof detail, "errno=%d (want ENOTDIR=%d)", errno, ENOTDIR);
	check("not-a-directory", !d && errno == ENOTDIR, detail);

	printf("DIR-READ-DONE failures=%d\n", failures);
	return failures;
}
