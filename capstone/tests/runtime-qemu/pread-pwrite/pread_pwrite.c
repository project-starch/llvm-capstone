/* pread, pwrite, preadv and pwritev in a musl domain, through the hostcall's
 * FILE_READ/FILE_WRITE at an explicit offset.
 *
 * One file on the share, written and read at offsets the position never
 * visits, with the position checked before and after each call; one transfer
 * of 10000 bytes at an offset, which the 4 KiB payload region splits into
 * three rounds; the vector forms across two buffers; and the answers for
 * stdout (ESPIPE) and for a descriptor nobody opened (EBADF). Every check
 * prints one line; the last line counts the failures and main returns that
 * count, which the host reports as the exit status. */
#define _XOPEN_SOURCE 700
#define _BSD_SOURCE /* preadv, pwritev */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/uio.h>
#include <unistd.h>

static int failures;

static void check(const char *name, int ok, const char *detail)
{
	printf("PREAD-PWRITE %s %s %s\n", ok ? "PASS" : "FAIL", name, detail);
	if (!ok)
		failures++;
}

#define BIG 10000
#define BIG_AT 4096
static char out[BIG], in[BIG];

int main(void)
{
	static const char path[] = "/mnt/host/pread-pwrite.dat";
	char detail[160];
	char buf[16];
	long n;

	int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
	snprintf(detail, sizeof detail, "fd=%d errno=%d", fd, fd < 0 ? errno : 0);
	check("open", fd >= 0, detail);
	if (fd < 0) {
		printf("PREAD-PWRITE-DONE failures=%d\n", failures);
		return failures;
	}

	/* write() sets the position to 10; pwrite far beyond it must not move it. */
	n = write(fd, "0123456789", 10);
	snprintf(detail, sizeof detail, "n=%ld", n);
	check("write", n == 10, detail);

	errno = 0;
	n = pwrite(fd, "ABCD", 4, 100);
	snprintf(detail, sizeof detail, "n=%ld errno=%d", n, n < 0 ? errno : 0);
	check("pwrite", n == 4, detail);

	long pos = lseek(fd, 0, SEEK_CUR);
	snprintf(detail, sizeof detail, "pos=%ld (want 10)", pos);
	check("pwrite-keeps-position", pos == 10, detail);

	memset(buf, 0, sizeof buf);
	errno = 0;
	n = pread(fd, buf, 4, 100);
	snprintf(detail, sizeof detail, "n=%ld data=%.4s errno=%d", n, buf, n < 0 ? errno : 0);
	check("pread-data", n == 4 && memcmp(buf, "ABCD", 4) == 0, detail);

	pos = lseek(fd, 0, SEEK_CUR);
	snprintf(detail, sizeof detail, "pos=%ld (want 10)", pos);
	check("pread-keeps-position", pos == 10, detail);

	/* read() goes on from 10: the hole pwrite left, four zero bytes. */
	memset(buf, 'x', sizeof buf);
	n = read(fd, buf, 4);
	int zeros = n == 4 && buf[0] == 0 && buf[1] == 0 && buf[2] == 0 && buf[3] == 0;
	snprintf(detail, sizeof detail, "n=%ld zeros=%d", n, zeros);
	check("read-continues-at-position", zeros, detail);

	long end = lseek(fd, 0, SEEK_END);
	snprintf(detail, sizeof detail, "size=%ld (want 104)", end);
	check("size-after-pwrite", end == 104, detail);

	/* More than the payload region holds: three rounds, one offset each. */
	for (int i = 0; i < BIG; i++)
		out[i] = (char)('a' + (i * 7 + i / 251) % 26);
	n = pwrite(fd, out, BIG, BIG_AT);
	snprintf(detail, sizeof detail, "n=%ld (want %d)", n, BIG);
	check("pwrite-large", n == BIG, detail);
	memset(in, 0, sizeof in);
	n = pread(fd, in, BIG, BIG_AT);
	snprintf(detail, sizeof detail, "n=%ld same=%d", n, n == BIG && memcmp(in, out, BIG) == 0);
	check("pread-large", n == BIG && memcmp(in, out, BIG) == 0, detail);

	n = pread(fd, buf, sizeof buf, BIG_AT + BIG);
	snprintf(detail, sizeof detail, "n=%ld (want 0)", n);
	check("pread-at-end", n == 0, detail);
	n = pread(fd, buf, sizeof buf, BIG_AT + BIG + 100000);
	snprintf(detail, sizeof detail, "n=%ld (want 0)", n);
	check("pread-past-end", n == 0, detail);

	/* The vector forms: two entries, one offset, and the second entry's
	   data must land right behind the first's. */
	char v1[8], v2[7];
	struct iovec wv[2] = { { "vec-one-", 8 }, { "vec-two", 7 } };
	struct iovec rv[2] = { { v1, sizeof v1 }, { v2, sizeof v2 } };
	errno = 0;
	n = pwritev(fd, wv, 2, 20000);
	snprintf(detail, sizeof detail, "n=%ld errno=%d", n, n < 0 ? errno : 0);
	check("pwritev", n == 15, detail);
	memset(v1, 0, sizeof v1);
	memset(v2, 0, sizeof v2);
	n = preadv(fd, rv, 2, 20000);
	snprintf(detail, sizeof detail, "n=%ld data=%.8s|%.7s", n, v1, v2);
	check("preadv", n == 15 && memcmp(v1, "vec-one-", 8) == 0 && memcmp(v2, "vec-two", 7) == 0, detail);
	n = pread(fd, buf, 15, 20000);
	snprintf(detail, sizeof detail, "n=%ld data=%.15s", n, buf);
	check("pwritev-contiguous", n == 15 && memcmp(buf, "vec-one-vec-two", 15) == 0, detail);

	/* The position is where lseek(SEEK_END) put it, 104: six offset calls
	   since, none of them moved it. (The first version of this check wanted
	   14 and forgot that SEEK_END seeks; the runtime was right.) */
	pos = lseek(fd, 0, SEEK_CUR);
	snprintf(detail, sizeof detail, "pos=%ld (want 104)", pos);
	check("position-untouched-by-all", pos == 104, detail);

	errno = 0;
	n = pread(1, buf, 1, 0);
	snprintf(detail, sizeof detail, "n=%ld errno=%d (want ESPIPE=%d)", n, errno, ESPIPE);
	check("pread-stdout", n < 0 && errno == ESPIPE, detail);
	errno = 0;
	n = pwrite(1, "x", 1, 0);
	snprintf(detail, sizeof detail, "n=%ld errno=%d (want ESPIPE=%d)", n, errno, ESPIPE);
	check("pwrite-stdout", n < 0 && errno == ESPIPE, detail);
	errno = 0;
	n = pread(9, buf, 1, 0);
	snprintf(detail, sizeof detail, "n=%ld errno=%d (want EBADF=%d)", n, errno, EBADF);
	check("pread-unopened", n < 0 && errno == EBADF, detail);

	close(fd);
	unlink(path);
	printf("PREAD-PWRITE-DONE failures=%d\n", failures);
	return failures;
}
