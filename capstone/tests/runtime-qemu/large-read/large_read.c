/* A domain reads a 64 KiB file from the 9p share, whole and in odd pieces, and
 * writes one back, through the hostcall file service. run.sh stages the file
 * with a known byte pattern; every check prints one line and main returns the
 * number that failed. */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define SIZE (64 * 1024)
static unsigned char buf[SIZE + 16];
static int failures;

static void check(const char *name, int ok, const char *detail)
{
	printf("LARGE-READ %s %s %s\n", ok ? "PASS" : "FAIL", name, detail);
	if (!ok)
		failures++;
}

static int pattern_ok(const unsigned char *p, long off, long n)
{
	for (long i = 0; i < n; i++)
		if (p[i] != (unsigned char)((off + i) * 7 + 3))
			return 0;
	return 1;
}

int main(void)
{
	char detail[96];
	int fd = open("/mnt/host/large.bin", O_RDONLY);
	snprintf(detail, sizeof detail, "errno=%d", fd < 0 ? errno : 0);
	check("open", fd >= 0, detail);
	if (fd < 0)
		goto done;

	ssize_t n = read(fd, buf, SIZE + 16);
	snprintf(detail, sizeof detail, "n=%zd errno=%d (want %d)", n, n < 0 ? errno : 0, SIZE);
	check("read-whole", n == SIZE && pattern_ok(buf, 0, SIZE), detail);

	/* Odd sizes and offsets: the service chunks at the payload size. */
	long total = 0, bad = 0;
	lseek(fd, 0, SEEK_SET);
	for (int k = 1; total < SIZE; k++) {
		size_t want = (size_t)(k * 1237) % 9001 + 1;
		ssize_t got = read(fd, buf, want);
		if (got <= 0) { bad = got < 0 ? -errno : -1; break; }
		if (!pattern_ok(buf, total, got)) { bad = total; break; }
		total += got;
	}
	snprintf(detail, sizeof detail, "total=%ld bad=%ld", total, bad);
	check("read-pieces", total == SIZE && bad == 0, detail);
	close(fd);

	fd = open("/mnt/host/written.bin", O_WRONLY | O_CREAT | O_TRUNC, 0644);
	for (long i = 0; i < SIZE; i++)
		buf[i] = (unsigned char)(i * 7 + 3);
	n = fd < 0 ? -1 : write(fd, buf, SIZE);
	snprintf(detail, sizeof detail, "n=%zd errno=%d", n, n < 0 ? errno : 0);
	check("write-whole", n == SIZE, detail);
	if (fd >= 0)
		close(fd);
	fd = open("/mnt/host/written.bin", O_RDONLY);
	memset(buf, 0, SIZE);
	n = fd < 0 ? -1 : read(fd, buf, SIZE);
	snprintf(detail, sizeof detail, "n=%zd", n);
	check("write-readback", n == SIZE && pattern_ok(buf, 0, SIZE), detail);
	if (fd >= 0)
		close(fd);
done:
	printf("LARGE-READ-DONE failures=%d\n", failures);
	return failures;
}
