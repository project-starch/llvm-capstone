/* pipe2, read/write on a pipe, fcntl/fstat/lseek on it, and poll, in a musl
 * domain: the self-pipe PostgreSQL's latch is built on (WAIT_USE_SELF_PIPE),
 * kept in the domain, where a pipe never blocks because nothing else could
 * fill or drain it.
 *
 * An empty pipe reads EAGAIN and polls not ready; a byte written polls POLLIN
 * and reads back; a fifo by fstat, ESPIPE by lseek, O_NONBLOCK by F_GETFL; a
 * full pipe takes a short write, then EAGAIN, and polls without POLLOUT until
 * drained; a closed write end polls POLLHUP and reads 0; a closed read end
 * makes writes EPIPE; a file polls ready, stdout POLLOUT, a stray fd POLLNVAL;
 * an unknown pipe2 flag is EINVAL. One poll with a timeout and nothing ready
 * returns 0 at once, and run.sh checks the exit report says so:
 * "capstone-domain: NO-OP syscalls: 73" (ppoll), exactly once. Every check
 * prints one line; the last line counts the failures and main returns that
 * count. */
#define _GNU_SOURCE /* pipe2 */
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* The runtime's pipe capacity, Linux's default (runtime/hostcall.c HC_PIPE_BYTES). */
#define PIPE_CAPACITY 65536

static int failures;

static void check(const char *name, int ok, const char *detail)
{
	printf("PIPE-POLL %s %s %s\n", ok ? "PASS" : "FAIL", name, detail);
	if (!ok)
		failures++;
}

int main(void)
{
	char detail[200], buf[64];
	static char big[PIPE_CAPACITY + 1000];
	int fds[2], rc;
	long n;

	errno = 0;
	rc = pipe2(fds, O_NONBLOCK | O_CLOEXEC);
	snprintf(detail, sizeof detail, "rc=%d errno=%d rd=%d wr=%d", rc, rc ? errno : 0, fds[0], fds[1]);
	check("pipe2", rc == 0 && fds[0] >= 3 && fds[1] >= 3 && fds[0] != fds[1], detail);
	if (rc != 0) {
		printf("PIPE-POLL-DONE failures=%d\n", failures);
		return failures;
	}
	int rd = fds[0], wr = fds[1];

	errno = 0;
	n = read(rd, buf, sizeof buf);
	snprintf(detail, sizeof detail, "n=%ld errno=%d (want EAGAIN=%d)", n, errno, EAGAIN);
	check("empty-read-eagain", n < 0 && errno == EAGAIN, detail);

	struct pollfd pf = { rd, POLLIN, 0 };
	rc = poll(&pf, 1, 0);
	snprintf(detail, sizeof detail, "rc=%d revents=%#x (want 0)", rc, pf.revents);
	check("poll-empty-now", rc == 0 && pf.revents == 0, detail);

	/* Asked to wait 10 ms with nothing that could arrive: returns as a timeout,
	   and the exit report says a wait was skipped. */
	pf.revents = 0;
	rc = poll(&pf, 1, 10);
	snprintf(detail, sizeof detail, "rc=%d revents=%#x (want 0: nothing can arrive)", rc, pf.revents);
	check("poll-empty-timeout", rc == 0 && pf.revents == 0, detail);

	n = write(wr, "hello", 5);
	pf.revents = 0;
	rc = poll(&pf, 1, 0);
	snprintf(detail, sizeof detail, "written=%ld poll=%d revents=%#x (want POLLIN=%#x)", n, rc, pf.revents, POLLIN);
	check("write-then-pollin", n == 5 && rc == 1 && (pf.revents & POLLIN), detail);

	struct stat st;
	rc = fstat(rd, &st);
	snprintf(detail, sizeof detail, "rc=%d fifo=%d size=%ld", rc, S_ISFIFO(st.st_mode), (long)st.st_size);
	check("fstat-fifo", rc == 0 && S_ISFIFO(st.st_mode) && st.st_size == 5, detail);
	errno = 0;
	n = lseek(rd, 0, SEEK_CUR);
	snprintf(detail, sizeof detail, "n=%ld errno=%d (want ESPIPE=%d)", n, errno, ESPIPE);
	check("lseek-espipe", n < 0 && errno == ESPIPE, detail);
	rc = fcntl(rd, F_GETFL);
	snprintf(detail, sizeof detail, "fl=%#x nonblock=%d", rc, (rc & O_NONBLOCK) != 0);
	check("fcntl-getfl-nonblock", rc >= 0 && (rc & O_NONBLOCK), detail);
	rc = fcntl(rd, F_SETFD, FD_CLOEXEC);
	snprintf(detail, sizeof detail, "rc=%d", rc);
	check("fcntl-setfd", rc == 0, detail);

	memset(buf, 0, sizeof buf);
	n = read(rd, buf, 3);
	long m = read(rd, buf + 3, 10);
	errno = 0;
	long z = read(rd, buf + 5, 10);
	snprintf(detail, sizeof detail, "n=%ld m=%ld data=%.5s then=%ld errno=%d", n, m, buf, z, errno);
	check("read-in-pieces", n == 3 && m == 2 && memcmp(buf, "hello", 5) == 0 && z < 0 && errno == EAGAIN, detail);

	memset(big, 'x', sizeof big);
	n = write(wr, big, sizeof big);
	errno = 0;
	m = write(wr, "y", 1);
	struct pollfd pw = { wr, POLLOUT, 0 };
	rc = poll(&pw, 1, 0);
	snprintf(detail, sizeof detail, "short=%ld (want %d) then=%ld errno=%d (want EAGAIN) pollout=%d", n, PIPE_CAPACITY, m, errno, rc);
	check("full-pipe", n == PIPE_CAPACITY && m < 0 && errno == EAGAIN && rc == 0, detail);
	long drained = 0;
	while ((n = read(rd, big, sizeof big)) > 0)
		drained += n;
	pw.revents = 0;
	rc = poll(&pw, 1, 0);
	snprintf(detail, sizeof detail, "drained=%ld (want %d) pollout=%d revents=%#x", drained, PIPE_CAPACITY, rc, pw.revents);
	check("drain-then-pollout", drained == PIPE_CAPACITY && rc == 1 && (pw.revents & POLLOUT), detail);

	close(wr);
	pf.revents = 0;
	rc = poll(&pf, 1, 0);
	n = read(rd, buf, sizeof buf);
	snprintf(detail, sizeof detail, "poll=%d revents=%#x (want POLLHUP=%#x) read=%ld (want 0)", rc, pf.revents, POLLHUP, n);
	check("writer-closed", rc == 1 && (pf.revents & POLLHUP) && n == 0, detail);
	close(rd);

	int p2[2];
	rc = pipe(p2);
	close(p2[0]);
	errno = 0;
	n = write(p2[1], "x", 1);
	snprintf(detail, sizeof detail, "pipe=%d n=%ld errno=%d (want EPIPE=%d)", rc, n, errno, EPIPE);
	check("reader-closed-epipe", rc == 0 && n < 0 && errno == EPIPE, detail);
	close(p2[1]);

	int fd = open("/mnt/host/pipe-poll.dat", O_RDWR | O_CREAT | O_TRUNC, 0644);
	struct pollfd three[3] = { { fd, POLLIN | POLLOUT, 0 }, { 1, POLLOUT, 0 }, { 42, POLLIN, 0 } };
	rc = poll(three, 3, 0);
	snprintf(detail, sizeof detail, "rc=%d file=%#x stdout=%#x stray=%#x (want NVAL=%#x)", rc,
	         three[0].revents, three[1].revents, three[2].revents, POLLNVAL);
	check("poll-file-stdout-stray", rc == 3 && three[0].revents == (POLLIN | POLLOUT) &&
	      three[1].revents == POLLOUT && three[2].revents == POLLNVAL, detail);
	close(fd);
	unlink("/mnt/host/pipe-poll.dat");

	errno = 0;
	rc = pipe2(fds, 1);
	snprintf(detail, sizeof detail, "rc=%d errno=%d (want EINVAL=%d)", rc, errno, EINVAL);
	check("pipe2-bad-flag", rc < 0 && errno == EINVAL, detail);

	printf("PIPE-POLL-DONE failures=%d\n", failures);
	return failures;
}
