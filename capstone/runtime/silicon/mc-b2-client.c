/* B2 (docs/plans/b0-silicon-delegated-runtime.md): the memcached milestone's client, a NATIVE guest program (the
 * guest's own libc) talking to the memcached domain over loopback.
 *
 * It connects (retrying for up to 60 s while the domain starts), then sends `version`, `set k 0 0 1` with value x,
 * and `get k`, and prints every reply line as `B2 < <line>`. Exit 0 only when the replies are exactly
 * VERSION 1.6.45, STORED, and VALUE k 0 1 / x / END. Distinct codes say where it stopped: 2 no connection, 3 the
 * version reply, 4 the set reply, 5 the get reply, 6 a read failed or the server closed. */
#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define PORT 21299

static int fd;
static char buf[4096];
static size_t have;

/* One CRLF-terminated line into out (without the CRLF); 0 on EOF or error. */
static int line(char *out, size_t cap)
{
	for (;;) {
		char *nl = memchr(buf, '\n', have);
		if (nl) {
			size_t n = (size_t)(nl - buf) + 1, len = n;
			if (len && buf[len - 1] == '\n') len--;
			if (len && buf[len - 1] == '\r') len--;
			if (len >= cap) len = cap - 1;
			memcpy(out, buf, len);
			out[len] = 0;
			memmove(buf, buf + n, have - n);
			have -= n;
			printf("B2 < %s\n", out);
			fflush(stdout);
			return 1;
		}
		if (have == sizeof buf) return 0;
		ssize_t r = read(fd, buf + have, sizeof buf - have);
		if (r <= 0) return 0;
		have += (size_t)r;
	}
}

static int say(const char *s)
{
	printf("B2 > %s", s);
	fflush(stdout);
	size_t n = strlen(s);
	return write(fd, s, n) == (ssize_t)n;
}

int main(void)
{
	struct sockaddr_in sa = {.sin_family = AF_INET, .sin_port = htons(PORT)};
	inet_pton(AF_INET, "127.0.0.1", &sa.sin_addr);
	fd = -1;
	for (int tries = 0; tries < 600 && fd < 0; tries++) {
		int s = socket(AF_INET, SOCK_STREAM, 0);
		if (s >= 0 && connect(s, (struct sockaddr *)&sa, sizeof sa) == 0) {
			fd = s;
			break;
		}
		if (s >= 0) close(s);
		usleep(100 * 1000);
	}
	if (fd < 0) { printf("B2: no connection\n"); return 2; }
	char l[512];
	if (!say("version\r\n") || !line(l, sizeof l)) return 6;
	if (strcmp(l, "VERSION 1.6.45")) return 3;
	if (!say("set k 0 0 1\r\nx\r\n") || !line(l, sizeof l)) return 6;
	if (strcmp(l, "STORED")) return 4;
	if (!say("get k\r\n") || !line(l, sizeof l)) return 6;
	if (strcmp(l, "VALUE k 0 1")) return 5;
	if (!line(l, sizeof l)) return 6;
	if (strcmp(l, "x")) return 5;
	if (!line(l, sizeof l)) return 6;
	if (strcmp(l, "END")) return 5;
	say("quit\r\n");
	close(fd);
	printf("B2: all replies as expected\n");
	return 0;
}
