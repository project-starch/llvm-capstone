/* mc-stats: after the protocol oracle's workload, ask a running memcached for its allocator statistics.
 *
 *   mc-stats PORT        connects to 127.0.0.1:PORT, sends `stats`, `stats slabs` and `stats items`
 *                        one after the other, and writes each reply (STAT lines through END) to
 *                        stdout under a "== <command>" header. Exit 0 when all three ended with END.
 *
 * Built twice from this source: for the host (the native reference) and statically for the guest.
 * It stores nothing in the server: the three commands only read counters.
 */
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static int fd;
static char rb[1 << 16];
static size_t rn, rpos;

static int getch(void) {
  if (rpos == rn) {
    struct pollfd p = {.fd = fd, .events = POLLIN};
    if (poll(&p, 1, 60000) <= 0) return -1;
    ssize_t r = read(fd, rb, sizeof rb);
    if (r <= 0) return -1;
    rn = (size_t)r; rpos = 0;
  }
  return (unsigned char)rb[rpos++];
}

static int one(const char *command) {
  char line[4096];
  printf("== %s\n", command);
  dprintf(fd, "%s\r\n", command);
  for (;;) {
    size_t n = 0; int c;
    while ((c = getch()) >= 0 && n + 1 < sizeof line) { line[n++] = (char)c; if (c == '\n') break; }
    line[n] = 0;
    if (c < 0) { printf("<EOF>\n"); return 1; }
    fputs(line, stdout);
    if (!strcmp(line, "END\r\n")) return 0;
    if (strncmp(line, "STAT ", 5)) return 1;
  }
}

int main(int argc, char **argv) {
  if (argc != 2) { fprintf(stderr, "usage: mc-stats PORT\n"); return 2; }
  struct sockaddr_in sa = {.sin_family = AF_INET, .sin_port = htons((unsigned short)atoi(argv[1]))};
  inet_pton(AF_INET, "127.0.0.1", &sa.sin_addr);
  fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0 || connect(fd, (struct sockaddr *)&sa, sizeof sa)) { perror("mc-stats: connect"); return 3; }
  int bad = one("stats") | one("stats slabs") | one("stats items");
  close(fd);
  return bad;
}
