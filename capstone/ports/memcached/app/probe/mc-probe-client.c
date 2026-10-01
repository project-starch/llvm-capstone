/* M0 probe client: a NATIVE guest program (glibc, riscv64) talking to the domain probe over
 * loopback. Opens NCONN connections first (retrying until the listener is up, for up to 30 s),
 * then sends "ping <j>" on each and reads the reply, which names the worker that served it.
 * Exit 0 only when every reply is the echo of its own line. */
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define PORT 11299
#define NCONN 8

static int open_conn(void) {
  struct sockaddr_in sa = {.sin_family = AF_INET, .sin_port = htons(PORT)};
  inet_pton(AF_INET, "127.0.0.1", &sa.sin_addr);
  for (int tries = 0; tries < 300; tries++) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s >= 0 && connect(s, (struct sockaddr *)&sa, sizeof sa) == 0) return s;
    if (s >= 0) close(s);
    usleep(100 * 1000);
  }
  return -1;
}

int main(void) {
  int fds[NCONN], per_worker[8] = {0}, ok = 0;
  for (int j = 0; j < NCONN; j++) {
    fds[j] = open_conn();
    if (fds[j] < 0) { printf("CLIENT connect %d failed errno=%d\n", j, errno); return 2; }
  }
  for (int j = 0; j < NCONN; j++) {
    char line[32];
    int len = snprintf(line, sizeof line, "ping %d\n", j);
    if (write(fds[j], line, (size_t)len) != len) printf("CLIENT write %d failed\n", j);
  }
  for (int j = 0; j < NCONN; j++) {
    char buf[160] = {0}, want[32];
    ssize_t n = 0, r;
    while (n < (ssize_t)sizeof buf - 1 && (r = read(fds[j], buf + n, sizeof buf - 1 - (size_t)n)) > 0) n += r;
    int w = -1;
    snprintf(want, sizeof want, " ping %d\n", j);
    if (sscanf(buf, "w%d", &w) == 1 && w >= 0 && w < 8 && strstr(buf, want)) { ok++; per_worker[w]++; }
    printf("CLIENT reply %d: %s", j, n > 0 ? buf : "(none)\n");
    close(fds[j]);
  }
  printf("CLIENT ok=%d/%d workers=%d,%d,%d,%d\n", ok, NCONN, per_worker[0], per_worker[1], per_worker[2],
         per_worker[3]);
  return ok == NCONN ? 0 : 1;
}
