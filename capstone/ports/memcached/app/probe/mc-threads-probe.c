/* M0 threads probe for the memcached port: memcached's threading, reduced to its shape.
 *
 * main listens on 127.0.0.1:PORT, waits in epoll_wait with a 1 s timeout (memcached's clock
 * tick), accepts, and hands each connection round-robin to one of NWORKERS workers through a
 * per-worker queue and an eventfd (memcached's thread.c notify). Each worker blocks in epoll_wait on
 * its own epoll set: its eventfd and the connections it owns; it answers one line per connection
 * with "w<i> <line>" and closes it. One more thread loops on pthread_cond_timedwait (the LRU
 * maintainer's shape). SIGTERM sets a flag; main leaves its loop, stops and joins every thread, and
 * prints STOPPED with the per-worker counts. Predictions: docs/plans/2026-10-01-memcached-full-app-port.md. */
#define _GNU_SOURCE
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define PORT 11299
#define NWORKERS 4
#define QCAP 64

struct worker {
  pthread_t tid;
  int idx, efd, epfd, served;
  pthread_mutex_t lock;
  int queue[QCAP], qn;
};

static struct worker workers[NWORKERS];
static volatile sig_atomic_t stop_flag;
static volatile int workers_stop;
static pthread_mutex_t tick_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t tick_cond = PTHREAD_COND_INITIALIZER;
static int ticks;

static void on_term(int sig) { (void)sig; stop_flag = 1; }

static void *worker_main(void *arg) {
  struct worker *w = arg;
  struct epoll_event ev = {.events = EPOLLIN, .data.fd = w->efd};
  epoll_ctl(w->epfd, EPOLL_CTL_ADD, w->efd, &ev);
  while (!workers_stop) {
    struct epoll_event got[8];
    int n = epoll_wait(w->epfd, got, 8, -1);
    if (n < 0) {
      if (errno == EINTR) continue;
      printf("w%d epoll_wait errno=%d\n", w->idx, errno);
      break;
    }
    for (int i = 0; i < n; i++) {
      int fd = got[i].data.fd;
      if (fd == w->efd) {
        uint64_t v;
        if (read(w->efd, &v, sizeof v) != sizeof v) continue;
        pthread_mutex_lock(&w->lock);
        for (int q = 0; q < w->qn; q++) {
          struct epoll_event ce = {.events = EPOLLIN, .data.fd = w->queue[q]};
          epoll_ctl(w->epfd, EPOLL_CTL_ADD, w->queue[q], &ce);
        }
        w->qn = 0;
        pthread_mutex_unlock(&w->lock);
        continue;
      }
      char line[128], reply[160];
      ssize_t r = read(fd, line, sizeof line - 1);
      if (r > 0) {
        line[r] = 0;
        char *nl = strchr(line, '\n');
        if (nl) *nl = 0;
        int len = snprintf(reply, sizeof reply, "w%d %s\n", w->idx, line);
        if (write(fd, reply, (size_t)len) == len) w->served++;
      }
      epoll_ctl(w->epfd, EPOLL_CTL_DEL, fd, NULL);
      close(fd);
    }
  }
  return NULL;
}

static void *ticker_main(void *arg) {
  (void)arg;
  pthread_mutex_lock(&tick_lock);
  while (!workers_stop) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_nsec += 100 * 1000 * 1000;
    if (ts.tv_nsec >= 1000000000) { ts.tv_sec++; ts.tv_nsec -= 1000000000; }
    pthread_cond_timedwait(&tick_cond, &tick_lock, &ts);
    ticks++;
  }
  pthread_mutex_unlock(&tick_lock);
  return NULL;
}

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  signal(SIGTERM, on_term);
  signal(SIGPIPE, SIG_IGN);
  int ls = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
  int one = 1;
  setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  struct sockaddr_in sa = {.sin_family = AF_INET, .sin_port = htons(PORT),
                           .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
  if (ls < 0 || bind(ls, (struct sockaddr *)&sa, sizeof sa) || listen(ls, 64)) {
    printf("LISTEN FAILED errno=%d\n", errno);
    return 2;
  }
  int joined = 0;
  for (int i = 0; i < NWORKERS; i++) {
    struct worker *w = &workers[i];
    w->idx = i;
    w->efd = eventfd(0, EFD_NONBLOCK);
    w->epfd = epoll_create1(0);
    pthread_mutex_init(&w->lock, NULL);
    if (w->efd < 0 || w->epfd < 0 || pthread_create(&w->tid, NULL, worker_main, w)) {
      printf("WORKER %d FAILED errno=%d\n", i, errno);
      return 3;
    }
  }
  pthread_t ticker;
  if (pthread_create(&ticker, NULL, ticker_main, NULL)) { printf("TICKER FAILED\n"); return 3; }
  int ep = epoll_create1(0);
  struct epoll_event lev = {.events = EPOLLIN, .data.fd = ls};
  epoll_ctl(ep, EPOLL_CTL_ADD, ls, &lev);
  printf("LISTENING %d workers=%d\n", PORT, NWORKERS);
  int next = 0, accepted = 0;
  while (!stop_flag) {
    struct epoll_event ev[4];
    int n = epoll_wait(ep, ev, 4, 1000);
    if (n < 0 && errno != EINTR) { printf("main epoll_wait errno=%d\n", errno); break; }
    for (;;) {
      int c = accept4(ls, NULL, NULL, SOCK_CLOEXEC);
      if (c < 0) break;
      struct worker *w = &workers[next];
      next = (next + 1) % NWORKERS;
      pthread_mutex_lock(&w->lock);
      if (w->qn < QCAP) w->queue[w->qn++] = c; else close(c);
      pthread_mutex_unlock(&w->lock);
      uint64_t one64 = 1;
      if (write(w->efd, &one64, sizeof one64) != sizeof one64) printf("notify failed\n");
      accepted++;
    }
  }
  workers_stop = 1;
  for (int i = 0; i < NWORKERS; i++) {
    uint64_t one64 = 1;
    (void)!write(workers[i].efd, &one64, sizeof one64);
  }
  pthread_mutex_lock(&tick_lock);
  pthread_cond_signal(&tick_cond);
  pthread_mutex_unlock(&tick_lock);
  for (int i = 0; i < NWORKERS; i++) joined += pthread_join(workers[i].tid, NULL) == 0;
  joined += pthread_join(ticker, NULL) == 0;
  printf("STOPPED joined=%d accepted=%d served=%d,%d,%d,%d ticks=%d\n", joined, accepted,
         workers[0].served, workers[1].served, workers[2].served, workers[3].served, ticks);
  return 0;
}
