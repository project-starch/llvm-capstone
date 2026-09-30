/* Probe B, native phase (docs/plans/delegation-threads.md): the launcher's
 * parking queue with ordinary Linux threads. The park code's test hook stops
 * a chosen thread at a named point, so each interleaving is forced rather
 * than hoped for. One case per run: b1, b2, b3-timeout-select,
 * b3-timeout-abort, b3-eintr-select, b3-eintr-abort, b4, b5, b12, b13a,
 * b13b, b13c, b13d. */
#define _GNU_SOURCE
#include "../../linux/park.h"

#include <assert.h>
#include <errno.h>
#include <linux/futex.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "park-test:%d: %s\n", __LINE__, #x); _exit(1); } } while (0)

static struct capstone_park park;
static _Atomic uint64_t gen[4];

/* ---- stop points ---- */
static struct {
  _Atomic int point;
  struct capstone_park_record *_Atomic record; /* stop only this record's thread, or any when NULL */
  _Atomic int armed;
  sem_t arrived, release;
} stop;
static _Atomic int sleeping; /* threads that reached CAPSTONE_PARK_SLEEPING */

static void hook(enum capstone_park_point point, struct capstone_park_record *record) {
  int armed = 1;
  if (point == CAPSTONE_PARK_SLEEPING)
    atomic_fetch_add(&sleeping, 1);
  if (atomic_load_explicit(&stop.armed, memory_order_acquire) &&
      point == (enum capstone_park_point)atomic_load(&stop.point) &&
      (!atomic_load(&stop.record) || atomic_load(&stop.record) == record) &&
      atomic_compare_exchange_strong(&stop.armed, &armed, 0)) {
    sem_post(&stop.arrived);
    while (sem_wait(&stop.release) && errno == EINTR)
      ;
  }
}

static void arm(enum capstone_park_point point, struct capstone_park_record *record) {
  atomic_store(&stop.point, point);
  atomic_store(&stop.record, record);
  atomic_store_explicit(&stop.armed, 1, memory_order_release);
}

static void await_stop(void) {
  while (sem_wait(&stop.arrived) && errno == EINTR)
    ;
}

static void release_stop(void) { sem_post(&stop.release); }

/* ---- waiters ---- */
struct waiter {
  pthread_t thread;
  struct capstone_park_record record;
  uint64_t key, gen;
  const struct timespec *deadline;
  _Atomic int done;
  int outcome;
  long ms; /* how long the wait took */
};

static long now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000 + t.tv_nsec / 1000000;
}

static struct timespec in_ms(long ms) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  t.tv_sec += ms / 1000;
  t.tv_nsec += (ms % 1000) * 1000000;
  if (t.tv_nsec >= 1000000000) {
    t.tv_sec++;
    t.tv_nsec -= 1000000000;
  }
  return t;
}

static void *wait_thread(void *arg) {
  struct waiter *w = arg;
  long t0 = now_ms();
  w->outcome = capstone_park_wait(&park, &w->record, w->key, w->gen, w->deadline);
  w->ms = now_ms() - t0;
  atomic_store(&w->done, 1);
  return NULL;
}

static uint64_t gen_of(uint64_t key) {
  return atomic_load_explicit(&gen[capstone_park_bucket(&park, key)], memory_order_acquire);
}

/* Start a waiter on key and return once it is queued and about to sleep. */
static void start(struct waiter *w, uint64_t key, const struct timespec *deadline) {
  memset(w, 0, sizeof *w);
  w->key = key;
  w->gen = gen_of(key);
  w->deadline = deadline;
  int before = atomic_load(&sleeping);
  CHECK(!pthread_create(&w->thread, NULL, wait_thread, w));
  while (atomic_load(&sleeping) == before)
    usleep(1000);
  usleep(20000); /* into the kernel */
}

/* The waiter's outcome; a waiter still asleep after 5 s fails the case
   instead of hanging it. */
static int finish(struct waiter *w) {
  for (int i = 0; i < 5000 && !atomic_load(&w->done); ++i)
    usleep(1000);
  CHECK(atomic_load(&w->done));
  CHECK(!pthread_join(w->thread, NULL));
  return w->outcome;
}

static void on_signal(int sig) { (void)sig; }

/* Keys for a bucket: the smallest multiples of 8 from start that land in it. */
static uint64_t key_in(unsigned bucket, uint64_t start) {
  for (uint64_t k = start;; k += 8)
    if (capstone_park_bucket(&park, k) == bucket)
      return k;
}

static void setup(unsigned buckets) {
  for (unsigned i = 0; i < 4; ++i)
    atomic_store(&gen[i], 0);
  CHECK(!capstone_park_init(&park, gen, buckets));
  capstone_park_test_hook = hook;
  sem_init(&stop.arrived, 0, 0);
  sem_init(&stop.release, 0, 0);
  struct sigaction sa = {.sa_handler = on_signal}; /* no SA_RESTART: EINTR */
  sigaction(SIGUSR1, &sa, NULL);
}

/* B1: a WAKE between enqueue and sleep is not lost. */
static void b1(void) {
  struct waiter w;
  setup(1);
  memset(&w, 0, sizeof w);
  w.key = 0x1000;
  w.gen = gen_of(w.key);
  arm(CAPSTONE_PARK_ENQUEUED, &w.record);
  CHECK(!pthread_create(&w.thread, NULL, wait_thread, &w));
  await_stop();
  CHECK(capstone_park_wake(&park, 0x1000, 1) == 1);
  release_stop();
  CHECK(finish(&w) == CAPSTONE_PARK_WOKEN);
  CHECK(w.ms < 1000);
  CHECK(atomic_load(&sleeping) == 0); /* it never slept */
}

/* B2: two keys in one bucket; WAKE(A) selects only A, and a spurious native
   wake of B is rechecked inside the wait. B is queued first, so a selection
   that ignored the key would take B. */
static void b2(void) {
  struct waiter a, b;
  setup(1);
  start(&b, 0x2000, NULL);
  start(&a, 0x1000, NULL);
  CHECK(capstone_park_wake(&park, 0x1000, 1) == 1);
  CHECK(finish(&a) == CAPSTONE_PARK_WOKEN);
  int sleeps = atomic_load(&sleeping);
  syscall(SYS_futex, &b.record.notified, FUTEX_WAKE | FUTEX_PRIVATE_FLAG, 1, NULL, NULL, 0);
  for (int i = 0; i < 1000 && atomic_load(&sleeping) == sleeps; ++i)
    usleep(1000);
  usleep(50000);
  CHECK(!atomic_load(&b.done));
  CHECK(atomic_load(&sleeping) > sleeps); /* B went back to sleep */
  CHECK(capstone_park_queued(&park, 0, 0x2000) == 1);
  CHECK(capstone_park_wake(&park, 0x2000, 1) == 1);
  CHECK(finish(&b) == CAPSTONE_PARK_WOKEN);
}

/* B3: an abort (timeout or EINTR) against a selection, in both orders. */
static void b3(int eintr, int select_first) {
  struct waiter w1, w2;
  struct timespec soon;
  setup(1);
  soon = in_ms(eintr ? 60000 : 200);
  start(&w1, 0x1000, &soon);
  start(&w2, 0x1000, NULL);
  if (select_first) {
    arm(CAPSTONE_PARK_ABORTING, &w1.record);
    if (eintr)
      pthread_kill(w1.thread, SIGUSR1);
    await_stop(); /* w1's sleep has ended without a notification */
    CHECK(capstone_park_wake(&park, 0x1000, 1) == 1);
    release_stop();
    CHECK(finish(&w1) == CAPSTONE_PARK_WOKEN); /* the notification wins */
  } else {
    if (eintr)
      pthread_kill(w1.thread, SIGUSR1);
    CHECK(finish(&w1) == (eintr ? CAPSTONE_PARK_EINTR : CAPSTONE_PARK_TIMEOUT));
    CHECK(capstone_park_queued(&park, 0, 0x1000) == 1);
  }
  CHECK(!atomic_load(&w2.done));
  CHECK(capstone_park_wake(&park, 0x1000, 1) == 1);
  CHECK(finish(&w2) == CAPSTONE_PARK_WOKEN);
  CHECK(capstone_park_queued(&park, 0, 0x1000) == 0);
}

static void *wake_thread(void *arg) {
  *(unsigned *)arg = capstone_park_wake(&park, 0x1000, 1);
  return NULL;
}

/* B4: the record is not reset while the waker still owns it, a spurious
   native wake meanwhile changes nothing, and it waits again afterwards. */
static void b4(void) {
  struct waiter w;
  pthread_t waker;
  unsigned woke = 0;
  setup(1);
  start(&w, 0x1000, NULL);
  arm(CAPSTONE_PARK_COMPLETING, &w.record);
  CHECK(!pthread_create(&waker, NULL, wake_thread, &woke));
  await_stop(); /* the waker holds the mutex at its last record access */
  syscall(SYS_futex, &w.record.notified, FUTEX_WAKE | FUTEX_PRIVATE_FLAG, 1, NULL, NULL, 0);
  usleep(100000);
  CHECK(!atomic_load(&w.done));
  CHECK(w.record.state == CAPSTONE_PARK_NOTIFIED);
  release_stop();
  CHECK(!pthread_join(waker, NULL));
  CHECK(woke == 1);
  CHECK(finish(&w) == CAPSTONE_PARK_WOKEN);
  CHECK(w.record.state == CAPSTONE_PARK_IDLE);
  /* The same record waits again. */
  w.key = 0x2000;
  w.gen = gen_of(0x2000);
  atomic_store(&w.done, 0);
  int before = atomic_load(&sleeping);
  CHECK(!pthread_create(&w.thread, NULL, wait_thread, &w));
  while (atomic_load(&sleeping) == before)
    usleep(1000);
  usleep(20000);
  CHECK(capstone_park_wake(&park, 0x2000, 1) == 1);
  CHECK(finish(&w) == CAPSTONE_PARK_WOKEN);
}

/* B5: saturation. */
static void b5(void) {
  struct waiter a1, a2, b;
  struct capstone_park_record delayed = {0}, fresh = {0};
  setup(1);
  atomic_store(&gen[0], UINT64_MAX - 1);
  start(&a1, 0x1000, NULL);
  start(&a2, 0x1000, NULL);
  start(&b, 0x2000, NULL);
  uint64_t early = gen_of(0x1000); /* a waiter delayed before its enqueue */
  CHECK(capstone_park_wake(&park, 0x1000, 1) == 1);
  CHECK(atomic_load(&gen[0]) == UINT64_MAX);
  CHECK(finish(&a1) == CAPSTONE_PARK_WOKEN);
  CHECK(finish(&a2) == CAPSTONE_PARK_RECHECK);
  CHECK(finish(&b) == CAPSTONE_PARK_RECHECK);
  CHECK(capstone_park_wait(&park, &delayed, 0x1000, early, NULL) == CAPSTONE_PARK_RECHECK);
  CHECK(capstone_park_wait(&park, &fresh, 0x1000, gen_of(0x1000), NULL) == CAPSTONE_PARK_RECHECK);
  CHECK(capstone_park_wake(&park, 0x1000, 3) == 0);
  CHECK(capstone_park_requeue(&park, 0x1000, 0x2000, 1, 1) == 0);
  CHECK(atomic_load(&gen[0]) == UINT64_MAX); /* no wrap */
  CHECK(capstone_park_queued(&park, 0, 0x1000) + capstone_park_queued(&park, 0, 0x2000) == 0);
}

/* B12: a REQUEUE with nobody queued still advances the source generation. */
static void b12(void) {
  struct capstone_park_record r = {0};
  setup(2);
  uint64_t src = key_in(0, 0x1000), dst = key_in(1, 0x1000);
  volatile int condition = 0;
  uint64_t g = gen_of(src); /* the waiter read the generation ... */
  CHECK(!condition);        /* ... and checked its condition, then stopped */
  condition = 1;            /* the releaser changes it */
  CHECK(capstone_park_requeue(&park, src, dst, 0, 1) == 0);
  struct timespec bound = in_ms(2000); /* a lost event would time out, not hang */
  CHECK(capstone_park_wait(&park, &r, src, g, &bound) == CAPSTONE_PARK_RECHECK);
}

/* B13: REQUEUE(src, dst, 0, 1) with colliding keys in both buckets, then
   a WAKE, a timeout, a signal, or dst saturated. */
static void b13(char variant) {
  struct waiter s1, s2, cs, cd;
  struct timespec deadline;
  setup(4);
  unsigned bs = 0, bd = 1;
  uint64_t src = key_in(bs, 0x1000), dst = key_in(bd, 0x1000);
  uint64_t csrc = key_in(bs, src + 8), cdst = key_in(bd, dst + 8);
  if (variant == 'd')
    atomic_store(&gen[bd], UINT64_MAX);
  deadline = in_ms(variant == 'b' ? 400 : 1500);
  start(&s1, src, variant == 'b' || variant == 'c' ? &deadline : NULL);
  start(&s2, src, NULL);
  start(&cs, csrc, NULL);
  if (variant != 'd')
    start(&cd, cdst, NULL);
  CHECK(capstone_park_requeue(&park, src, dst, 0, 1) == 1);
  CHECK(capstone_park_queued(&park, bs, src) == 1);
  if (variant == 'a') {
    CHECK(capstone_park_queued(&park, bd, dst) == 1);
    CHECK(capstone_park_wake(&park, dst, 1) == 1);
    CHECK(finish(&s1) == CAPSTONE_PARK_WOKEN);
  } else if (variant == 'b') {
    CHECK(finish(&s1) == CAPSTONE_PARK_TIMEOUT);
    CHECK(s1.ms >= 350);
  } else if (variant == 'c') {
    pthread_kill(s1.thread, SIGUSR1);
    CHECK(finish(&s1) == CAPSTONE_PARK_EINTR);
    CHECK(capstone_park_queued(&park, bd, dst) == 0);
    /* The continuation keeps the original deadline, not a fresh budget. */
    CHECK(capstone_park_wait(&park, &s1.record, dst, gen_of(dst), &deadline) ==
          CAPSTONE_PARK_TIMEOUT);
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    long late_ms = (t.tv_sec - deadline.tv_sec) * 1000 + (t.tv_nsec - deadline.tv_nsec) / 1000000;
    CHECK(late_ms >= 0 && late_ms < 200);
  } else {
    CHECK(finish(&s1) == CAPSTONE_PARK_RECHECK); /* released at the saturated dst, counted */
  }
  /* No record in two queues; the colliding waiters stayed where they were. */
  CHECK(capstone_park_queued(&park, bd, dst) == 0);
  CHECK(capstone_park_queued(&park, bs, csrc) == 1);
  CHECK(!atomic_load(&s2.done) && !atomic_load(&cs.done));
  CHECK(capstone_park_wake(&park, src, 1) == 1);
  CHECK(capstone_park_wake(&park, csrc, 1) == 1);
  CHECK(finish(&s2) == CAPSTONE_PARK_WOKEN && finish(&cs) == CAPSTONE_PARK_WOKEN);
  if (variant != 'd') {
    CHECK(!atomic_load(&cd.done));
    CHECK(capstone_park_wake(&park, cdst, 1) == 1);
    CHECK(finish(&cd) == CAPSTONE_PARK_WOKEN);
  }
}

/* A sleep that ends as RETRY (the launcher's signal stub under SA_RESTART):
   the wait is aborted like EINTR, its record leaves the queue, and a later
   WAKE selects nobody. The second sleep of the record would really sleep. */
static int retry_sleeps;
static long retrying_sleep(void *context, _Atomic uint32_t *word, const struct timespec *deadline) {
  (void)context; (void)word; (void)deadline;
  ++retry_sleeps;
  return CAPSTONE_PARK_SLEEP_RETRY;
}

static void retry(void) {
  setup(1);
  struct capstone_park_record record = {0};
  CHECK(capstone_park_wait_with(&park, &record, 0x3000, 0, NULL, retrying_sleep, NULL) ==
        CAPSTONE_PARK_RETRY);
  CHECK(retry_sleeps == 1 && record.state == CAPSTONE_PARK_IDLE);
  CHECK(capstone_park_queued(&park, 0, 0x3000) == 0);
  CHECK(capstone_park_wake(&park, 0x3000, 1) == 0);
}

/* The other two orders against RETRY: a WAKE that selected the record before
   the sleep ended as RETRY wins (WOKEN); a REQUEUE that moved it and then
   RETRY leaves it in neither queue. The sleep does the waker's part itself,
   so the order is fixed. */
static int wake_first;
static long waking_then_retrying(void *context, _Atomic uint32_t *word,
                                 const struct timespec *deadline) {
  (void)word; (void)deadline;
  if (wake_first)
    CHECK(capstone_park_wake(&park, 0x3000, 1) == 1);
  else
    CHECK(capstone_park_requeue(&park, 0x3000, (uint64_t)(uintptr_t)context, 0, 1) == 1);
  return CAPSTONE_PARK_SLEEP_RETRY;
}

static void retry_after(int wake) {
  setup(1);
  wake_first = wake;
  struct capstone_park_record record = {0};
  enum capstone_park_outcome o = capstone_park_wait_with(&park, &record, 0x3000, 0, NULL,
                                                         waking_then_retrying, (void *)0x4000);
  CHECK(o == (wake ? CAPSTONE_PARK_WOKEN : CAPSTONE_PARK_RETRY));
  CHECK(record.state == CAPSTONE_PARK_IDLE);
  CHECK(capstone_park_queued(&park, 0, 0x3000) == 0 && capstone_park_queued(&park, 0, 0x4000) == 0);
}

int main(int argc, char **argv) {
  CHECK(argc == 2);
  const char *c = argv[1];
  if (!strcmp(c, "b1")) b1();
  else if (!strcmp(c, "b2")) b2();
  else if (!strcmp(c, "b3-timeout-select")) b3(0, 1);
  else if (!strcmp(c, "b3-timeout-abort")) b3(0, 0);
  else if (!strcmp(c, "b3-eintr-select")) b3(1, 1);
  else if (!strcmp(c, "b3-eintr-abort")) b3(1, 0);
  else if (!strcmp(c, "b4")) b4();
  else if (!strcmp(c, "b5")) b5();
  else if (!strcmp(c, "b12")) b12();
  else if (!strcmp(c, "b13a")) b13('a');
  else if (!strcmp(c, "b13b")) b13('b');
  else if (!strcmp(c, "b13c")) b13('c');
  else if (!strcmp(c, "b13d")) b13('d');
  else if (!strcmp(c, "retry")) retry();
  else if (!strcmp(c, "retry-woken")) retry_after(1);
  else if (!strcmp(c, "retry-requeued")) retry_after(0);
  else {
    fprintf(stderr, "park-test: unknown case %s\n", c);
    return 2;
  }
  printf("park-test %s: PASS\n", c);
  return 0;
}
