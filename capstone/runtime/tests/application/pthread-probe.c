/* Probe B, domain phase, T4: musl's threads on minted contexts
 * (docs/plans/delegation-threads.md, Q4, B10, B11).
 *
 * pthread_create, join, detach and exit are musl's own; the runtime supplies
 * __clone, a thread's end and its reaping. The modes cover joins with values,
 * a joiner parked while the child is between its announcement and its last
 * store (B10), detached threads made and reaped without bound (B11), a
 * reservation that waits for a finished thread's transport, the main thread
 * leaving before another, exit() from a thread, thread-specific data, and a
 * condition variable between two threads.
 *
 * Every mode prints "pthread-probe <mode>: PASS" and exits 0, or fails the
 * CHECK naming the broken property; exit-thread exits 5 from a thread. */
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <capstone/context.h>
#include <capstone/delegate.h>

#define CHECK(test) do { if (!(test)) { \
  fprintf(stderr, "pthread-probe:%d: %s\n", __LINE__, #test); return 1; \
} } while (0)

/* The image declares CONTEXTS 7: seven threads besides the first at once. */
#define THREADS 7

extern void (*__capstone_thread_exit_test_gap)(void);
void __capstone_clone_stats(unsigned *made, unsigned *live);

static long monotonic_ms(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000 + t.tv_nsec / 1000000;
}

static void sleep_ms(long ms)
{
  struct timespec t = {ms / 1000, (ms % 1000) * 1000000};
  while (nanosleep(&t, &t) && errno == EINTR);
}

#define F_WAIT 0
#define F_WAKE 1
#define F_PRIVATE 128
static void fwait(volatile int *w, int val)
{
  syscall(SYS_futex, w, F_WAIT | F_PRIVATE, val, 0);
}
static void fwake(volatile int *w) { syscall(SYS_futex, w, F_WAKE | F_PRIVATE, 0x7fffffff); }

/* ---- join-values: 60 rounds of seven threads, each joined with its value.
   The rounds reuse areas, transports and mappings; a thread made right after
   a join may find its transport still ending, and the reservation waits. */
struct job {
  long in, out;
  int tid;
};

/* all seven threads and the main thread meet here each round, so seven
   contexts are alive at once */
static pthread_barrier_t met;

static void *triple(void *arg)
{
  struct job *j = arg;
  j->tid = (int)syscall(SYS_gettid);
  pthread_barrier_wait(&met);
  return (void *)(intptr_t)(j->in * 3 + 1);
}

static int join_values(void)
{
  CHECK(pthread_barrier_init(&met, 0, THREADS + 1) == 0);
  for (int round = 0; round < 60; ++round) {
    pthread_t t[THREADS];
    struct job jobs[THREADS];
    for (int i = 0; i < THREADS; ++i) {
      jobs[i].in = round * 10 + i;
      int e = pthread_create(&t[i], 0, triple, &jobs[i]);
      if (e) {
        fprintf(stderr, "round %d thread %d: %s\n", round, i, strerror(e));
        return 1;
      }
    }
    pthread_barrier_wait(&met);
    for (int i = 0; i < THREADS; ++i) {
      void *v;
      CHECK(pthread_join(t[i], &v) == 0);
      CHECK((long)(intptr_t)v == jobs[i].in * 3 + 1);
      /* a minted context's identity: above every pid, below musl's lock bits */
      CHECK(jobs[i].tid >= 0x400000 && jobs[i].tid <= 0x3ffffffe);
      for (int k = 0; k < i; ++k)
        CHECK(jobs[k].tid != jobs[i].tid);
    }
  }
  unsigned made, live;
  __capstone_clone_stats(&made, &live);
  printf("join-values: 420 threads, records made %u, live %u\n", made, live);
  /* seven at once; areas are reused, not taken anew */
  CHECK(made >= THREADS && made <= 2 * THREADS);
  return 0;
}

/* ---- join-parked (B10): the child holds in its end, after CONTEXT_EXITING
   and before it releases musl's thread-list lock, for 40 ms. Its joiner has
   seen detach_state and waits on that lock, which only the launcher's wake
   after the child's last return ends. */
static volatile int gap_mode, gap_reached;
static void exit_gap(void)
{
  if (!gap_mode)
    return;
  gap_reached = 1;
  fwake(&gap_reached);
  sleep_ms(40);
}

static void *returns_seven(void *arg) { (void)arg; return (void *)7; }

static int join_parked(void)
{
  __capstone_thread_exit_test_gap = exit_gap;
  for (int i = 0; i < 5; ++i) {
    pthread_t t;
    void *v;
    gap_mode = 1;
    gap_reached = 0;
    CHECK(pthread_create(&t, 0, returns_seven, 0) == 0);
    long t0 = monotonic_ms();
    CHECK(pthread_join(t, &v) == 0);
    long waited = monotonic_ms() - t0;
    CHECK(v == (void *)7 && gap_reached);
    /* the join could not return before the child's end: it waited the gap */
    CHECK(waited >= 40);
    printf("join-parked: join returned after %ld ms\n", waited);
  }
  gap_mode = 0;
  return 0;
}

/* ---- detach-many (B11): 300 detached threads, at most six at once. Each
   one's mapping goes back to the heap and its area to the next thread when it
   is reaped; without that the 32 mappings or the arena would run out. */
static volatile int in_flight, finished;
static void *detached(void *arg)
{
  (void)arg;
  __atomic_fetch_add(&finished, 1, __ATOMIC_RELAXED);
  __atomic_fetch_sub(&in_flight, 1, __ATOMIC_RELEASE);
  fwake(&in_flight);
  return 0;
}

static int detach_many(void)
{
  pthread_attr_t a;
  pthread_attr_init(&a);
  pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
  for (int i = 0; i < 300; ++i) {
    int n;
    while ((n = __atomic_load_n(&in_flight, __ATOMIC_ACQUIRE)) >= THREADS - 1)
      fwait(&in_flight, n);
    __atomic_fetch_add(&in_flight, 1, __ATOMIC_RELAXED);
    pthread_t t;
    int e = pthread_create(&t, &a, detached, 0);
    if (e) {
      fprintf(stderr, "detached thread %d: %s\n", i, strerror(e));
      return 1;
    }
  }
  int n;
  while ((n = __atomic_load_n(&in_flight, __ATOMIC_ACQUIRE)))
    fwait(&in_flight, n);
  CHECK(finished == 300);
  /* one more thread reaps what finished since the last one */
  sleep_ms(20);
  pthread_t t;
  CHECK(pthread_create(&t, 0, returns_seven, 0) == 0);
  CHECK(pthread_join(t, 0) == 0);
  unsigned made, live;
  __capstone_clone_stats(&made, &live);
  printf("detach-many: 300 detached, records made %u, live %u\n", made, live);
  CHECK(made <= 2 * THREADS && live <= 1);
  return 0;
}

/* ---- reserve-waits: six threads and one context hold every transport. The
   context (the runtime's own interface, not musl's) ends with SYS_exit and
   holds in its end after announcing it, with no lock held. A thread made then
   finds no transport free and one ending: the reservation waits for it rather
   than answer EAGAIN. (A musl thread cannot show this: it holds musl's
   thread-list lock until its clear, and pthread_create waits on that lock.) */
static volatile int hold = 1;
static void *holder(void *arg)
{
  (void)arg;
  while (__atomic_load_n(&hold, __ATOMIC_ACQUIRE))
    fwait(&hold, 1);
  return 0;
}

static unsigned long raw_exit(void *arg)
{
  (void)arg;
  syscall(SYS_exit, 0);
  return 1;
}

static int reserve_waits(void)
{
  pthread_t t[THREADS], extra;
  struct capstone_context c;
  __capstone_thread_exit_test_gap = exit_gap;
  gap_mode = 0;
  for (int i = 1; i < THREADS; ++i)
    CHECK(pthread_create(&t[i], 0, holder, 0) == 0);
  CHECK(capstone_context_mint(&c, 32768, raw_exit, 0) == 0);
  gap_reached = 0;
  gap_mode = 1;
  CHECK(capstone_context_create(&c, CAPSTONE_CONTEXT_THREAD) >= 0);
  while (!__atomic_load_n(&gap_reached, __ATOMIC_ACQUIRE))
    fwait(&gap_reached, 0);
  gap_mode = 0;
  long t0 = monotonic_ms();
  int e = pthread_create(&extra, 0, returns_seven, 0);
  long waited = monotonic_ms() - t0;
  printf("reserve-waits: create answered %d after %ld ms\n", e, waited);
  CHECK(e == 0 && waited >= 20);
  CHECK(*c.done == 1);
  capstone_context_revoke(&c);
  __atomic_store_n(&hold, 0, __ATOMIC_RELEASE);
  fwake(&hold);
  for (int i = 1; i < THREADS; ++i)
    CHECK(pthread_join(t[i], 0) == 0);
  CHECK(pthread_join(extra, 0) == 0);
  /* and an eighth thread at once is refused, not waited for */
  hold = 1;
  for (int i = 0; i < THREADS; ++i)
    CHECK(pthread_create(&t[i], 0, holder, 0) == 0);
  CHECK(pthread_create(&extra, 0, returns_seven, 0) == EAGAIN);
  __atomic_store_n(&hold, 0, __ATOMIC_RELEASE);
  fwake(&hold);
  for (int i = 0; i < THREADS; ++i)
    CHECK(pthread_join(t[i], 0) == 0);
  return 0;
}

/* ---- main-exit: the main thread leaves with pthread_exit while a thread
   still runs; the application goes on, and ends with status 0 when that
   thread, the last, returns (musl then calls exit(0) for it). */
static void *outlives(void *arg)
{
  (void)arg;
  sleep_ms(50);
  printf("pthread-probe main-exit: PASS\n");
  return 0;
}

static int main_exit(void)
{
  pthread_t t;
  CHECK(pthread_create(&t, 0, outlives, 0) == 0);
  pthread_exit(0);
}

/* ---- exit-thread: exit() from a thread ends the application with its status. */
static void *exits_five(void *arg)
{
  (void)arg;
  exit(5);
}

static int exit_thread(void)
{
  pthread_t t;
  CHECK(pthread_create(&t, 0, exits_five, 0) == 0);
  pthread_join(t, 0);
  printf("REACHED\n");
  return 1;
}

/* ---- tsd: each thread's value under one key, and the destructor once per
   thread that set one. */
static pthread_key_t key;
static volatile int destroyed;
static void destructor(void *v)
{
  (void)v;
  __atomic_fetch_add(&destroyed, 1, __ATOMIC_RELAXED);
}

static void *keeps(void *arg)
{
  pthread_setspecific(key, arg);
  sleep_ms(5);
  return pthread_getspecific(key);
}

static int tsd(void)
{
  pthread_t t[THREADS];
  CHECK(pthread_key_create(&key, destructor) == 0);
  CHECK(pthread_setspecific(key, (void *)100) == 0);
  for (int i = 0; i < THREADS; ++i)
    CHECK(pthread_create(&t[i], 0, keeps, (void *)(intptr_t)(i + 1)) == 0);
  for (int i = 0; i < THREADS; ++i) {
    void *v;
    CHECK(pthread_join(t[i], &v) == 0 && v == (void *)(intptr_t)(i + 1));
  }
  CHECK(destroyed == THREADS && pthread_getspecific(key) == (void *)100);
  CHECK(pthread_key_delete(key) == 0);
  return 0;
}

/* ---- cond: two threads take turns 2000 times under one mutex. */
static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t turn_changed = PTHREAD_COND_INITIALIZER;
static int turn, count;

static void *player(void *arg)
{
  int me = (int)(intptr_t)arg;
  for (int i = 0; i < 1000; ++i) {
    pthread_mutex_lock(&m);
    while (turn != me)
      pthread_cond_wait(&turn_changed, &m);
    ++count;
    turn = !me;
    pthread_cond_broadcast(&turn_changed);
    pthread_mutex_unlock(&m);
  }
  return 0;
}

static int cond(void)
{
  pthread_t a, b;
  CHECK(pthread_create(&a, 0, player, (void *)0) == 0);
  CHECK(pthread_create(&b, 0, player, (void *)1) == 0);
  CHECK(pthread_join(a, 0) == 0 && pthread_join(b, 0) == 0);
  CHECK(count == 2000);
  return 0;
}

/* ---- pi-mutex: PTHREAD_PRIO_INHERIT mutexes (FUTEX_LOCK_PI and UNLOCK_PI):
   an exact count from four threads, a timed lock that times out against its
   absolute realtime deadline, and EDEADLK for an error-checking relock. */
static pthread_mutex_t pim;
static long pi_count;

static void *pi_adder(void *arg)
{
  (void)arg;
  for (int i = 0; i < 500; ++i) {
    pthread_mutex_lock(&pim);
    long c = pi_count;
    if (!(i % 50))
      sched_yield();
    pi_count = c + 1;
    pthread_mutex_unlock(&pim);
  }
  return 0;
}

static void *pi_holder(void *arg)
{
  (void)arg;
  pthread_mutex_lock(&pim);
  gap_reached = 1;
  fwake(&gap_reached);
  sleep_ms(80);
  pthread_mutex_unlock(&pim);
  return 0;
}

static int pi_mutex(void)
{
  pthread_mutexattr_t a;
  pthread_t t[4];
  CHECK(pthread_mutexattr_init(&a) == 0);
  CHECK(pthread_mutexattr_setprotocol(&a, PTHREAD_PRIO_INHERIT) == 0);
  CHECK(pthread_mutexattr_settype(&a, PTHREAD_MUTEX_ERRORCHECK) == 0);
  CHECK(pthread_mutex_init(&pim, &a) == 0);
  for (int i = 0; i < 4; ++i)
    CHECK(pthread_create(&t[i], 0, pi_adder, 0) == 0);
  for (int i = 0; i < 4; ++i)
    CHECK(pthread_join(t[i], 0) == 0);
  CHECK(pi_count == 2000);
  CHECK(pthread_mutex_lock(&pim) == 0);
  CHECK(pthread_mutex_lock(&pim) == EDEADLK);
  CHECK(pthread_mutex_unlock(&pim) == 0);
  CHECK(pthread_mutex_unlock(&pim) == EPERM);
  gap_reached = 0;
  CHECK(pthread_create(&t[0], 0, pi_holder, 0) == 0);
  while (!__atomic_load_n(&gap_reached, __ATOMIC_ACQUIRE))
    fwait(&gap_reached, 0);
  struct timespec at;
  clock_gettime(CLOCK_REALTIME, &at);
  at.tv_nsec += 30 * 1000000;
  if (at.tv_nsec >= 1000000000) {
    at.tv_nsec -= 1000000000;
    ++at.tv_sec;
  }
  long t0 = monotonic_ms();
  int e = pthread_mutex_timedlock(&pim, &at);
  long waited = monotonic_ms() - t0;
  printf("pi-mutex: timed lock answered %d after %ld ms\n", e, waited);
  CHECK(e == ETIMEDOUT && waited >= 25);
  /* a deadline at the end of time waits for the holder's unlock */
  struct timespec never = {.tv_sec = (time_t)0x7fffffffffffffffLL, .tv_nsec = 999999999};
  t0 = monotonic_ms();
  e = pthread_mutex_timedlock(&pim, &never);
  printf("pi-mutex: far deadline answered %d after %ld ms\n", e, monotonic_ms() - t0);
  CHECK(e == 0);
  CHECK(pthread_mutex_unlock(&pim) == 0);
  CHECK(pthread_join(t[0], 0) == 0);
  CHECK(pthread_mutex_destroy(&pim) == 0);
  return 0;
}

/* ---- user-stack: a thread on a stack the application gives it
   (pthread_attr_setstack), and the stack it is told it has
   (pthread_getattr_np, pthread_attr_getstack) is memory it can read. */
static char *given;
static void *on_given(void *arg)
{
  (void)arg;
  volatile char local = 1;
  pthread_attr_t a;
  void *base = 0;
  size_t size = 0;
  if (pthread_getattr_np(pthread_self(), &a) || pthread_attr_getstack(&a, &base, &size))
    return (void *)1;
  if ((char *)&local < given || (char *)&local >= given + 65536)
    return (void *)2;   /* not on the stack it was given */
  if ((char *)base < given || (char *)base + size > given + 65536)
    return (void *)3;
  if (*(volatile char *)base != 0x5a)
    return (void *)4;   /* a pointer it cannot read through */
  pthread_attr_destroy(&a);
  return (void *)(intptr_t)local;
}

static int user_stack(void)
{
  pthread_attr_t a;
  pthread_t t;
  void *v;
  char *raw = malloc(65536 + 16);
  CHECK(raw);
  given = raw + (-(uintptr_t)raw & 15);
  memset(given, 0x5a, 65536);
  CHECK(pthread_attr_init(&a) == 0);
  CHECK(pthread_attr_setstack(&a, given, 65536) == 0);
  void *back = 0;
  size_t size = 0;
  CHECK(pthread_attr_getstack(&a, &back, &size) == 0 && back == given && size == 65536);
  CHECK(pthread_create(&t, &a, on_given, 0) == 0);
  CHECK(pthread_join(t, &v) == 0);
  printf("user-stack: thread answered %ld\n", (long)(intptr_t)v);
  CHECK(v == (void *)(intptr_t)1);
  return 0;
}

/* ---- main-exit-more: after the main thread left, the remaining thread makes
   three more, which use the heap at once; the application then ends 0. */
static void *churn(void *arg)
{
  (void)arg;
  for (int i = 0; i < 2000; ++i) {
    char *p = malloc(64 + i % 256);
    if (!p)
      return (void *)1;
    memset(p, i, 64);
    free(p);
  }
  return 0;
}

static void *spawner(void *arg)
{
  (void)arg;
  pthread_t t[3];
  sleep_ms(30);
  for (int i = 0; i < 3; ++i)
    if (pthread_create(&t[i], 0, churn, 0))
      return 0;
  for (int i = 0; i < 3; ++i) {
    void *v;
    if (pthread_join(t[i], &v) || v)
      return 0;
  }
  printf("pthread-probe main-exit-more: PASS\n");
  return 0;
}

static int main_exit_more(void)
{
  pthread_t t;
  CHECK(pthread_create(&t, 0, spawner, 0) == 0);
  pthread_exit(0);
}

/* ---- clone-refused: clone() for anything but a thread is not a context. */
static int child_fn(void *arg) { (void)arg; return 0; }
static int clone_refused(void)
{
  static char stack[4096] __attribute__((aligned(16)));
  errno = 0;
  CHECK(clone(child_fn, stack + sizeof stack, SIGCHLD, 0) == -1 && errno == ENOSYS);
  return 0;
}

int main(int argc, char **argv)
{
  int rc;
  if (argc < 2) {
    fprintf(stderr, "usage: pthread-probe MODE\n");
    return 2;
  }
  const char *mode = argv[1];
  if (!strcmp(mode, "join-values")) rc = join_values();
  else if (!strcmp(mode, "join-parked")) rc = join_parked();
  else if (!strcmp(mode, "detach-many")) rc = detach_many();
  else if (!strcmp(mode, "reserve-waits")) rc = reserve_waits();
  else if (!strcmp(mode, "main-exit")) rc = main_exit();
  else if (!strcmp(mode, "exit-thread")) rc = exit_thread();
  else if (!strcmp(mode, "tsd")) rc = tsd();
  else if (!strcmp(mode, "cond")) rc = cond();
  else if (!strcmp(mode, "clone-refused")) rc = clone_refused();
  else if (!strcmp(mode, "pi-mutex")) rc = pi_mutex();
  else if (!strcmp(mode, "user-stack")) rc = user_stack();
  else if (!strcmp(mode, "main-exit-more")) rc = main_exit_more();
  else {
    fprintf(stderr, "pthread-probe: unknown mode %s\n", mode);
    return 2;
  }
  if (!rc)
    printf("pthread-probe %s: PASS\n", mode);
  return rc;
}
