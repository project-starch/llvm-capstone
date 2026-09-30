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
#include <semaphore.h>
#include <sys/syscall.h>
#include <ucontext.h>
#include <capstone/context.h>
#include <capstone/delegate.h>

#define CHECK(test) do { if (!(test)) { \
  fprintf(stderr, "pthread-probe:%d: %s\n", __LINE__, #test); return 1; \
} } while (0)

/* The image declares CONTEXTS 15, the most (T5): fifteen threads besides the
   first at once. */
#define THREADS 15

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
  printf("join-values: %d threads, records made %u, live %u\n", 60 * THREADS, made, live);
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

/* ==== B8: signals per thread. ==== */
static volatile int handler_tid, handler_runs, handler_calls_ok;
static void on_signal(int sig)
{
  (void)sig;
  handler_tid = (int)syscall(SYS_gettid);
  ++handler_runs;
}

static int install(int sig, void (*fn)(int), int flags)
{
  struct sigaction sa = {0};
  sa.sa_handler = fn;
  sa.sa_flags = flags;
  return sigaction(sig, &sa, 0);
}

/* Pipes shared by the cancellation and handler-call modes below. The directed
   kill-thread modes live in pthread-kill-probe.c, also exercised natively. */
static int pipefd[2];
extern int capstone_probe_kill_thread(int early);

/* ---- mask-routing: a signal sent to the process runs in the one thread that
   does not block it. */
static volatile int routed_ready;
static void *unblocks_usr2(void *arg)
{
  (void)arg;
  sigset_t set;
  sigemptyset(&set);
  sigaddset(&set, SIGUSR2);
  pthread_sigmask(SIG_UNBLOCK, &set, 0);
  reader_tid = (int)syscall(SYS_gettid);
  routed_ready = 1;
  long t0 = monotonic_ms();
  while (!handler_runs && monotonic_ms() - t0 < 5000)
    sched_yield();   /* a round each time: where the handler runs */
  return 0;
}

static int mask_routing(void)
{
  pthread_t t;
  sigset_t set;
  sigemptyset(&set);
  sigaddset(&set, SIGUSR2);
  CHECK(install(SIGUSR2, on_signal, 0) == 0);
  CHECK(pthread_sigmask(SIG_BLOCK, &set, 0) == 0);
  CHECK(pthread_create(&t, 0, unblocks_usr2, 0) == 0);
  while (!routed_ready)
    sched_yield();
  CHECK(kill(getpid(), SIGUSR2) == 0);
  CHECK(pthread_join(t, 0) == 0);
  printf("mask-routing: handler in %d, the unblocking thread %d\n", handler_tid, reader_tid);
  CHECK(handler_runs == 1 && handler_tid == reader_tid);
  return 0;
}

/* ---- raise-thread: raise in a thread runs the handler there before it returns. */
static volatile int raised_in, ran_before_return;
static void *raises(void *arg)
{
  (void)arg;
  raised_in = (int)syscall(SYS_gettid);
  raise(SIGUSR1);
  ran_before_return = handler_runs == 1;
  return 0;
}

static int raise_thread(void)
{
  pthread_t t;
  CHECK(install(SIGUSR1, on_signal, 0) == 0);
  CHECK(pthread_create(&t, 0, raises, 0) == 0 && pthread_join(t, 0) == 0);
  CHECK(ran_before_return && handler_tid == raised_in);
  return 0;
}

/* ---- sigwait-thread: a thread that blocks SIGUSR1 takes it with sigwait. */
static volatile int waited_sig = -1;
static void *waits_for_usr1(void *arg)
{
  (void)arg;
  sigset_t set;
  int sig;
  sigemptyset(&set);
  sigaddset(&set, SIGUSR1);
  if (sigwait(&set, &sig) == 0)
    waited_sig = sig;
  return 0;
}

static int sigwait_thread(void)
{
  pthread_t t;
  sigset_t set;
  sigemptyset(&set);
  sigaddset(&set, SIGUSR1);
  CHECK(pthread_sigmask(SIG_BLOCK, &set, 0) == 0);   /* the thread inherits it */
  CHECK(pthread_create(&t, 0, waits_for_usr1, 0) == 0);
  sleep_ms(30);
  CHECK(pthread_kill(t, SIGUSR1) == 0);
  CHECK(pthread_join(t, 0) == 0);
  CHECK(waited_sig == SIGUSR1);
  return 0;
}

/* ---- cancel-sem, cancel-read: pthread_cancel of a thread blocked in a
   cancellation point; its cleanup handler runs and join sees PTHREAD_CANCELED. */
static volatile int cleaned;
static void cleanup(void *arg) { (void)arg; cleaned = 1; }
static sem_t never;
static void *waits_sem(void *arg)
{
  (void)arg;
  pthread_cleanup_push(cleanup, 0);
  sem_wait(&never);
  pthread_cleanup_pop(0);
  return (void *)1;
}

static volatile long forever_rc = -2, forever_errno;
static void *reads_forever(void *arg)
{
  (void)arg;
  char c;
  pthread_cleanup_push(cleanup, 0);
  forever_rc = read(pipefd[0], &c, 1);
  forever_errno = errno;
  pthread_cleanup_pop(0);
  return (void *)1;
}

static int cancel_blocked(void *(*fn)(void *))
{
  pthread_t t;
  void *v;
  CHECK(sem_init(&never, 0, 0) == 0 && pipe(pipefd) == 0);
  CHECK(pthread_create(&t, 0, fn, 0) == 0);
  sleep_ms(40);
  CHECK(pthread_cancel(t) == 0);
  CHECK(pthread_join(t, &v) == 0);
  if (v != PTHREAD_CANCELED)
    fprintf(stderr, "cancel: the thread returned %p, its read %ld (errno %ld)\n", v, forever_rc,
            forever_errno);
  CHECK(v == PTHREAD_CANCELED && cleaned);
  return 0;
}

/* ---- cancel-disabled: a cancellation requested while disabled waits for
   the thread to enable it and reach a cancellation point. */
static volatile int passed_disabled;
static void *disables(void *arg)
{
  (void)arg;
  int old;
  pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &old);
  sleep_ms(80);   /* nanosleep is a cancellation point, but not now */
  passed_disabled = 1;
  pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &old);
  pthread_testcancel();
  return (void *)1;
}

static int cancel_disabled(void)
{
  pthread_t t;
  void *v;
  CHECK(pthread_create(&t, 0, disables, 0) == 0);
  sleep_ms(20);
  CHECK(pthread_cancel(t) == 0);
  CHECK(pthread_join(t, &v) == 0);
  CHECK(v == PTHREAD_CANCELED && passed_disabled);
  return 0;
}

/* ---- abort-thread: abort in a thread ends the application with SIGABRT. */
static void *aborts(void *arg)
{
  (void)arg;
  abort();
}

static int abort_thread(void)
{
  pthread_t t;
  CHECK(pthread_create(&t, 0, aborts, 0) == 0);
  pthread_join(t, 0);
  printf("REACHED\n");
  return 1;
}

/* ---- main-exit-signal: after the main thread left (its signals blocked by
   pthread_exit), a signal sent to the process runs in the thread that is left. */
static volatile int term_seen;
static void on_term(int sig)
{
  (void)sig;
  term_seen = (int)syscall(SYS_gettid);
}

static void *stays(void *arg)
{
  (void)arg;
  sleep_ms(30);
  kill(getpid(), SIGTERM);
  long t0 = monotonic_ms();
  while (!term_seen && monotonic_ms() - t0 < 5000)
    sched_yield();
  if (term_seen == (int)syscall(SYS_gettid))
    printf("pthread-probe main-exit-signal: PASS\n");
  return 0;
}

static int main_exit_signal(void)
{
  pthread_t t;
  CHECK(install(SIGTERM, on_term, 0) == 0);
  CHECK(pthread_create(&t, 0, stays, 0) == 0);
  pthread_exit(0);
}

/* ---- park-signal-thread (B8's continuation): a thread parked in a futex
   wait takes a signal whose handler makes a delegated call; under SA_RESTART
   the wait goes on and ends with the wake, the handler having run once. */
static volatile int parked_word;
static void on_signal_calls(int sig)
{
  (void)sig;
  handler_tid = (int)syscall(SYS_gettid);
  ++handler_runs;
  handler_calls_ok = write(pipefd[1], "h", 1) == 1;
}

static volatile int waiter_tid, wait_rc = -2;
static void *parks(void *arg)
{
  (void)arg;
  waiter_tid = (int)syscall(SYS_gettid);
  while (__atomic_load_n(&parked_word, __ATOMIC_ACQUIRE) == 0)
    wait_rc = (int)syscall(SYS_futex, &parked_word, F_WAIT | F_PRIVATE, 0, 0);
  return 0;
}

static int park_signal_thread(void)
{
  pthread_t t;
  char c;
  CHECK(pipe(pipefd) == 0 && install(SIGUSR1, on_signal_calls, SA_RESTART) == 0);
  CHECK(pthread_create(&t, 0, parks, 0) == 0);
  sleep_ms(40);
  CHECK(pthread_kill(t, SIGUSR1) == 0);
  CHECK(read(pipefd[0], &c, 1) == 1 && c == 'h');   /* the handler's call arrived */
  sleep_ms(20);
  __atomic_store_n(&parked_word, 1, __ATOMIC_RELEASE);
  fwake(&parked_word);
  CHECK(pthread_join(t, 0) == 0);
  printf("park-signal-thread: handler in %d, waiter %d, wait answered %d\n", handler_tid,
         waiter_tid, wait_rc);
  CHECK(handler_runs == 1 && handler_tid == waiter_tid && handler_calls_ok && wait_rc == 0);
  return 0;
}

/* ---- cancel-installed-early: the cancellation handler is installed before
   the first thread exists (musl installs it at the first pthread_cancel); a
   thread made afterwards is cancelled through it. */
static int cancel_installed_early(void)
{
  int old;
  pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &old);   /* for good: its own request stays pending */
  CHECK(pthread_cancel(pthread_self()) == 0);
  return cancel_blocked(waits_sem);
}

/* ---- sigreturn-mask: a handler that runs during sigsuspend and adds
   SIGUSR2 to its uc_sigmask leaves SIGUSR2 blocked after the wait: the mask
   it is shown is the one from before the wait, and the one it leaves is
   the one the wait returns to. */
static void on_usr1_blocks_usr2(int sig, siginfo_t *si, void *context)
{
  (void)sig;
  (void)si;
  ucontext_t *uc = context;
  sigaddset(&uc->uc_sigmask, SIGUSR2);
  ++handler_runs;
}

static int sigreturn_mask(void)
{
  struct sigaction sa = {0};
  sigset_t block, wait_mask, after;
  sa.sa_sigaction = on_usr1_blocks_usr2;
  sa.sa_flags = SA_SIGINFO;
  CHECK(sigaction(SIGUSR1, &sa, 0) == 0);
  sigemptyset(&block);
  sigaddset(&block, SIGUSR1);
  CHECK(sigprocmask(SIG_BLOCK, &block, 0) == 0);
  CHECK(kill(getpid(), SIGUSR1) == 0);
  sigemptyset(&wait_mask);
  sigaddset(&wait_mask, SIGUSR2);   /* the wait blocks SIGUSR2, which is otherwise open */
  sigsuspend(&wait_mask);
  CHECK(sigprocmask(SIG_BLOCK, 0, &after) == 0);
  printf("sigreturn-mask: handler ran %d, SIGUSR2 blocked after the wait %d, SIGUSR1 %d\n",
         handler_runs, sigismember(&after, SIGUSR2), sigismember(&after, SIGUSR1));
  CHECK(handler_runs == 1 && sigismember(&after, SIGUSR2) == 1 && sigismember(&after, SIGUSR1) == 1);
  return 0;
}

/* ---- setuid-threads: setuid answers at once while another thread computes
   without a single call (it answers ENOSYS here, 0 on Linux for one's own
   uid). */
static volatile int spin_until;
static volatile long spun;
static void *computes(void *arg)
{
  (void)arg;
  /* no call at all, not even a clock read: every entry into the dispatcher
     would take a signal */
  while (!__atomic_load_n(&spin_until, __ATOMIC_RELAXED) && spun < 300000000)
    ++spun;
  return 0;
}

static int setuid_threads(void)
{
  pthread_t t;
  CHECK(pthread_create(&t, 0, computes, 0) == 0);
  sleep_ms(20);
  long t0 = monotonic_ms();
  errno = 0;
  int r = setuid(getuid());
  long took = monotonic_ms() - t0;
  int e = errno;
  __atomic_store_n(&spin_until, 1, __ATOMIC_RELAXED);
  CHECK(pthread_join(t, 0) == 0);
  printf("setuid-threads: answered %d (errno %d) after %ld ms, the other thread spun %ld times\n", r, e,
         took, spun);
  CHECK((r == 0 || e == ENOSYS) && took < 1000);
  return 0;
}

/* ---- sigaction-race: one thread takes SIGUSR1 again and again while the
   main thread switches its handler between two, and to SIG_IGN and back:
   every run is one of the two handlers, none is a call through SIG_IGN. */
static volatile int race_a, race_b, race_stop;
static void race_one(int sig) { (void)sig; ++race_a; }
static void race_two(int sig) { (void)sig; ++race_b; }
static void *takes_usr1(void *arg)
{
  (void)arg;
  while (!__atomic_load_n(&race_stop, __ATOMIC_ACQUIRE))
    if (write(pipefd[1], "", 0) < 0)   /* a round each time, where handlers run */
      return (void *)1;
  return 0;
}

static int sigaction_race(void)
{
  pthread_t t;
  sigset_t set;
  CHECK(pipe(pipefd) == 0);
  CHECK(install(SIGUSR1, race_one, SA_RESTART) == 0);
  sigemptyset(&set);
  sigaddset(&set, SIGUSR1);
  CHECK(pthread_create(&t, 0, takes_usr1, 0) == 0);
  CHECK(pthread_sigmask(SIG_BLOCK, &set, 0) == 0);   /* SIGUSR1 goes to the thread */
  for (int i = 0; i < 600; ++i) {
    pthread_kill(t, SIGUSR1);
    install(SIGUSR1, (i % 3) == 0 ? race_one : (i % 3) == 1 ? race_two : SIG_IGN, SA_RESTART);
  }
  __atomic_store_n(&race_stop, 1, __ATOMIC_RELEASE);
  void *v;
  CHECK(pthread_join(t, &v) == 0 && v == 0);
  printf("sigaction-race: %d and %d runs of the two handlers for 600 signals\n", race_a, race_b);
  CHECK(race_a + race_b > 0 && race_a + race_b <= 600);
  return 0;
}

/* ---- clone-refused: clone() for anything but a thread is not a context. */
static int child_fn(void *arg) { (void)arg; return 0; }
/* A thread's name is the name of the Linux thread that serves its context: a
   thread names itself and reads it back, main keeps its own, and naming another
   thread (musl writes /proc/self/task/<tid>/comm; a minted thread's tid, from
   0x400000, names no Linux task) is refused rather than landing on another. */
static pthread_barrier_t named_met;

static void *named(void *arg)
{
  char name[16];
  (void)arg;
  if (pthread_setname_np(pthread_self(), "probe-worker")) return (void *)1;
  pthread_barrier_wait(&named_met);   /* main tries to rename this thread */
  pthread_barrier_wait(&named_met);
  if (pthread_getname_np(pthread_self(), name, sizeof name)) return (void *)2;
  if (strcmp(name, "probe-worker")) return (void *)3;
  return 0;
}

static int thread_name(void)
{
  char before[16], after[16];
  pthread_t t;
  void *r;
  CHECK(pthread_barrier_init(&named_met, 0, 2) == 0);
  CHECK(pthread_getname_np(pthread_self(), before, sizeof before) == 0);
  CHECK(pthread_create(&t, 0, named, 0) == 0);
  pthread_barrier_wait(&named_met);
  int other = pthread_setname_np(t, "renamed");
  pthread_barrier_wait(&named_met);
  CHECK(pthread_join(t, &r) == 0);
  CHECK(r == 0);
  CHECK(other != 0);
  CHECK(pthread_getname_np(pthread_self(), after, sizeof after) == 0);
  CHECK(!strcmp(before, after));
  CHECK(pthread_setname_np(pthread_self(), "probe-main") == 0);
  CHECK(pthread_getname_np(pthread_self(), after, sizeof after) == 0);
  CHECK(!strcmp(after, "probe-main"));
  printf("thread-name: main was '%s'; naming another thread: %s\n", before, strerror(other));
  return 0;
}

/* A thread's CPU set is its Linux thread's: the caller's own is read and set, by
   0 or by pthread_self, in main and in a thread; another thread's is ESRCH. */
static void *affinity_worker(void *arg)
{
  cpu_set_t set;
  (void)arg;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof set, &set) || CPU_COUNT(&set) < 1) return (void *)1;
  CPU_ZERO(&set);
  if (pthread_getaffinity_np(pthread_self(), sizeof set, &set) || CPU_COUNT(&set) < 1) return (void *)2;
  if (pthread_setaffinity_np(pthread_self(), sizeof set, &set)) return (void *)3;
  pthread_barrier_wait(&named_met);
  pthread_barrier_wait(&named_met);
  return 0;
}

static int affinity(void)
{
  cpu_set_t set;
  pthread_t t;
  void *r;
  CPU_ZERO(&set);
  CHECK(sched_getaffinity(0, sizeof set, &set) == 0);
  int cpus = CPU_COUNT(&set);
  CHECK(cpus >= 1);
  CHECK(pthread_barrier_init(&named_met, 0, 2) == 0);
  CHECK(pthread_create(&t, 0, affinity_worker, 0) == 0);
  pthread_barrier_wait(&named_met);
  int other = pthread_getaffinity_np(t, sizeof set, &set);
  pthread_barrier_wait(&named_met);
  CHECK(pthread_join(t, &r) == 0);
  CHECK(r == 0);
  CHECK(other == ESRCH);
  printf("affinity: %d CPUs; another thread's set: %s\n", cpus, strerror(other));
  return 0;
}

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
  else if (!strcmp(mode, "thread-name")) rc = thread_name();
  else if (!strcmp(mode, "affinity")) rc = affinity();
  else if (!strcmp(mode, "pi-mutex")) rc = pi_mutex();
  else if (!strcmp(mode, "user-stack")) rc = user_stack();
  else if (!strcmp(mode, "main-exit-more")) rc = main_exit_more();
  else if (!strcmp(mode, "kill-thread")) rc = capstone_probe_kill_thread(0);
  else if (!strcmp(mode, "kill-thread-early")) rc = capstone_probe_kill_thread(1);
  else if (!strcmp(mode, "mask-routing")) rc = mask_routing();
  else if (!strcmp(mode, "raise-thread")) rc = raise_thread();
  else if (!strcmp(mode, "sigwait-thread")) rc = sigwait_thread();
  else if (!strcmp(mode, "cancel-sem")) rc = cancel_blocked(waits_sem);
  else if (!strcmp(mode, "cancel-read")) rc = cancel_blocked(reads_forever);
  else if (!strcmp(mode, "cancel-disabled")) rc = cancel_disabled();
  else if (!strcmp(mode, "abort-thread")) rc = abort_thread();
  else if (!strcmp(mode, "main-exit-signal")) rc = main_exit_signal();
  else if (!strcmp(mode, "park-signal-thread")) rc = park_signal_thread();
  else if (!strcmp(mode, "cancel-installed-early")) rc = cancel_installed_early();
  else if (!strcmp(mode, "sigreturn-mask")) rc = sigreturn_mask();
  else if (!strcmp(mode, "setuid-threads")) rc = setuid_threads();
  else if (!strcmp(mode, "sigaction-race")) rc = sigaction_race();
  else {
    fprintf(stderr, "pthread-probe: unknown mode %s\n", mode);
    return 2;
  }
  if (!rc)
    printf("pthread-probe %s: PASS\n", mode);
  return rc;
}
