/* Probe B, domain phase, first part: a transport per context
 * (docs/plans/delegation-threads.md).
 *
 * Every context the launcher runs on a thread of its own gets its own entry
 * block and exchange region, reserved before it is minted, so its delegated
 * calls are served by that thread and a call that blocks in Linux blocks only
 * that context (B7). The modes also cover what ends with a context (a fault or
 * exit() in it ends the process), the reservation's lifetime, the reuse of
 * transports, contexts at once, and the calls a further context may not make
 * yet (signal state stays with the first context).
 *
 * Every mode prints "thread-probe <mode>: PASS" and exits 0, or fails the
 * CHECK naming the broken property. exit-child exits 7 from the child, and
 * fault-child must end in a domain fault. */
#define _GNU_SOURCE   /* syscall(), for futex as musl issues it */
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <capstone/capability.h>
#include <capstone/context.h>
#include <capstone/delegate.h>

#define CHECK(test) do { if (!(test)) { \
  fprintf(stderr, "thread-probe:%d: %s\n", __LINE__, #test); return 1; \
} } while (0)

/* The image declares CONTEXTS 7; each thread area holds a TLS block, a
   start block, a seal region and the stack. */
#define TRANSPORTS 7
#define AREA_BYTES (128 * 1024)

long __capstone_delegate_ints(uint64_t nr, uint64_t a, uint64_t b, uint64_t c);
uint64_t __capstone_park_generation(volatile int *word);
extern void (*__capstone_futex_test_gap)(void);

/* futex as musl issues it (FUTEX_PRIVATE_FLAG set). */
#define F_WAIT 0
#define F_WAKE 1
#define F_REQUEUE 3
#define F_WAKE_OP 5
#define F_PRIVATE 128
static long fwait(volatile int *w, int val, const struct timespec *to)
{
  return syscall(SYS_futex, w, F_WAIT | F_PRIVATE, val, to);
}
static long fwake(volatile int *w, int n) { return syscall(SYS_futex, w, F_WAKE | F_PRIVATE, n); }
static long frequeue(volatile int *a, int nwake, long nmove, volatile int *b)
{
  return syscall(SYS_futex, a, F_REQUEUE | F_PRIVATE, nwake, nmove, b);
}

static const char *mode;

static long monotonic_ms(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

/* Until the context has published done, at most `ms`. The waiting context's
   own thread yields in Linux, so the child's thread gets the hart. */
static int wait_done(struct capstone_context *c, long ms)
{
  long t0 = monotonic_ms();
  while (!*c->done && monotonic_ms() - t0 < ms)
    sched_yield();
  return *c->done == 1;
}

/* Mint and create a THREAD context; a transport may still be winding down
   from a context that just ended, so EAGAIN is retried for a while. */
static long start_thread(struct capstone_context *c, unsigned long (*fn)(void *), void *arg)
{
  if (capstone_context_mint(c, AREA_BYTES, fn, arg))
    return -ENOMEM;
  long t0 = monotonic_ms(), id;
  while ((id = capstone_context_create(c, CAPSTONE_CONTEXT_THREAD)) == -EAGAIN &&
         monotonic_ms() - t0 < 5000) {
    sched_yield();
    if (capstone_cap_type(&c->seal) != CAPSTONE_CAP_EMPTY)
      continue;
    return -EIO;   /* the seal was offered and is gone: cannot retry */
  }
  return id;
}

/* The child's side of `transport`: delegated calls through its own
   transport, a local answer, and a result that needs all of them. */
static char child_line[64];
static volatile long child_pid, child_written, child_read;
static unsigned long child_calls(void *arg)
{
  int fds[2];
  char buf[32] = {0};
  child_pid = getpid();
  if (pipe(fds)) return 100 + errno;
  child_written = write(fds[1], child_line, strlen(child_line));
  child_read = read(fds[0], buf, sizeof buf - 1);
  close(fds[0]);
  close(fds[1]);
  if (strcmp(buf, child_line)) return 2;
  printf("thread-probe child: %s", buf);
  fflush(stdout);
  return 42 + (unsigned long)(uintptr_t)arg;
}

static int transport(void)
{
  struct capstone_context c;
  strcpy(child_line, "through its own transport\n");
  long id = start_thread(&c, child_calls, (void *)(uintptr_t)1);
  CHECK(id > 0);
  /* the first context keeps making calls meanwhile */
  for (int i = 0; i < 20 && !*c.done; ++i)
    CHECK(write(1, "", 0) == 0);
  CHECK(wait_done(&c, 20000));
  CHECK(*c.value == 43);
  CHECK(child_pid == getpid());
  CHECK(child_written == (long)strlen(child_line) && child_read == child_written);
  capstone_context_revoke(&c);
  return 0;
}

/* B7: context 1 in a delegated read on an empty pipe; context 2 (the first)
   runs and writes; the read returns the data. The child says when it is about
   to read; the first context then computes for several quanta and checks the
   child has not come back before the write. */
static int pipe_fds[2];
static volatile int b7_reading;
static volatile long b7_got;
static char b7_buf[32];
static unsigned long b7_reader(void *arg)
{
  (void)arg;
  b7_reading = 1;
  b7_got = read(pipe_fds[0], b7_buf, sizeof b7_buf - 1);
  return 7;
}

static volatile unsigned long spin_sink;
static void spin_ms(long ms)
{
  long t0 = monotonic_ms();
  unsigned long x = 1;
  while (monotonic_ms() - t0 < ms)
    for (int i = 0; i < 100000; ++i)
      x = x * 6364136223846793005u + 1442695040888963407u;
  spin_sink = x;
}

static int blocking(void)
{
  struct capstone_context c;
  CHECK(!pipe(pipe_fds));
  long id = start_thread(&c, b7_reader, 0);
  CHECK(id > 0);
  long t0 = monotonic_ms();
  while (!b7_reading && monotonic_ms() - t0 < 20000)
    sched_yield();
  CHECK(b7_reading);
  /* the child is in read (or about to be); this context keeps the hart */
  spin_ms(100);
  CHECK(!*c.done && b7_got == 0);
  CHECK(write(pipe_fds[1], "b7 data", 7) == 7);
  CHECK(wait_done(&c, 20000));
  CHECK(*c.value == 7 && b7_got == 7 && !memcmp(b7_buf, "b7 data", 7));
  capstone_context_revoke(&c);
  close(pipe_fds[0]);
  close(pipe_fds[1]);
  return 0;
}

/* exit() in a further context ends the process with its status. */
static unsigned long exiter(void *arg)
{
  (void)arg;
  exit(7);
}

static int exit_child(void)
{
  struct capstone_context c;
  long id = start_thread(&c, exiter, 0);
  CHECK(id > 0);
  long t0 = monotonic_ms();
  while (monotonic_ms() - t0 < 20000)
    sched_yield();
  fprintf(stderr, "thread-probe exit-child: REACHED\n");
  return 1;
}

/* A fault in a further context ends the process with a domain fault. */
void probe_fault_store(volatile int *where);
static unsigned long faulter(void *arg)
{
  probe_fault_store((volatile int *)arg);
  return 0;
}

static int fault_child(void)
{
  struct capstone_context c;
  long id = start_thread(&c, faulter, 0);
  CHECK(id > 0);
  long t0 = monotonic_ms();
  while (monotonic_ms() - t0 < 20000)
    sched_yield();
  fprintf(stderr, "thread-probe fault-child: REACHED\n");
  return 1;
}

/* Every transport can be reserved once; CREATE gives a reservation back
   whatever its outcome, here a request with no offer. */
static int reserve(void)
{
  long got[TRANSPORTS];
  for (int i = 0; i < TRANSPORTS; ++i) {
    got[i] = __capstone_delegate_ints(CAPSTONE_NR_CONTEXT_RESERVE, 0, 0, 0);
    CHECK(got[i] >= 1 && got[i] <= TRANSPORTS);
    for (int j = 0; j < i; ++j)
      CHECK(got[j] != got[i]);
  }
  CHECK(__capstone_delegate_ints(CAPSTONE_NR_CONTEXT_RESERVE, 0, 0, 0) == -EAGAIN);
  for (int i = 0; i < TRANSPORTS; ++i)
    CHECK(__capstone_delegate_ints(CAPSTONE_NR_CONTEXT_CREATE, 900 + i, CAPSTONE_CONTEXT_THREAD,
                                   (uint64_t)got[i]) == -ENOENT);
  /* a transport that is not reserved, a REGISTER request naming one */
  CHECK(__capstone_delegate_ints(CAPSTONE_NR_CONTEXT_CREATE, 950, CAPSTONE_CONTEXT_THREAD, 1) ==
        -EINVAL);
  long again = __capstone_delegate_ints(CAPSTONE_NR_CONTEXT_RESERVE, 0, 0, 0);
  CHECK(again >= 1);
  CHECK(__capstone_delegate_ints(CAPSTONE_NR_CONTEXT_CREATE, 951, CAPSTONE_CONTEXT_REGISTER,
                                 (uint64_t)again) == -EINVAL);
  for (int i = 0; i < TRANSPORTS; ++i)
    CHECK(__capstone_delegate_ints(CAPSTONE_NR_CONTEXT_RESERVE, 0, 0, 0) >= 1);
  CHECK(__capstone_delegate_ints(CAPSTONE_NR_CONTEXT_RESERVE, 0, 0, 0) == -EAGAIN);
  return 0;
}

/* Four times as many sequential contexts as there are transports, in one
   area: every transport, registration and slot is given back and reused. */
static int reuse(void)
{
  struct capstone_context c;
  for (int round = 0; round < 4 * TRANSPORTS; ++round) {
    snprintf(child_line, sizeof child_line, "reuse round %d\n", round);
    long id;
    if (round == 0) {
      id = start_thread(&c, child_calls, (void *)(uintptr_t)round);
    } else {
      CHECK(!capstone_context_remint(&c, child_calls, (void *)(uintptr_t)round));
      long t0 = monotonic_ms();
      while ((id = capstone_context_create(&c, CAPSTONE_CONTEXT_THREAD)) == -EAGAIN &&
             capstone_cap_type(&c.seal) != CAPSTONE_CAP_EMPTY && monotonic_ms() - t0 < 5000)
        sched_yield();
    }
    CHECK(id > 0);
    CHECK(wait_done(&c, 20000));
    CHECK(*c.value == 42 + (unsigned long)round);
    capstone_context_revoke(&c);
  }
  return 0;
}

/* Every transport at once: each child writes its own pattern through a pipe
   of its own and reads it back, many times; the first context does the same.
   Transports that shared anything would mix the patterns. */
#define ROUNDS 40
static volatile int loop_errors[TRANSPORTS + 1];
static int pattern_loop(int who)
{
  int fds[2];
  char out[48], in[48];
  if (pipe(fds)) return 1;
  for (int r = 0; r < ROUNDS; ++r) {
    int n = snprintf(out, sizeof out, "context %d round %d %08x", who, r,
                     (unsigned)(who * 2654435761u + (unsigned)r));
    memset(in, 0, sizeof in);
    if (write(fds[1], out, (size_t)n) != n || read(fds[0], in, sizeof in) != n ||
        memcmp(in, out, (size_t)n)) {
      ++loop_errors[who];
    }
  }
  close(fds[0]);
  close(fds[1]);
  return loop_errors[who];
}

static unsigned long looper(void *arg)
{
  return pattern_loop((int)(uintptr_t)arg) ? 1 : 100 + (unsigned long)(uintptr_t)arg;
}

static int concurrent(void)
{
  static struct capstone_context c[TRANSPORTS];
  for (int i = 0; i < TRANSPORTS; ++i)
    CHECK(start_thread(&c[i], looper, (void *)(uintptr_t)(i + 1)) > 0);
  CHECK(pattern_loop(0) == 0);
  for (int i = 0; i < TRANSPORTS; ++i) {
    CHECK(wait_done(&c[i], 60000));
    CHECK(*c[i].value == 101 + (unsigned long)i);
    CHECK(loop_errors[i + 1] == 0);
    capstone_context_revoke(&c[i]);
  }
  return 0;
}

/* A further context preempted many times while the first context makes
   rounds: its loan and continuation survive every preemption, and the first
   context's rounds are served meanwhile. */
static unsigned long computer(void *arg)
{
  (void)arg;
  spin_ms(300);
  return 11;
}

static int preempted(void)
{
  struct capstone_context c;
  long id = start_thread(&c, computer, 0);
  CHECK(id > 0);
  unsigned long rounds = 0;
  long t0 = monotonic_ms();
  while (!*c.done && monotonic_ms() - t0 < 20000) {
    sched_yield();
    ++rounds;
  }
  CHECK(*c.done == 1 && *c.value == 11);
  printf("thread-probe preempted: %lu rounds meanwhile\n", rounds);
  CHECK(rounds > 0);
  capstone_context_revoke(&c);
  return 0;
}

/* More delegated rounds than the platform has revocation nodes (65536 in the
   VM), from one context: every round's loan must give its node back. */
static int many_rounds(void)
{
  long slowest = 0, fastest = 1L << 40, t0 = monotonic_ms(), start = t0;
  for (long i = 1; i <= 100000; ++i) {
    CHECK(sched_yield() == 0);
    if (i % 10000 == 0) {
      long t = monotonic_ms(), block = t - t0;
      if (block > slowest) slowest = block;
      if (block < fastest) fastest = block;
      t0 = t;
    }
  }
  printf("thread-probe many-rounds: 100000 rounds in %ld ms; 10000-round blocks %ld to %ld ms\n",
         monotonic_ms() - start, fastest, slowest);
  return 0;
}

/* Signal state stays with the first context for now: a further context's
   signal requests answer ENOSYS, visibly, instead of acting on the first
   context's state. */
static volatile long sig_action_rc, sig_action_errno, sig_mask_rc, sig_mask_errno;
static void on_usr1(int sig) { (void)sig; }
static unsigned long signaller(void *arg)
{
  (void)arg;
  struct sigaction sa = {0};
  sigset_t set;
  sa.sa_handler = on_usr1;
  sig_action_rc = sigaction(SIGUSR1, &sa, 0);
  sig_action_errno = errno;
  sigemptyset(&set);
  sigaddset(&set, SIGUSR2);
  sig_mask_rc = sigprocmask(SIG_BLOCK, &set, 0);
  sig_mask_errno = errno;
  return 5;
}

/* Signals are each context's (B8): a further context installs a handler for
   the process and blocks a signal in its own mask, which leaves the first
   context's mask as it was. */
static int signals_own(void)
{
  struct capstone_context c;
  sigset_t now;
  long id = start_thread(&c, signaller, 0);
  CHECK(id > 0);
  CHECK(wait_done(&c, 20000));
  CHECK(*c.value == 5);
  CHECK(sig_action_rc == 0 && sig_mask_rc == 0);
  CHECK(sigprocmask(SIG_BLOCK, 0, &now) == 0 && !sigismember(&now, SIGUSR2));
  struct sigaction old;
  CHECK(sigaction(SIGUSR1, 0, &old) == 0 && old.sa_handler == on_usr1);
  capstone_context_revoke(&c);
  return 0;
}

/* A signal sent to the process reaches a context that does not block it
   (Linux picks the thread): the first context blocks both signals, the other
   does not, and every handler runs in the other. Two signals per burst, and a
   handler that takes a while. */
static __thread int who;
static volatile int handled_by[3];
static volatile int stop_rounds;
static int devnull = -1;
static void on_burst(int sig)
{
  (void)sig;
  handled_by[who == 2 ? 2 : 1]++;
  spin_ms(30);
}

static unsigned long rounds_until_stopped(void *arg)
{
  (void)arg;
  who = 2;
  while (!stop_rounds)
    if (write(devnull, "", 0) < 0)
      return 1;
  return 3;
}

static int signal_unblocked_context(void)
{
  struct capstone_context c;
  struct sigaction sa = {0};
  who = 1;
  sa.sa_handler = on_burst;
  CHECK(sigaction(SIGUSR1, &sa, 0) == 0 && sigaction(SIGUSR2, &sa, 0) == 0);
  devnull = open("/dev/null", O_WRONLY);
  CHECK(devnull >= 0);
  long id = start_thread(&c, rounds_until_stopped, 0);
  CHECK(id > 0);
  sigset_t both;
  sigemptyset(&both);
  sigaddset(&both, SIGUSR1);
  sigaddset(&both, SIGUSR2);
  CHECK(sigprocmask(SIG_BLOCK, &both, 0) == 0);
  for (int i = 0; i < 10; ++i) {
    CHECK(kill(getpid(), SIGUSR1) == 0 && kill(getpid(), SIGUSR2) == 0);
    long t0 = monotonic_ms();
    while (handled_by[2] < 2 * (i + 1) && monotonic_ms() - t0 < 5000)
      sched_yield();
  }
  stop_rounds = 1;
  CHECK(wait_done(&c, 20000) && *c.value == 3);
  printf("thread-probe signal-unblocked-context: %d handlers in the first context, %d in the other\n",
         handled_by[1], handled_by[2]);
  CHECK(handled_by[1] == 0 && handled_by[2] == 20);
  CHECK(sigprocmask(SIG_UNBLOCK, &both, 0) == 0);
  capstone_context_revoke(&c);
  return 0;
}

/* A write to a pipe without a reader in a further context: Linux sends
   SIGPIPE to the writing thread, and its default action ends the process. */
static volatile long pipe_rc, pipe_errno;
static unsigned long broken_pipe_writer(void *arg)
{
  (void)arg;
  int fds[2];
  if (pipe(fds)) return 1;
  close(fds[0]);
  pipe_rc = write(fds[1], "x", 1);
  pipe_errno = errno;
  return 2;
}

static int sigpipe_child(int ignored)
{
  struct capstone_context c;
  if (ignored)
    CHECK(signal(SIGPIPE, SIG_IGN) != SIG_ERR);
  long id = start_thread(&c, broken_pipe_writer, 0);
  CHECK(id > 0);
  CHECK(wait_done(&c, 20000));
  if (!ignored) {
    /* the process should be gone by now */
    long t0 = monotonic_ms();
    while (monotonic_ms() - t0 < 2000)
      sched_yield();
    fprintf(stderr, "thread-probe sigpipe-child: REACHED (write %ld, errno %ld)\n",
            pipe_rc, pipe_errno);
    return 1;
  }
  CHECK(pipe_rc == -1 && pipe_errno == EPIPE);
  capstone_context_revoke(&c);
  return 0;
}

/* Exec in place from a further context: the new image starts with the
   application's signal mask, not with the context thread's full block. */
static const char *self_path;
static volatile long exec_rc, exec_errno;
static unsigned long execer(void *arg)
{
  (void)arg;
  char *const args[] = {(char *)self_path, "print-mask", 0};
  exec_rc = execv(self_path, args);
  exec_errno = errno;
  return 4;
}

static int exec_child(void)
{
  struct capstone_context c;
  long id = start_thread(&c, execer, 0);
  CHECK(id > 0);
  CHECK(wait_done(&c, 20000));
  fprintf(stderr, "thread-probe exec-child: REACHED (execv %ld, errno %ld)\n", exec_rc, exec_errno);
  return 1;
}

/* The launcher's blocked mask, as the program it runs sees it. */
static int print_mask(void)
{
  char buf[4096] = {0};
  int fd = open("/proc/self/status", O_RDONLY);
  CHECK(fd >= 0);
  long n = read(fd, buf, sizeof buf - 1);
  close(fd);
  CHECK(n > 0);
  char *line = strstr(buf, "SigBlk:");
  CHECK(line);
  char *end = strchr(line, '\n');
  if (end) *end = 0;
  printf("thread-probe print-mask: %s\n", line);
  return 0;
}

/* ---- T2: parking through the launcher (B6, B12 in the domain) ---- */

/* The value check, a timeout that keeps its budget, a WAKE of nobody, and an
   operation that is not served. */
static int futex_basic(void)
{
  static volatile int w = 5;
  CHECK(fwait(&w, 4, 0) == -1 && errno == EAGAIN);
  struct timespec to = {0, 50 * 1000000L};
  long t0 = monotonic_ms();
  CHECK(fwait(&w, 5, &to) == -1 && errno == ETIMEDOUT);
  long waited = monotonic_ms() - t0;
  printf("thread-probe futex-basic: timed out after %ld ms of 50\n", waited);
  CHECK(waited >= 50 && waited < 5000);
  CHECK(fwake(&w, 1) == 0);
  CHECK(syscall(SYS_futex, &w, F_WAKE_OP | F_PRIVATE, 1, 0, &w, 0) == -1 && errno == ENOSYS);
  /* Linux's argument checks: the realtime clock flag outside WAIT_BITSET,
     a misaligned word, negative REQUEUE counts */
  CHECK(syscall(SYS_futex, &w, F_WAKE | F_PRIVATE | 256, 1) == -1 && errno == ENOSYS);
  static volatile int pair[2];
  CHECK(syscall(SYS_futex, (volatile char *)pair + 1, F_WAKE | F_PRIVATE, 1) == -1 &&
        errno == EINVAL);
  CHECK(frequeue(&w, -1, 1, &w) == -1 && errno == EINVAL);
  CHECK(frequeue(&w, 0, -1, &w) == -1 && errno == EINVAL);
  return 0;
}

/* A waiter parked in another context is selected by WAKE (1, not 0: it was
   queued) and sees the value that was stored before the wake. */
static volatile int fword, fready;
static volatile long fresult, fseen;
static unsigned long futex_waiter(void *arg)
{
  (void)arg;
  fready = 1;
  fresult = fwait(&fword, 0, 0);
  fseen = fword;
  return 6;
}

static int futex_wake(void)
{
  struct capstone_context c;
  CHECK(start_thread(&c, futex_waiter, 0) > 0);
  long t0 = monotonic_ms();
  while (!fready && monotonic_ms() - t0 < 20000)
    sched_yield();
  spin_ms(50);
  fword = 1;
  CHECK(fwake(&fword, 1) == 1);
  CHECK(wait_done(&c, 20000) && *c.value == 6);
  CHECK(fresult == 0 && fseen == 1);
  capstone_context_revoke(&c);
  return 0;
}

/* REQUEUE as musl's condition variables issue it: no wake, one moved. The
   moved waiter answers a WAKE on the destination, the other one on the
   source; nobody is left. */
static volatile int qa, qb, qready[2];
static volatile long qres[2];
static unsigned long queue_waiter(void *arg)
{
  int i = (int)(uintptr_t)arg;
  qready[i] = 1;
  qres[i] = fwait(&qa, 0, 0);
  return 10 + (unsigned long)i;
}

static int futex_requeue(void)
{
  struct capstone_context c[2];
  for (int i = 0; i < 2; ++i)
    CHECK(start_thread(&c[i], queue_waiter, (void *)(uintptr_t)i) > 0);
  long t0 = monotonic_ms();
  while (!(qready[0] && qready[1]) && monotonic_ms() - t0 < 20000)
    sched_yield();
  spin_ms(100);
  CHECK(frequeue(&qa, 0, 1, &qb) == 1);
  CHECK(fwake(&qa, 1) == 1);
  CHECK(fwake(&qb, 1) == 1);
  for (int i = 0; i < 2; ++i) {
    CHECK(wait_done(&c[i], 20000) && *c[i].value == 10 + (unsigned long)i);
    CHECK(qres[i] == 0);
    capstone_context_revoke(&c[i]);
  }
  CHECK(fwake(&qa, 1) == 0 && fwake(&qb, 1) == 0);
  return 0;
}

/* B6 and B12: the window between the waiter's compare and its WAIT request.
   The waiter (a further context) compares the lock word (2, held), then
   stops there for four quanta (the runtime's test gap); meanwhile the first
   context releases the word and WAKEs (B6) or REQUEUEs (B12) with nobody
   queued. Because the waiter read the bucket's generation before its
   compare, its WAIT answers RECHECK and returns at once instead of sleeping
   through the release. The control builds the same wait with the generation
   read after the window and must lose the wake: a 500 ms deadline expires. */
static volatile int m6, other6, in_gap;
static volatile long w6_rc, w6_errno, w6_ms;
static void gap_hold(void)
{
  if (who == 2) {
    in_gap = 1;
    spin_ms(20);
  }
}

static unsigned long window_waiter(void *arg)
{
  int reversed = (int)(uintptr_t)arg;
  who = 2;
  long t0 = monotonic_ms();
  if (!reversed) {
    w6_rc = fwait(&m6, 2, 0);
    w6_errno = errno;
  } else {
    struct timespec now;
    if (m6 != 2) return 1;
    in_gap = 1;
    spin_ms(20);
    uint64_t gen = __capstone_park_generation(&m6);
    clock_gettime(CLOCK_MONOTONIC, &now);
    uint64_t deadline = (uint64_t)now.tv_sec * 1000000000u + (uint64_t)now.tv_nsec + 500000000u;
    w6_rc = __capstone_delegate_ints(CAPSTONE_NR_PARK_WAIT,
                                     (uint64_t)__builtin_capstone_cap_get_cursor((void *)&m6),
                                     gen, deadline);
  }
  w6_ms = monotonic_ms() - t0;
  return 12;
}

static int window(int requeue, int reversed)
{
  struct capstone_context c;
  who = 1;
  m6 = 2;
  __capstone_futex_test_gap = gap_hold;
  CHECK(start_thread(&c, window_waiter, (void *)(uintptr_t)reversed) > 0);
  long t0 = monotonic_ms();
  while (!in_gap && monotonic_ms() - t0 < 20000)
    sched_yield();
  CHECK(in_gap);
  m6 = 0;
  long selected = requeue ? frequeue(&m6, 0, 1, &other6) : fwake(&m6, 1);
  CHECK(wait_done(&c, 20000) && *c.value == 12);
  __capstone_futex_test_gap = 0;
  printf("thread-probe %s%s: %s selected %ld, the waiter returned %ld after %ld ms\n",
         requeue ? "b12" : "b6", reversed ? "-control" : "", requeue ? "REQUEUE" : "WAKE",
         selected, w6_rc, w6_ms);
  CHECK(selected == 0);
  if (reversed)
    CHECK(w6_rc == -ETIMEDOUT && w6_ms >= 500);     /* the wake was lost */
  else
    CHECK(w6_rc == 0 && w6_ms < 500);               /* RECHECK, no sleep */
  capstone_context_revoke(&c);
  return 0;
}

/* B6's counter: two contexts take a futex mutex (Drepper's three-state lock)
   400 times each around an increment, holding it across preemptions so that
   they contend and park; the count is exact and nobody sleeps through a
   release. */
#define B6_ROUNDS 400
static volatile int mtx, parked_waits;
static volatile unsigned long b6_count;
static void mtx_lock(void)
{
  int c = 0;
  if (__atomic_compare_exchange_n(&mtx, &c, 1, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
    return;
  if (c != 2)
    c = __atomic_exchange_n(&mtx, 2, __ATOMIC_ACQUIRE);
  while (c != 0) {
    fwait(&mtx, 2, 0);
    __atomic_fetch_add(&parked_waits, 1, __ATOMIC_RELAXED);
    c = __atomic_exchange_n(&mtx, 2, __ATOMIC_ACQUIRE);
  }
}
static void mtx_unlock(void)
{
  if (__atomic_fetch_sub(&mtx, 1, __ATOMIC_RELEASE) != 1) {
    __atomic_store_n(&mtx, 0, __ATOMIC_RELEASE);
    fwake(&mtx, 1);
  }
}
static int counting(void)
{
  for (int i = 0; i < B6_ROUNDS; ++i) {
    mtx_lock();
    unsigned long v = b6_count;
    spin_ms(1);                  /* hold it: a quantum ends inside now and then */
    b6_count = v + 1;
    mtx_unlock();
  }
  return 0;
}
static unsigned long counter_thread(void *arg)
{
  (void)arg;
  counting();
  return 13;
}

static int b6_count_mode(void)
{
  struct capstone_context c;
  CHECK(start_thread(&c, counter_thread, 0) > 0);
  counting();
  CHECK(wait_done(&c, 120000) && *c.value == 13);
  printf("thread-probe b6-count: count %lu of %d, %d waits parked\n", b6_count, 2 * B6_ROUNDS,
         parked_waits);
  CHECK(b6_count == 2 * B6_ROUNDS);
  CHECK(parked_waits > 0);
  capstone_context_revoke(&c);
  return 0;
}

/* The first context parked when a signal arrives, three ways, each as Linux
   answers the same futex call (checked natively): an untimed wait under
   SA_RESTART runs the handler at once and goes on to the wake; without
   SA_RESTART it fails with EINTR; a timed wait fails with EINTR even under
   SA_RESTART (Linux restarts it through a restart block, which a handler
   turns into EINTR). The record is out of the queue before the handler runs. */
static volatile int ps_word, ps_handled;
static volatile long ps_handled_at;
static long ps_start;
static void on_park_signal(int sig)
{
  (void)sig;
  ps_handled++;
  ps_handled_at = monotonic_ms() - ps_start;
}
static unsigned long signal_then_wake(void *arg)
{
  (void)arg;
  spin_ms(200);
  kill(getpid(), SIGUSR1);
  spin_ms(300);
  ps_word = 1;
  fwake(&ps_word, 1);
  return 14;
}

static int park_signal(int restart, int timed)
{
  struct capstone_context c;
  struct sigaction sa = {0};
  sa.sa_handler = on_park_signal;
  sa.sa_flags = restart ? SA_RESTART : 0;
  CHECK(sigaction(SIGUSR1, &sa, 0) == 0);
  ps_start = monotonic_ms();
  CHECK(start_thread(&c, signal_then_wake, 0) > 0);
  struct timespec to = {3, 0};
  long rc = fwait(&ps_word, 0, timed ? &to : 0);
  long err = errno, took = monotonic_ms() - ps_start;
  CHECK(wait_done(&c, 20000) && *c.value == 14);
  printf("thread-probe park-signal%s: futex %ld (errno %ld) after %ld ms, handler at %ld ms\n",
         timed ? "-timed" : restart ? "" : "-eintr", rc, err, took, ps_handled_at);
  CHECK(ps_handled == 1);
  if (restart && !timed)
    CHECK(rc == 0 && took >= 450 && took < 2500 && ps_handled_at < 400);  /* before the wake */
  else
    CHECK(rc == -1 && err == EINTR && took < 450);
  capstone_context_revoke(&c);
  return 0;
}

/* ---- T3: runtime locks and identities (Q6, Q2's first part) ---- */

/* Every context has its own thread identity: the first context's is the pid,
   a minted one's lies above Linux's pid range, and no two are the same. */
static volatile long tids[TRANSPORTS + 1];
static unsigned long tid_child(void *arg)
{
  tids[(uintptr_t)arg] = syscall(SYS_gettid);
  return 20;
}

static int tid_identity(void)
{
  struct capstone_context c[3];
  tids[0] = syscall(SYS_gettid);
  CHECK(tids[0] == getpid());
  for (int i = 1; i <= 3; ++i)
    CHECK(start_thread(&c[i - 1], tid_child, (void *)(uintptr_t)i) > 0);
  for (int i = 1; i <= 3; ++i) {
    CHECK(wait_done(&c[i - 1], 20000));
    /* above Linux's pid range, and inside the 30 bits musl keeps in a lock
       word (bit 30 is MAYBE_WAITERS, 0x3fffffff a marker) */
    CHECK(tids[i] >= 0x400000 && tids[i] <= 0x3ffffffe);
    for (int j = 0; j < i; ++j)
      CHECK(tids[i] != tids[j]);
    capstone_context_revoke(&c[i - 1]);
  }
  printf("thread-probe tid-identity: %ld %ld %ld %ld\n", tids[0], tids[1], tids[2], tids[3]);
  return 0;
}

/* stdio from every context at once into one FILE: musl locks it per call once
   a second context exists, by thread identity, so every line comes out whole
   and each context's lines in order. Enough lines that quanta end inside
   fprintf many times. */
#define LINES 1500
static FILE *shared_file;
static void print_lines(int who)
{
  for (int i = 0; i < LINES; ++i)
    fprintf(shared_file, "L %d %d %d ........................................\n", who, i,
            who * 10000 + i);
}
static unsigned long line_printer(void *arg)
{
  print_lines((int)(uintptr_t)arg);
  return 21;
}

static int stdio_lines(void)
{
  struct capstone_context c[TRANSPORTS];
  char path[] = "/tmp/thread-probe-stdio-XXXXXX";
  int fd = mkstemp(path);
  CHECK(fd >= 0);
  shared_file = fdopen(fd, "w+");
  CHECK(shared_file);
  for (int i = 0; i < TRANSPORTS; ++i)
    CHECK(start_thread(&c[i], line_printer, (void *)(uintptr_t)(i + 1)) > 0);
  print_lines(0);
  for (int i = 0; i < TRANSPORTS; ++i) {
    CHECK(wait_done(&c[i], 120000) && *c[i].value == 21);
    capstone_context_revoke(&c[i]);
  }
  CHECK(fflush(shared_file) == 0 && fseek(shared_file, 0, SEEK_SET) == 0);
  int seen[TRANSPORTS + 1] = {0}, bad = 0;
  long lines = 0;
  char line[128];
  while (fgets(line, sizeof line, shared_file)) {
    int who, i, check, used = 0;
    ++lines;
    if (sscanf(line, "L %d %d %d %n", &who, &i, &check, &used) != 3 || who < 0 ||
        who > TRANSPORTS || check != who * 10000 + i || i != seen[who] ||
        strcmp(line + used, "........................................\n")) {
      ++bad;
      continue;
    }
    ++seen[who];
  }
  fclose(shared_file);
  unlink(path);
  printf("thread-probe stdio-lines: %ld lines, %d malformed or out of order\n", lines, bad);
  CHECK(bad == 0);
  for (int w = 0; w <= TRANSPORTS; ++w)
    CHECK(seen[w] == LINES);
  return 0;
}

/* The heap from every context at once: blocks of varying size, each filled
   with its owner's pattern, checked before it is freed or grown. */
#define HEAP_ROUNDS 1500
static volatile int heap_errors[TRANSPORTS + 1];
static void heap_work(int who)
{
  unsigned char *live[8] = {0};
  size_t size[8] = {0};
  for (int i = 0; i < HEAP_ROUNDS; ++i) {
    int k = i % 8;
    if (live[k]) {
      for (size_t j = 0; j < size[k]; ++j)
        if (live[k][j] != (unsigned char)(who * 31 + k)) { ++heap_errors[who]; break; }
      if (i % 5 == 0) {
        size_t grown = size[k] * 2 + 1;
        unsigned char *q = realloc(live[k], grown);
        if (!q) { ++heap_errors[who]; continue; }
        for (size_t j = 0; j < size[k]; ++j)
          if (q[j] != (unsigned char)(who * 31 + k)) { ++heap_errors[who]; break; }
        free(q);
      } else {
        free(live[k]);
      }
    }
    size[k] = (size_t)((i * 37 + who * 11) % 500 + 1);
    live[k] = malloc(size[k]);
    if (!live[k]) { ++heap_errors[who]; continue; }
    memset(live[k], who * 31 + k, size[k]);
  }
  for (int k = 0; k < 8; ++k)
    free(live[k]);
}
static unsigned long heap_worker(void *arg)
{
  heap_work((int)(uintptr_t)arg);
  return 22;
}

static int heap_stress(void)
{
  struct capstone_context c[TRANSPORTS];
  for (int i = 0; i < TRANSPORTS; ++i)
    CHECK(start_thread(&c[i], heap_worker, (void *)(uintptr_t)(i + 1)) > 0);
  heap_work(0);
  for (int i = 0; i < TRANSPORTS; ++i) {
    CHECK(wait_done(&c[i], 120000) && *c[i].value == 22);
    capstone_context_revoke(&c[i]);
  }
  int total = 0;
  for (int w = 0; w <= TRANSPORTS; ++w)
    total += heap_errors[w];
  printf("thread-probe heap-stress: %d contexts, %d rounds each, %d errors\n", TRANSPORTS + 1,
         HEAP_ROUNDS, total);
  CHECK(total == 0);
  return 0;
}

/* stdio calls that take a FILE's lock again while they hold it (puts through
   fwrite, perror, fclose through fflush, putc inside flockfile) from a minted
   context: musl's recursive check compares its thread identity with the lock
   word's owner, so an identity that collides with the word's flag bits would
   make the context wait for itself. */
static unsigned long nested_stdio(void *arg)
{
  (void)arg;
  puts("thread-probe stdio-nested: puts from a minted context");
  errno = ENOENT;
  perror("thread-probe stdio-nested: perror");
  FILE *f = fopen("/dev/null", "w");
  if (!f) return 1;
  fputs("x", f);
  if (fclose(f)) return 2;
  flockfile(stdout);
  putc('.', stdout);
  putc('\n', stdout);
  funlockfile(stdout);
  fflush(stdout);
  return 25;
}

static int stdio_nested(void)
{
  struct capstone_context c[2];
  for (int i = 0; i < 2; ++i)
    CHECK(start_thread(&c[i], nested_stdio, 0) > 0);
  for (int i = 0; i < 2; ++i) {
    CHECK(wait_done(&c[i], 20000) && *c[i].value == 25);
    capstone_context_revoke(&c[i]);
  }
  return 0;
}

/* posix_spawn from every context at once: the request is packed in one static
   block, so each child's exit status must be the one its own request asked for. */
#include <spawn.h>
#include <sys/wait.h>
extern char **environ;
static volatile int spawn_errors[TRANSPORTS + 1];
static void spawn_work(int who)
{
  for (int k = 0; k < 3; ++k) {
    int want = who * 3 + k + 1, status = 0;
    char code[16];
    snprintf(code, sizeof code, "exit %d", want);
    char *args[] = {"sh", "-c", code, 0};
    pid_t pid;
    if (posix_spawn(&pid, "/bin/sh", 0, 0, args, environ) ||
        waitpid(pid, &status, 0) != pid || !WIFEXITED(status) || WEXITSTATUS(status) != want)
      ++spawn_errors[who];
  }
}
static unsigned long spawner(void *arg)
{
  spawn_work((int)(uintptr_t)arg);
  return 26;
}

static int spawn_concurrent(void)
{
  struct capstone_context c[TRANSPORTS];
  for (int i = 0; i < TRANSPORTS; ++i)
    CHECK(start_thread(&c[i], spawner, (void *)(uintptr_t)(i + 1)) > 0);
  spawn_work(0);
  int total = 0;
  for (int i = 0; i < TRANSPORTS; ++i) {
    CHECK(wait_done(&c[i], 120000) && *c[i].value == 26);
    capstone_context_revoke(&c[i]);
  }
  for (int w = 0; w <= TRANSPORTS; ++w)
    total += spawn_errors[w];
  printf("thread-probe spawn-concurrent: %d spawns, %d wrong\n", 3 * (TRANSPORTS + 1), total);
  CHECK(total == 0);
  return 0;
}

/* Anonymous mappings from every context at once, through the runtime's
   mapping table: each mapping holds its owner's pattern until it is unmapped. */
#include <sys/mman.h>
static volatile int map_errors[TRANSPORTS + 1];
static void map_work(int who)
{
  for (int i = 0; i < 200; ++i) {
    size_t len = (size_t)(4096 * (1 + (i + who) % 3));
    unsigned char *m = mmap(0, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) { ++map_errors[who]; continue; }
    memset(m, who + 1, len);
    for (size_t j = 0; j < len; j += 512)
      if (m[j] != who + 1) { ++map_errors[who]; break; }
    if (munmap(m, len)) ++map_errors[who];
  }
}
static unsigned long mapper(void *arg)
{
  map_work((int)(uintptr_t)arg);
  return 27;
}

static int mmap_concurrent(void)
{
  struct capstone_context c[TRANSPORTS];
  for (int i = 0; i < TRANSPORTS; ++i)
    CHECK(start_thread(&c[i], mapper, (void *)(uintptr_t)(i + 1)) > 0);
  map_work(0);
  int total = 0;
  for (int i = 0; i < TRANSPORTS; ++i) {
    CHECK(wait_done(&c[i], 120000) && *c[i].value == 27);
    capstone_context_revoke(&c[i]);
  }
  for (int w = 0; w <= TRANSPORTS; ++w)
    total += map_errors[w];
  printf("thread-probe mmap-concurrent: %d contexts, 200 mappings each, %d errors\n",
         TRANSPORTS + 1, total);
  CHECK(total == 0);
  return 0;
}

/* Contexts that create contexts: three at once each mint (from the one arena)
   and create a THREAD context of its own, which makes a delegated call; the
   creator waits for it and revokes it. */
static unsigned long grandchild(void *arg)
{
  char line[48];
  int n = snprintf(line, sizeof line, "grandchild of %d\n", (int)(uintptr_t)arg);
  return write(devnull, line, (size_t)n) == n ? 30 : 1;
}
static unsigned long creator(void *arg)
{
  struct capstone_context g;
  if (start_thread(&g, grandchild, arg) <= 0) return 1;
  if (!wait_done(&g, 60000)) return 2;
  unsigned long v = *g.value;
  capstone_context_revoke(&g);
  return v == 30 ? 31 : 3;
}

static int nested_create(void)
{
  struct capstone_context c[3];
  devnull = open("/dev/null", O_WRONLY);
  CHECK(devnull >= 0);
  for (int i = 0; i < 3; ++i)
    CHECK(start_thread(&c[i], creator, (void *)(uintptr_t)i) > 0);
  for (int i = 0; i < 3; ++i) {
    CHECK(wait_done(&c[i], 120000));
    printf("thread-probe nested-create: creator %d returned %lu\n", i, *c[i].value);
    CHECK(*c[i].value == 31);
    capstone_context_revoke(&c[i]);
  }
  return 0;
}

/* B9: two contexts on scalar and capability-valued atomics while quanta end
   inside them. The capability operations go through the runtime's generic
   atomics, which hold its leaf spin lock.
   - exchange: every value stored is a distinct tagged capability to one cell
     of an array (cell i: stored once, by one context); the history is
     linearizable only if every stored value but the last comes back from
     exactly one exchange, tagged, with the array's bounds;
   - compare-and-swap: both contexts advance one shared capability by one
     cell per successful CAS; the cursor ends exactly 2 * rounds cells on, a
     lost update would leave it short;
   - fetch_add on a scalar counter: exact. */
#define B9_ROUNDS 20000
unsigned long __capstone_spin_contended(void);
static int b9_cells[2 * B9_ROUNDS + 1];
static int *volatile b9_slot, *volatile b9_cursor;
static volatile int b9_counter;
static unsigned char b9_returned[2 * B9_ROUNDS];
static volatile int b9_bad[2];
static long b9_index(int *p)
{
  if (!__builtin_capstone_cap_get_tag(p) ||
      __builtin_capstone_cap_get_base(p) != __builtin_capstone_cap_get_base(b9_cells) ||
      __builtin_capstone_cap_get_end(p) != __builtin_capstone_cap_get_end(b9_cells))
    return -1;
  return (long)(__builtin_capstone_cap_get_cursor(p) - __builtin_capstone_cap_get_cursor(b9_cells)) /
         (long)sizeof(int);
}
static void b9_work(int who)
{
  for (int i = 0; i < B9_ROUNDS; ++i) {
    __atomic_fetch_add(&b9_counter, 1, __ATOMIC_RELAXED);
    int *old = __atomic_exchange_n(&b9_slot, &b9_cells[who * B9_ROUNDS + i], __ATOMIC_SEQ_CST);
    if (old) {
      long k = b9_index(old);
      if (k < 0 || k >= 2 * B9_ROUNDS || b9_returned[k]++)
        ++b9_bad[who];
    }
    int *expected = __atomic_load_n(&b9_cursor, __ATOMIC_ACQUIRE);
    while (!__atomic_compare_exchange_n(&b9_cursor, &expected, expected + 1, 0,
                                        __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))
      ;
  }
}
static unsigned long b9_worker(void *arg)
{
  (void)arg;
  b9_work(1);
  return 23;
}

static int b9(void)
{
  struct capstone_context c;
  unsigned long before = __capstone_spin_contended();
  b9_cursor = b9_cells;
  CHECK(start_thread(&c, b9_worker, 0) > 0);
  b9_work(0);
  CHECK(wait_done(&c, 120000) && *c.value == 23);
  unsigned long contended = __capstone_spin_contended() - before;
  long last = b9_index(b9_slot), missing = 0;
  for (long k = 0; k < 2 * B9_ROUNDS; ++k)
    if (b9_returned[k] != (k == last ? 0 : 1))
      ++missing;
  long advanced = b9_index(b9_cursor);
  printf("thread-probe b9: counter %d of %d; exchange: %d and %d bad, %ld cells not returned "
         "exactly once; cursor %ld of %d; the lock found taken %lu times\n", b9_counter,
         2 * B9_ROUNDS, b9_bad[0], b9_bad[1], missing, advanced, 2 * B9_ROUNDS, contended);
  CHECK(b9_counter == 2 * B9_ROUNDS);
  CHECK(b9_bad[0] == 0 && b9_bad[1] == 0 && last >= 0 && missing == 0);
  CHECK(advanced == 2 * B9_ROUNDS);
  CHECK(contended > 0);
  capstone_context_revoke(&c);
  return 0;
}

/* B14: no handler while a runtime-internal lock is held. The first context
   holds lock A and waits for lock B, which a further context holds; that
   context sends SIGUSR1 meanwhile, so the event arrives in one of the first
   context's rounds inside the wait. The handler takes A itself: run there, it
   would wait for its own context forever. It must run once, after A is
   released, with no lock held. */
#include <capstone/lock.h>
static volatile int lock_a, lock_b, b_held, a_released, b14_handled, b14_depth = -1;
static void on_b14(int sig)
{
  (void)sig;
  b14_depth = __capstone_lock_depth();
  if (!a_released) return;           /* too early: leave the counters unset */
  capstone_lock(&lock_a);
  capstone_unlock(&lock_a);
  b14_handled++;
}
static unsigned long b_holder(void *arg)
{
  (void)arg;
  capstone_lock(&lock_b);
  b_held = 1;
  spin_ms(100);
  kill(getpid(), SIGUSR1);
  spin_ms(100);
  capstone_unlock(&lock_b);
  return 24;
}

static int b14(void)
{
  struct capstone_context c;
  struct sigaction sa = {0};
  sa.sa_handler = on_b14;
  sa.sa_flags = SA_RESTART;
  CHECK(sigaction(SIGUSR1, &sa, 0) == 0);
  CHECK(start_thread(&c, b_holder, 0) > 0);   /* the runtime's locks are on from here */
  long t0 = monotonic_ms();
  while (!b_held && monotonic_ms() - t0 < 20000)
    sched_yield();
  CHECK(b_held);
  capstone_lock(&lock_a);
  capstone_lock(&lock_b);                      /* waits for the other context */
  capstone_unlock(&lock_b);
  int handled_inside = b14_handled || b14_depth >= 0;
  a_released = 1;
  capstone_unlock(&lock_a);                    /* the handler runs here, at depth 0 */
  int at_release = b14_handled;                /* before any further round */
  CHECK(wait_done(&c, 20000) && *c.value == 24);
  printf("thread-probe b14: handler ran %d time(s), at lock depth %d, %s, %s\n", b14_handled,
         b14_depth, handled_inside ? "while a lock was held" : "after the last release",
         at_release ? "by that release" : "only at a later round");
  CHECK(!handled_inside && b14_handled == 1 && b14_depth == 0 && at_release == 1);
  capstone_context_revoke(&c);
  return 0;
}

/* A REGISTER context has no transport: its calls fail with EIO instead of
   using anyone else's. */
static volatile long unserved_rc, unserved_errno;
static unsigned long no_transport_child(void *arg)
{
  (void)arg;
  unserved_rc = write(1, "x", 1);
  unserved_errno = errno;
  return 9;
}

static int no_transport(void)
{
  struct capstone_context c;
  struct capstone_context_event ev;
  CHECK(!capstone_context_mint(&c, AREA_BYTES, no_transport_child, 0));
  long id = capstone_context_create(&c, CAPSTONE_CONTEXT_REGISTER);
  CHECK(id > 0);
  do {
    memset(&ev, 0, sizeof ev);
    CHECK(capstone_context_step((unsigned long)id, &ev) == 0);
  } while (ev.kind == 1);
  CHECK(ev.kind == 0 && ev.result == CAPSTONE_CONTEXT_EXITED);
  CHECK(*c.value == 9 && unserved_rc == -1 && unserved_errno == EIO);
  CHECK(capstone_context_forget((unsigned long)id) == 0);
  capstone_context_revoke(&c);
  return 0;
}

int main(int argc, char **argv)
{
  int rc;
  if (argc < 2) {
    fprintf(stderr, "usage: thread-probe MODE\n");
    return 2;
  }
  mode = argv[1];
  self_path = argv[0];
  if (!strcmp(mode, "transport")) rc = transport();
  else if (!strcmp(mode, "blocking")) rc = blocking();
  else if (!strcmp(mode, "exit-child")) rc = exit_child();
  else if (!strcmp(mode, "fault-child")) rc = fault_child();
  else if (!strcmp(mode, "reserve")) rc = reserve();
  else if (!strcmp(mode, "reuse")) rc = reuse();
  else if (!strcmp(mode, "concurrent")) rc = concurrent();
  else if (!strcmp(mode, "signals-own")) rc = signals_own();
  else if (!strcmp(mode, "no-transport")) rc = no_transport();
  else if (!strcmp(mode, "preempted")) rc = preempted();
  else if (!strcmp(mode, "many-rounds")) rc = many_rounds();
  else if (!strcmp(mode, "signal-unblocked-context")) rc = signal_unblocked_context();
  else if (!strcmp(mode, "sigpipe-child")) rc = sigpipe_child(0);
  else if (!strcmp(mode, "sigpipe-ignored")) rc = sigpipe_child(1);
  else if (!strcmp(mode, "exec-child")) rc = exec_child();
  else if (!strcmp(mode, "print-mask")) rc = print_mask();
  else if (!strcmp(mode, "futex-basic")) rc = futex_basic();
  else if (!strcmp(mode, "futex-wake")) rc = futex_wake();
  else if (!strcmp(mode, "futex-requeue")) rc = futex_requeue();
  else if (!strcmp(mode, "b6")) rc = window(0, 0);
  else if (!strcmp(mode, "b6-control")) rc = window(0, 1);
  else if (!strcmp(mode, "b12")) rc = window(1, 0);
  else if (!strcmp(mode, "b6-count")) rc = b6_count_mode();
  else if (!strcmp(mode, "park-signal")) rc = park_signal(1, 0);
  else if (!strcmp(mode, "park-signal-eintr")) rc = park_signal(0, 0);
  else if (!strcmp(mode, "park-signal-timed")) rc = park_signal(1, 1);
  else if (!strcmp(mode, "tid-identity")) rc = tid_identity();
  else if (!strcmp(mode, "stdio-lines")) rc = stdio_lines();
  else if (!strcmp(mode, "heap-stress")) rc = heap_stress();
  else if (!strcmp(mode, "b9")) rc = b9();
  else if (!strcmp(mode, "b14")) rc = b14();
  else if (!strcmp(mode, "stdio-nested")) rc = stdio_nested();
  else if (!strcmp(mode, "spawn-concurrent")) rc = spawn_concurrent();
  else if (!strcmp(mode, "mmap-concurrent")) rc = mmap_concurrent();
  else if (!strcmp(mode, "nested-create")) rc = nested_create();
  else {
    fprintf(stderr, "thread-probe: unknown mode %s\n", mode);
    return 2;
  }
  if (!rc)
    printf("thread-probe %s: PASS\n", mode);
  return rc;
}
