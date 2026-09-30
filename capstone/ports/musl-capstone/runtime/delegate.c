/* The delegated syscall stub: musl's syscall entry under CAPSTONE_DELEGATE_RUNTIME.
 *
 * Every Linux call becomes one entry in the entry region and one yield. Pointer
 * arguments are copied into the exchange region and cross as offsets; nothing
 * that names domain memory ever leaves the domain. The launcher runs the real
 * syscall and writes the result back. The three exception groups never cross:
 * memory is the domain allocator's, processes are not this branch's, signals
 * are a table here and a mask there. See docs/plans/delegation-abi.md.
 */
#include "capstone/delegate.h"
#include "capstone/spawn.h"
#include "capstone/launch.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <signal.h>
#include <time.h>
#include <capstone/lock.h>

typedef void *syscall_arg_t;
/* Integers travel in pointer-typed argument slots, as musl's syscall_arch.h
   states; the casts are the port's convention, not an accident. */
#pragma clang diagnostic ignored "-Wint-to-void-pointer-cast"
#pragma clang diagnostic ignored "-Wvoid-pointer-to-int-cast"
#pragma clang diagnostic ignored "-Wcapstone-pointer-roundtrip"
extern void __capstone_yield(void);
extern int __capstone_at_exit(int status);
void __capstone_hc_note_unserved(long n);
void __capstone_hc_note_noop(long n);
void __capstone_hc_report_unserved(void);
long __capstone_delegate_spawn(const void *block, unsigned long bytes);
const struct capstone_launch_task *__capstone_launch_task(void);
/* signals.c */
struct k_sigaction;
void __capstone_signals_regions(void *meta, unsigned long bytes);
int __capstone_signals_hint(void);
void __capstone_signals_take(void);
void __capstone_signals_deliver(void);
void __capstone_sigmask_note(int how, uint64_t m);
long __capstone_sigaction(int sig, const struct k_sigaction *new, struct k_sigaction *old);
long __capstone_sigprocmask(int how, const sigset_t *set, sigset_t *old, unsigned long size);
long __capstone_sigaltstack(const stack_t *ss, stack_t *old);
void __capstone_siginfo_translate(const unsigned char *raw, int sig, siginfo_t *si);

/* Answered without a round, from the launch record (see launch.h): the task's
 * identity, which cannot change under a domain, and the two clocks as rdtime
 * since the launcher read them, the way a vDSO answers. Neither invents state:
 * the values are Linux's, read once by the task itself. Every other clock, and
 * everything when the record is missing, is a round. */
static int dl_clock(long clock, struct timespec *ts) {
  const struct capstone_launch_task *t = __capstone_launch_task();
  uint64_t base, now, elapsed, ns;
  if (!t || !t->ticks_per_second || !ts)
    return 0;
  switch (clock) {
  case CLOCK_REALTIME: case CLOCK_REALTIME_COARSE:
    base = t->realtime_ns;
    break;
  case CLOCK_MONOTONIC: case CLOCK_MONOTONIC_COARSE: case CLOCK_MONOTONIC_RAW:
  case CLOCK_BOOTTIME:
    base = t->monotonic_ns;
    break;
  default:
    return 0;
  }
  __asm__ volatile("rdtime %0" : "=r"(now));
  elapsed = now - t->ticks;
  ns = base + elapsed / t->ticks_per_second * 1000000000u +
       elapsed % t->ticks_per_second * 1000000000u / t->ticks_per_second;
  ts->tv_sec = (time_t)(ns / 1000000000u);
  ts->tv_nsec = (long)(ns % 1000000000u);
  return 1;
}

int __capstone_context_tid(void);   /* tls.c: the calling context's */
long __capstone_set_tid_address(volatile int *word);   /* context.c */
static __thread void *dl_robust_head;
static __thread size_t dl_robust_len;
long __capstone_thread_exit(int status);    /* context.c */
static long dl_identity(long n, long *answer) {
  const struct capstone_launch_task *t = __capstone_launch_task();
  if (!t || !t->pid)
    return 0;
  switch (n) {
  case SYS_getpid: *answer = t->pid; return 1;
  case SYS_gettid: *answer = __capstone_context_tid(); return 1;
  case SYS_getppid: *answer = t->ppid; return 1;
  case SYS_getuid: *answer = t->uid; return 1;
  case SYS_geteuid: *answer = t->euid; return 1;
  case SYS_getgid: *answer = t->gid; return 1;
  case SYS_getegid: *answer = t->egid; return 1;
  default: return 0;
  }
}

/* The transport of the calling context: its entry block and exchange region.
   Per context, in its TLS block (docs/plans/delegation-threads.md): the first
   context installs transport 0 when the regions arrive, a further context the
   transport its creator reserved, before its first delegated call. A context
   without one gets -EIO from every call. */
static __thread volatile struct capstone_delegate_entry *dl_entry;
static __thread char *dl_exchange;
static __thread size_t dl_capacity, dl_used;

/* Argument marshalling record for one call. */
struct dl_slot {
  void *domain;      /* the caller's buffer, or NULL */
  uint64_t offset;   /* where it went in the exchange region */
  size_t bytes;
  unsigned char copy_back;
};

static size_t cap_bytes(void *cap) {
  return (size_t)(__builtin_capstone_cap_get_end(cap) -
                  __builtin_capstone_cap_get_cursor(cap));
}

static __thread uint64_t dl_status;  /* of the last round: DONE or RETRY */

/* posix_spawn's and execve's one request block (capstone/lock.h, Q6). */
volatile int __capstone_spawn_lock;

/* The two regions every transport is cut from, for the whole application:
   META blocks of CAPSTONE_DELEGATE_META_BYTES and exchange slices of equal
   size, transport i at i times the block size in both. */
static char *dl_meta_region, *dl_data_region;
static size_t dl_transports, dl_slice_bytes;
/* The park table after the last META block: one generation word per bucket,
   written only by the launcher. */
static volatile uint64_t *dl_park_gen;

/* Install transport `index` for the calling context: bounded capabilities to
   its entry block and its exchange slice. -1 when there is no such transport. */
static int dl_install(size_t index) {
  if (index >= dl_transports)
    return -1;
  char *meta = dl_meta_region + index * CAPSTONE_DELEGATE_META_BYTES;
  char *data = dl_data_region + index * dl_slice_bytes;
  dl_entry = (volatile struct capstone_delegate_entry *)__builtin_capstone_cap_shrink(
      meta, meta, meta + CAPSTONE_DELEGATE_META_BYTES);
  dl_exchange = __builtin_capstone_cap_shrink(data, data, data + dl_slice_bytes);
  dl_capacity = dl_slice_bytes > 0x40000000u ? 0x40000000u : dl_slice_bytes;
  return 0;
}

/* The first context, when the launcher's regions have arrived: record them and
   take transport 0, whose META block also carries the signal handover. The
   transports are cut by the sizes this image declared (application.c): a
   region's capability may cover more, and one that covers less leaves the
   application without transports. */
void __capstone_application_transports(unsigned long *count, unsigned long *exchange_bytes);
void __capstone_delegate_regions(void *meta, void *exchange) {
  unsigned long count, slice;
  __capstone_application_transports(&count, &slice);
  dl_meta_region = meta;
  dl_data_region = exchange;
  dl_transports = count;
  dl_slice_bytes = slice;
  if (!meta || !exchange ||
      cap_bytes(meta) < count * CAPSTONE_DELEGATE_META_BYTES + CAPSTONE_PARK_BYTES ||
      cap_bytes(exchange) / slice < count)
    dl_transports = 0;
  if (dl_transports) {
    char *table = (char *)meta + count * CAPSTONE_DELEGATE_META_BYTES;
    dl_park_gen = (volatile uint64_t *)__builtin_capstone_cap_shrink(table, table,
                                                                     table + CAPSTONE_PARK_BYTES);
  }
  if (dl_install(0)) {
    dl_entry = 0;
    dl_exchange = 0;
    dl_capacity = 0;
    __capstone_signals_regions(0, 0);
    return;
  }
  __capstone_signals_regions((void *)dl_entry, CAPSTONE_DELEGATE_META_BYTES);
}

/* A further context, at its first entry: the transport its creator reserved,
   and with it the context's signal handover block (B8). */
void __capstone_signals_attach(void *meta);
int __capstone_delegate_transport(unsigned long index) {
  if (index == 0 || dl_install(index))
    return -1;
  __capstone_signals_attach((void *)dl_entry);
  return 0;
}

int __capstone_delegate_ready(void) {
  return dl_entry && dl_exchange && cap_bytes((void *)dl_entry) >= sizeof *dl_entry;
}

/* Offset 0 means NULL for an optional buffer, so no buffer ever lives there:
   the first 16 bytes of the exchange region stay unused. */
#define DL_FIRST 16
/* A call's attempt starts with no round made: a loop that ends before its
   round never sees an earlier round's RETRY. */
static void dl_reset(void) { dl_used = DL_FIRST; dl_status = CAPSTONE_ROUND_DONE; }

static int dl_alloc(size_t bytes, uint64_t *offset) {
  size_t aligned = (dl_used + 15) & ~(size_t)15;
  if (aligned > dl_capacity || bytes > dl_capacity - aligned)
    return -1;
  *offset = aligned;
  dl_used = aligned + bytes;
  return 0;
}

/* One round trip. args are already offsets where the shape says so. The
 * events the launcher published are taken here; they run in dl_settle, once
 * the caller has its output data back and the exchange region is free. */
static long dl_round(uint64_t nr, const uint64_t args[CAPSTONE_DELEGATE_ARGS]) {
  struct capstone_delegate_entry e;
  if (capstone_delegate_pack(&e, nr, args)) {
    dl_status = CAPSTONE_ROUND_DONE;
    return -EINVAL;
  }
  memcpy((void *)dl_entry, &e, sizeof e);
  __capstone_yield();
  dl_status = dl_entry->status;
  __capstone_signals_take();
  /* A RETRY round has no result of its own. When it is not issued again (a
     cancellation point whose thread is being cancelled, below), the call was
     interrupted. */
  return dl_status == CAPSTONE_ROUND_RETRY ? -EINTR : (long)dl_entry->result;
}

/* signals.c: cancellation points */
void __capstone_cp_result(void);
int __capstone_cp_cancelled(void);
int __capstone_cp_enter(void);
int __capstone_cp_leave(int saved);
int __capstone_cp_pause(void);

/* After a call's round, once its data is safe: deliver, and say whether the
 * call must be issued again because it has no result yet. A cancellation
 * point is not: its thread's cancellation handler abandoned it. */
static int dl_settle(void) {
  int retry = dl_status == CAPSTONE_ROUND_RETRY;
  if (!retry)
    __capstone_cp_result();
  __capstone_signals_deliver();
  if (retry && __capstone_cp_cancelled())
    return 0;
  return retry;
}

/* The same for the runtime's own requests (SIGPOLL, SIGDONE, ...), made
 * inside a call and never its result. */
static int dl_settle_own(void) {
  int retry = dl_status == CAPSTONE_ROUND_RETRY;
  __capstone_signals_deliver();
  return retry;
}

/* Runtime requests with integer arguments only. */
long __capstone_delegate_ints4(uint64_t nr, uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
  uint64_t args[CAPSTONE_DELEGATE_ARGS] = {a, b, c, d, 0, 0};
  long r;
  if (!__capstone_delegate_ready())
    return -EIO;
  do {
    dl_reset();
    r = dl_round(nr, args);
  } while (dl_settle_own());
  return r;
}

long __capstone_delegate_ints(uint64_t nr, uint64_t a, uint64_t b, uint64_t c) {
  return __capstone_delegate_ints4(nr, a, b, c, 0);
}

static long dl_string(const char *s, uint64_t *offset) {
  size_t n;
  if (!s)
    return -EFAULT;
  n = strlen(s) + 1;
  if (dl_alloc(n, offset))
    return -ENAMETOOLONG;
  memcpy(dl_exchange + *offset, s, n);
  return 0;
}

/* Buffers whose length is another argument are clamped to what fits: a short
 * read, write or listing is a result every caller already handles. */
static long dl_call_once(const struct capstone_delegate_shape *s, uint64_t nr,
                         syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS]) {
  uint64_t args[CAPSTONE_DELEGATE_ARGS];
  struct dl_slot slots[CAPSTONE_DELEGATE_ARGS] = {{0}};
  long result;
  dl_reset();
  dl_entry->nr = nr; /* named in a fault record if the copy below faults */
  for (unsigned i = 0; i < CAPSTONE_DELEGATE_ARGS; ++i)
    args[i] = (uint64_t)(unsigned long)raw[i];
  for (unsigned i = 0; i < s->argc; ++i) {
    const struct capstone_delegate_arg *a = &s->args[i];
    size_t bytes;
    int optional = a->kind == CAPSTONE_ARG_OPT_IN || a->kind == CAPSTONE_ARG_OPT_OUT ||
                   a->kind == CAPSTONE_ARG_OPT_INOUT;
    if (a->kind == CAPSTONE_ARG_INT)
      continue;
    if (a->kind == CAPSTONE_ARG_STR || a->kind == CAPSTONE_ARG_OPT_STR) {
      if (a->kind == CAPSTONE_ARG_OPT_STR && !raw[i]) { args[i] = 0; continue; }
      long rc = dl_string((const char *)raw[i], &args[i]);
      if (rc)
        return rc;
      continue;
    }
    if (!raw[i]) {
      if (optional) {
        args[i] = 0;
        continue;
      }
      if (!((a->length == CAPSTONE_LEN_ARG || a->length == CAPSTONE_LEN_ARG_SCALED) &&
            args[a->size] == 0))
        return -EFAULT;
    }
    size_t aligned = (dl_used + 15) & ~(size_t)15;
    if (aligned >= dl_capacity)
      return -ENOMEM;
    if (a->length == CAPSTONE_LEN_ARG && a->size < CAPSTONE_DELEGATE_ARGS &&
        args[a->size] > dl_capacity - aligned)
      args[a->size] = dl_capacity - aligned;
    {
      struct capstone_delegate_entry probe = {0};
      memcpy(probe.args, args, sizeof args);
      bytes = capstone_delegate_arg_bytes(s, &probe, i);
    }
    if (dl_alloc(bytes, &args[i]))
      return -ENOMEM;
    if (bytes && a->kind != CAPSTONE_ARG_OUT && a->kind != CAPSTONE_ARG_OPT_OUT)
      memcpy(dl_exchange + args[i], raw[i], bytes);
    slots[i].domain = raw[i];
    slots[i].offset = args[i];
    slots[i].bytes = bytes;
    slots[i].copy_back = a->kind != CAPSTONE_ARG_IN && a->kind != CAPSTONE_ARG_OPT_IN;
  }
  result = dl_round(nr, args);
  if (dl_status == CAPSTONE_ROUND_RETRY)
    return -EINTR;   /* no result, no output: issued again, unless cancelled (dl_settle) */
  for (unsigned i = 0; i < CAPSTONE_DELEGATE_ARGS; ++i)
    if (slots[i].domain && slots[i].copy_back) {
      size_t bytes = capstone_delegate_result_bytes(nr, i, slots[i].bytes, result);
      if (bytes) memcpy(slots[i].domain, dl_exchange + slots[i].offset, bytes);
    }
  return result;
}

static long dl_call(uint64_t nr, syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS]) {
  const struct capstone_delegate_shape *s = capstone_delegate_shape(nr);
  long result;
  if (!s)
    return -ENOSYS;
  do result = dl_call_once(s, nr, raw);
  while (dl_settle());
  return result;
}

/* futex (docs/plans/delegation-threads.md, "Parking"). The lock word stays in
 * the domain; the launcher only parks and wakes threads by key, the word's
 * address. FUTEX_WAIT reads the bucket's generation before it compares the
 * word, so a WAKE between the compare and the sleep moves the generation and
 * the launcher answers RECHECK instead of sleeping. A relative timeout becomes
 * one absolute CLOCK_MONOTONIC deadline at entry, kept across every round.
 * After a round that delivered signals (RETRY) the word is compared again, as
 * Linux does when it restarts the call. WAIT, WAKE, REQUEUE and the
 * priority-inheritance lock and unlock (below) are served; every other
 * operation (TRYLOCK_PI, WAKE_OP, CMP_REQUEUE, WAIT_BITSET, and any with the
 * realtime clock flag, which Linux refuses on these) answers ENOSYS and is
 * reported as unserved. */
#define DL_FUTEX_WAIT 0
#define DL_FUTEX_WAKE 1
#define DL_FUTEX_REQUEUE 3
#define DL_FUTEX_LOCK_PI 6
#define DL_FUTEX_UNLOCK_PI 7
#define DL_FUTEX_PRIVATE 128
#define DL_FUTEX_WAITERS 0x80000000u
#define DL_FUTEX_OWNER_DIED 0x40000000u
#define DL_FUTEX_TID_MASK 0x3fffffffu

/* Test only (thread-probe): runs after the compare and before the WAIT
   request when set, to hold a context in the window the generation protects
   across a quantum. */
void (*__capstone_futex_test_gap)(void);

static uint64_t dl_park_key(volatile int *word) {
  return (uint64_t)__builtin_capstone_cap_get_cursor((void *)word);
}

/* A word's key, for the launcher's wake at a context's end (context.c). */
uint64_t __capstone_park_key(volatile int *word) { return dl_park_key(word); }

/* The generation of word's bucket now (thread-probe's control case builds a
   wait with the order reversed from it). */
uint64_t __capstone_park_generation(volatile int *word) {
  return dl_park_gen ? dl_park_gen[capstone_park_bucket_of(dl_park_key(word), CAPSTONE_PARK_BUCKETS)]
                     : 0;
}

/* Sleep while *word == val, until deadline (absolute CLOCK_MONOTONIC
   nanoseconds, 0 for none): 0 (woken, or look again), -EAGAIN (the word
   differs), -ETIMEDOUT or -EINTR. */
static long dl_futex_wait_until(volatile int *word, int val, uint64_t deadline) {
  uint64_t key = dl_park_key(word);
  long r;
  for (;;) {
    /* acquire: the generation before the word, as the plan's Parking requires */
    uint64_t gen = __atomic_load_n(&dl_park_gen[capstone_park_bucket_of(key, CAPSTONE_PARK_BUCKETS)],
                                   __ATOMIC_ACQUIRE);
    if (*word != val)
      return -EAGAIN;
    if (__capstone_futex_test_gap)
      __capstone_futex_test_gap();
    uint64_t args[CAPSTONE_DELEGATE_ARGS] = {key, gen, deadline, 0, 0, 0};
    dl_reset();
    r = dl_round(CAPSTONE_NR_PARK_WAIT, args);
    if (!dl_settle())
      break;
  }
  return r == CAPSTONE_PARK_RESULT_WOKEN || r == CAPSTONE_PARK_RESULT_RECHECK ? 0 : r;
}

static long dl_futex_wait(volatile int *word, int val, const struct timespec *timeout) {
  uint64_t deadline = 0;
  if (timeout) {
    struct timespec now;
    if (timeout->tv_sec < 0 || timeout->tv_nsec < 0 || timeout->tv_nsec >= 1000000000L)
      return -EINVAL;
    int cp = __capstone_cp_pause();   /* the runtime's own read, not the call's round */
    int failed = clock_gettime(CLOCK_MONOTONIC, &now);
    (void)__capstone_cp_leave(cp);
    if (failed)
      return -errno;
    uint64_t at = (uint64_t)now.tv_sec * 1000000000u + (uint64_t)now.tv_nsec, span;
    if ((uint64_t)timeout->tv_sec >= UINT64_MAX / 1000000000u)
      span = UINT64_MAX;
    else
      span = (uint64_t)timeout->tv_sec * 1000000000u + (uint64_t)timeout->tv_nsec;
    deadline = span > UINT64_MAX - at ? UINT64_MAX : at + span;
    if (!deadline)
      deadline = 1;
  }
  return dl_futex_wait_until(word, val, deadline);
}

/* Priority-inheritance futexes, musl's PTHREAD_PRIO_INHERIT mutexes. The word
 * holds the owner's tid with FUTEX_WAITERS and FUTEX_OWNER_DIED, as on Linux,
 * and only the domain changes it; the launcher parks and wakes as for WAIT.
 * Two differences from Linux, neither visible to musl: an unlock frees the
 * word and wakes one waiter to take it, where Linux hands it over, so a
 * waiter that took it keeps FUTEX_WAITERS set for whoever may still sleep;
 * and no priority is inherited, because every context runs on a launcher
 * thread of the same priority. LOCK_PI's timeout is absolute CLOCK_REALTIME,
 * made one CLOCK_MONOTONIC deadline at entry; a signal restarts the wait, as
 * Linux restarts the call. */
static long dl_futex_lock_pi(volatile int *word, const struct timespec *at) {
  unsigned tid = (unsigned)__capstone_context_tid() & DL_FUTEX_TID_MASK;
  volatile unsigned *w = (volatile unsigned *)word;
  uint64_t deadline = 0;
  int waited = 0;
  if (at) {
    struct timespec real, mono;
    if (at->tv_nsec < 0 || at->tv_nsec >= 1000000000L)
      return -EINVAL;
    int cp = __capstone_cp_pause();
    int failed = clock_gettime(CLOCK_REALTIME, &real) || clock_gettime(CLOCK_MONOTONIC, &mono);
    (void)__capstone_cp_leave(cp);
    if (failed)
      return -errno;
    uint64_t now = (uint64_t)mono.tv_sec * 1000000000u + (uint64_t)mono.tv_nsec, left;
    if (at->tv_sec < real.tv_sec || (at->tv_sec == real.tv_sec && at->tv_nsec <= real.tv_nsec))
      left = 0;
    else if ((uint64_t)at->tv_sec - (uint64_t)real.tv_sec >= UINT64_MAX / 1000000000u - 1)
      left = UINT64_MAX;   /* saturates, as a relative WAIT's deadline does */
    else
      left = ((uint64_t)at->tv_sec - (uint64_t)real.tv_sec) * 1000000000u +
             (uint64_t)at->tv_nsec - (uint64_t)real.tv_nsec;
    deadline = !left ? 1 : left > UINT64_MAX - now ? UINT64_MAX : now + left;
  }
  for (;;) {
    unsigned v = __atomic_load_n(w, __ATOMIC_ACQUIRE);
    unsigned owner = v & DL_FUTEX_TID_MASK;
    if (owner == tid)
      return -EDEADLK;
    if (!owner) {
      unsigned mine = tid | (v & DL_FUTEX_OWNER_DIED) | (waited ? DL_FUTEX_WAITERS : 0);
      if (__atomic_compare_exchange_n(w, &v, mine, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
        return 0;
      continue;
    }
    if (!(v & DL_FUTEX_WAITERS)) {
      if (!__atomic_compare_exchange_n(w, &v, v | DL_FUTEX_WAITERS, 0, __ATOMIC_RELAXED,
                                       __ATOMIC_RELAXED))
        continue;
      v |= DL_FUTEX_WAITERS;
    }
    waited = 1;
    long r = dl_futex_wait_until(word, (int)v, deadline);
    if (r == -ETIMEDOUT || r == -ENOSYS || r == -EIO)
      return r;
  }
}

static long dl_futex_unlock_pi(volatile int *word) {
  unsigned tid = (unsigned)__capstone_context_tid() & DL_FUTEX_TID_MASK;
  volatile unsigned *w = (volatile unsigned *)word;
  unsigned v = __atomic_load_n(w, __ATOMIC_RELAXED);
  do {
    if ((v & DL_FUTEX_TID_MASK) != tid)
      return -EPERM;
  } while (!__atomic_compare_exchange_n(w, &v, 0, 0, __ATOMIC_RELEASE, __ATOMIC_RELAXED));
  if (v & DL_FUTEX_WAITERS)
    __capstone_delegate_ints(CAPSTONE_NR_PARK_WAKE, dl_park_key(word), 1, 0);
  return 0;
}

static long dl_futex(volatile int *word, long op, long val, void *arg4, volatile int *word2) {
  long cmd = op & ~(long)DL_FUTEX_PRIVATE;
  if (!dl_park_gen)
    return -ENOSYS;
  /* Linux's own argument checks, as far as the served operations reach */
  if (dl_park_key(word) & 3)
    return -EINVAL;
  switch (cmd) {
  case DL_FUTEX_WAIT:
    return dl_futex_wait(word, (int)val, (const struct timespec *)arg4);
  case DL_FUTEX_LOCK_PI:
    return dl_futex_lock_pi(word, (const struct timespec *)arg4);
  case DL_FUTEX_UNLOCK_PI:
    return dl_futex_unlock_pi(word);
  case DL_FUTEX_WAKE:
    /* Linux wakes one when asked for none or fewer */
    return __capstone_delegate_ints(CAPSTONE_NR_PARK_WAKE, dl_park_key(word),
                                    (int)val <= 0 ? 1 : (uint64_t)(int)val, 0);
  case DL_FUTEX_REQUEUE: {
    int nwake = (int)val, nmove = (int)(long)(uintptr_t)arg4;
    if (nwake < 0 || nmove < 0)
      return -EINVAL;
    if (dl_park_key(word2) & 3)
      return -EINVAL;
    uint64_t args[CAPSTONE_DELEGATE_ARGS] = {dl_park_key(word), dl_park_key(word2),
                                             (uint64_t)nwake, (uint64_t)nmove, 0, 0};
    long r;
    do {
      dl_reset();
      r = dl_round(CAPSTONE_NR_PARK_REQUEUE, args);
    } while (dl_settle());
    return r;
  }
  default:
    __capstone_hc_note_unserved(SYS_futex);
    return -ENOSYS;
  }
}

/* rt_sigtimedwait returns the kernel's 128-byte siginfo. Translate it into
 * musl's capability layout before a handler from this round can inspect the
 * caller's output buffer. */
static long dl_sigtimedwait(syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS]) {
  const struct capstone_delegate_shape *shape = capstone_delegate_shape(CAPSTONE_SYS_rt_sigtimedwait);
  siginfo_t *out = (siginfo_t *)raw[1];
  unsigned char wire[128];
  long result;
  raw[1] = out ? (syscall_arg_t)wire : 0;
  do {
    result = dl_call_once(shape, CAPSTONE_SYS_rt_sigtimedwait, raw);
    if (result > 0 && dl_status != CAPSTONE_ROUND_RETRY && out)
      __capstone_siginfo_translate(wire, (int)result, out);
  } while (dl_settle());
  return result;
}

/* rt_sigprocmask for signals.c: two eight-byte buffers, and the mirror. */
long __capstone_delegate_procmask(int how, const uint64_t *set, uint64_t *old) {
  uint64_t in = set ? *set : 0, out = 0;
  syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS] = {(syscall_arg_t)(long)how, set ? (syscall_arg_t)&in : 0,
                                               old ? (syscall_arg_t)&out : 0, (syscall_arg_t)8, 0, 0};
  long r = dl_call(CAPSTONE_SYS_rt_sigprocmask, raw);
  if (r) return r;
  if (set) __capstone_sigmask_note(how, in);
  if (old) *old = out;
  return 0;
}

/* One kernel vector operation preserves pipe atomicity and file offsets.
 * Nested pointers are wire offsets as well. A large vector may return a short
 * prefix; a small vector (including PIPE_BUF writes) fits in one round. */
static long dl_vector_once(long fd, const struct iovec *iov, long count, int writing,
                           int positioned, long long offset) {
  dl_reset();
  uint64_t args[6] = {(uint64_t)fd, 0, 0, (uint64_t)offset & UINT32_MAX,
                      (uint64_t)offset >> 32, 0};
  uint64_t offsets[1024];
  size_t lengths[1024], total = 0;
  if (count < 0 || count > 1024)
    return -EINVAL;
  if (count && !iov)
    return -EFAULT;
  for (long i = 0; i < count; ++i) {
    if (iov[i].iov_len > (size_t)LONG_MAX - total)
      return -EINVAL;
    total += iov[i].iov_len;
  }
  dl_reset();
  /* Reserve the entire descriptor array; oversized metadata is an error. */
  if (dl_alloc((size_t)count * 16, &args[1]))
    return -EMSGSIZE;
  for (long i = 0; i < count; ++i) {
    size_t aligned = (dl_used + 15) & ~(size_t)15;
    size_t room = aligned < dl_capacity ? dl_capacity - aligned : 0;
    size_t bytes = iov[i].iov_len < room ? iov[i].iov_len : room;
    if (!bytes && iov[i].iov_len)
      break;
    if (bytes && !iov[i].iov_base)
      return -EFAULT;
    if (dl_alloc(bytes, &offsets[i]))
      break;
    lengths[i] = bytes;
    uint64_t wire[2] = {offsets[i], bytes};
    memcpy(dl_exchange + args[1] + (size_t)i * 16, wire, sizeof wire);
    if (writing && bytes)
      memcpy(dl_exchange + offsets[i], iov[i].iov_base, bytes);
    ++args[2];
    if (bytes < iov[i].iov_len)
      break;
  }
  if (count && !args[2])
    return -EMSGSIZE;
  uint64_t nr = positioned ? (writing ? CAPSTONE_SYS_pwritev : CAPSTONE_SYS_preadv)
                           : (writing ? CAPSTONE_SYS_writev : CAPSTONE_SYS_readv);
  long result = dl_round(nr, args);
  if (dl_status == CAPSTONE_ROUND_RETRY)
    return -EINTR;   /* as dl_call_once */
  if (!writing && result > 0) {
    size_t left = (size_t)result;
    for (size_t i = 0; i < args[2] && left; ++i) {
      size_t n = lengths[i] < left ? lengths[i] : left;
      memcpy(iov[i].iov_base, dl_exchange + offsets[i], n);
      left -= n;
    }
  }
  return result;
}

static long dl_vector(long fd, const struct iovec *iov, long count, int writing,
                      int positioned, long long offset) {
  long result;
  do result = dl_vector_once(fd, iov, count, writing, positioned, offset);
  while (dl_settle());
  return result;
}

/* ioctl and fcntl carry a request-specific third argument. The ones a libc
 * uses are listed; anything else passes 0 and the kernel says EFAULT or
 * EINVAL, which is visible rather than a silent domain address. */
static long dl_ioctl(long fd, unsigned long request, void *argp) {
  syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS] = {(syscall_arg_t)fd, (syscall_arg_t)request, 0, 0, 0, 0};
  size_t bytes = 0;
  switch (request) {
  case TIOCGWINSZ: case TIOCSWINSZ: bytes = 8; break;
  case TCGETS: case TCSETS: case TCSETSW: case TCSETSF: bytes = 60; break;
  case FIONREAD: case FIONBIO: case FIOCLEX: case FIONCLEX: bytes = request == FIOCLEX || request == FIONCLEX ? 0 : 4; break;
  default: bytes = 0;
  }
  if (bytes && argp) {
    unsigned char buffer[64] = {0};
    long rc;
    memcpy(buffer, argp, bytes);
    raw[2] = buffer;
    dl_reset();
    {
      uint64_t args[CAPSTONE_DELEGATE_ARGS] = {(uint64_t)fd, request, 0, 0, 0, 0};
      do {
        dl_reset();
        if (dl_alloc(64, &args[2]))
          return -ENOMEM;
        memcpy(dl_exchange + args[2], buffer, sizeof buffer);
        rc = dl_round(CAPSTONE_NR_IOCTL_BUF, args);
        if (rc >= 0 && dl_status != CAPSTONE_ROUND_RETRY)
          memcpy(argp, dl_exchange + args[2], bytes);
      } while (dl_settle());
      return rc;
    }
  }
  if (bytes)
    return -EFAULT;
  if (request != FIOCLEX && request != FIONCLEX)
    return -ENOSYS;
  raw[2] = 0;
  return dl_call(CAPSTONE_SYS_ioctl, raw);
}

/* prctl serves a thread's name and nothing else: the launcher thread that
   serves this context sets or reads it on itself (THREAD_NAME). A name is read
   up to its NUL or 15 bytes, never past a short string. */
static long dl_prctl(long option, void *arg) {
  uint64_t args[CAPSTONE_DELEGATE_ARGS] = {0, 0, 0, 0, 0, 0};
  char name[CAPSTONE_THREAD_NAME_BYTES] = {0};
  long rc;
  if (option == PR_SET_NAME) {
    if (!arg)
      return -EFAULT;
    memcpy(name, arg, strnlen(arg, sizeof name - 1));
    args[0] = CAPSTONE_THREAD_NAME_SET;
  } else if (option == PR_GET_NAME) {
    if (!arg)
      return -EFAULT;
    args[0] = CAPSTONE_THREAD_NAME_GET;
  } else {
    __capstone_hc_note_unserved(SYS_prctl);
    return -ENOSYS;
  }
  do {
    dl_reset();
    if (dl_alloc(sizeof name, &args[1]))
      return -ENOMEM;
    memcpy(dl_exchange + args[1], name, sizeof name);
    rc = dl_round(CAPSTONE_NR_THREAD_NAME, args);
    if (rc >= 0 && option == PR_GET_NAME && dl_status != CAPSTONE_ROUND_RETRY)
      memcpy(arg, dl_exchange + args[1], sizeof name);
  } while (dl_settle());
  return rc;
}

static long dl_fcntl(long fd, long cmd, void *arg) {
  if (cmd == F_GETLK || cmd == F_SETLK || cmd == F_SETLKW) {
    uint64_t args[CAPSTONE_DELEGATE_ARGS] = {(uint64_t)fd, (uint64_t)cmd, 0, 0, 0, 0};
    long rc;
    if (!arg)
      return -EFAULT;
    do {
      dl_reset();
      if (dl_alloc(32, &args[2]))
        return -ENOMEM;
      memcpy(dl_exchange + args[2], arg, 32);
      rc = dl_round(CAPSTONE_NR_FCNTL_LOCK, args);
      if (rc >= 0 && cmd == F_GETLK && dl_status != CAPSTONE_ROUND_RETRY)
        memcpy(arg, dl_exchange + args[2], 32);
    } while (dl_settle());
    return rc;
  }
  {
    /* the integer commands: the value travels as the number it is */
    syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS] = {(syscall_arg_t)fd, (syscall_arg_t)cmd,
                                                 (syscall_arg_t)(unsigned long)arg, 0, 0, 0};
    return dl_call(CAPSTONE_SYS_fcntl, raw);
  }
}

long __capstone_delegate_hello(unsigned long entry_address, unsigned long code_base,
                               unsigned long code_end) {
  uint64_t args[CAPSTONE_DELEGATE_ARGS] = {entry_address, code_base, code_end, 0, 0, 0};
  long r;
  do r = dl_round(CAPSTONE_NR_HELLO, args);
  while (dl_settle());
  return r;
}

long __capstone_delegate_call(long n, syscall_arg_t a, syscall_arg_t b,
                              syscall_arg_t c, syscall_arg_t d,
                              syscall_arg_t e, syscall_arg_t f) {
  syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS] = {a, b, c, d, e, f};
  /* musl's thread setup: the tid is the context's runtime identity (the pid
     for the first context), and the word to clear at the context's end is the
     runtime's to keep (context.c). The first context asks before its
     transport exists. Neither reaches Linux. */
  if (n == SYS_set_tid_address)
    return __capstone_set_tid_address((volatile int *)a);
  /* One context ends (context.c); exit_group ends them all. A further
     context ends even when its transport could not be installed. */
  if (n == SYS_exit)
    return __capstone_thread_exit((int)(long)a);
  if (!__capstone_delegate_ready())
    return -EIO;
  /* The hint: the launcher's trampoline accepted a signal since the last
     round. Fetch it before this call, local answers included, so the handler
     runs at the next entry into this dispatcher and not at the next yield. */
  if (__capstone_signals_hint())
    __capstone_delegate_ints(CAPSTONE_NR_SIGPOLL, 0, 0, 0);
  /* a cancellation point cancelled before its call was made makes none */
  if (__capstone_cp_cancelled())
    return -EINTR;
  /* Signals: handlers are domain addresses and stay here; the class, the
     flags, the mask and the alternate stack are what crosses or what the
     domain keeps. */
  switch (n) {
  case SYS_rt_sigaction:
    if ((unsigned long)d != 8) return -EINVAL;
    return __capstone_sigaction((int)(long)a, (const struct k_sigaction *)b, (struct k_sigaction *)c);
  case SYS_rt_sigprocmask:
    return __capstone_sigprocmask((int)(long)a, (const sigset_t *)b, (sigset_t *)c, (unsigned long)d);
  case SYS_sigaltstack:
    return __capstone_sigaltstack((const stack_t *)a, (stack_t *)b);
  default:
    break;
  }
  switch (n) {
  case SYS_futex:
    return dl_futex((volatile int *)a, (long)b, (long)c, d, (volatile int *)e);
  case SYS_rt_sigtimedwait:
    return dl_sigtimedwait(raw);
  case SYS_readv:
    return dl_vector((long)a, (const struct iovec *)b, (long)c, 0, 0, 0);
  case SYS_writev:
    return dl_vector((long)a, (const struct iovec *)b, (long)c, 1, 0, 0);
  case SYS_preadv:
    return dl_vector((long)a, (const struct iovec *)b, (long)c, 0, 1,
                     (long long)((unsigned long)d | ((unsigned long)e << 32)));
  case SYS_pwritev:
    return dl_vector((long)a, (const struct iovec *)b, (long)c, 1, 1,
                     (long long)((unsigned long)d | ((unsigned long)e << 32)));
  case SYS_ioctl:
    return dl_ioctl((long)a, (unsigned long)b, c);
  case SYS_fcntl:
    return dl_fcntl((long)a, (long)b, c);
  case SYS_prctl:
    return dl_prctl((long)a, b);
  case SYS_execve: {
    /* exec in place: the task replaces itself with the named image. The
       block is static and one at a time, as posix_spawn's. */
    static char block[CAPSTONE_SPAWN_BYTES];
    size_t bytes;
    long rc;
    capstone_lock(&__capstone_spawn_lock);
    int error = capstone_spawn_pack(block, sizeof block, CAPSTONE_SPAWN_EXEC, 0, (const char *)a,
                                    (char *const *)b, (char *const *)c, NULL, 0, NULL, &bytes);
    rc = error ? -error : __capstone_delegate_spawn(block, bytes);
    capstone_unlock(&__capstone_spawn_lock);
    if (rc == -ENOSYS)
      __capstone_hc_note_unserved(n);
    return rc;
  }
  case SYS_getpid: case SYS_gettid: case SYS_getppid:
  case SYS_getuid: case SYS_geteuid: case SYS_getgid: case SYS_getegid: {
    long answer;
    if (dl_identity(n, &answer))
      return answer;
    break;
  }
  case SYS_clock_gettime:
    if (dl_clock((long)a, (struct timespec *)b))
      return 0;
    break;
  /* A thread's CPU set is the one of the Linux thread that serves its context:
     the caller's own, named by 0 or by its tid, and no other thread's. */
  case SYS_sched_getaffinity:
  case SYS_sched_setaffinity:
    if ((long)a && (long)a != __capstone_context_tid())
      return -ESRCH;
    raw[0] = 0;
    break;
  /* The robust list: musl walks a thread's list itself when the thread ends
     (pthread_exit); Linux's walk matters only when a whole process dies,
     which ends every context of a domain with it. So the head is kept here,
     per context, and answered to the calling context: musl asks for its own
     (pid 0) to learn that robust mutexes are supported. */
  case SYS_set_robust_list:
    dl_robust_head = (void *)a;
    dl_robust_len = (size_t)(unsigned long)b;
    return 0;
  case SYS_get_robust_list:
    if ((long)a && (long)a != __capstone_context_tid())
      return -ESRCH;
    if (!b || !c)
      return -EFAULT;
    *(void **)b = dl_robust_head;
    *(size_t *)c = dl_robust_len;
    return 0;
  case SYS_exit_group: {
    /* The program's last words before the task ends it: the at-exit hook,
       then the unserved report, then the real exit_group. Nothing resumes. */
    long status = __capstone_at_exit((int)(long)a);
    __capstone_hc_report_unserved();
    raw[0] = (syscall_arg_t)status;
    return dl_call(CAPSTONE_SYS_exit_group, raw);
  }
  default:
    break;
  }
  switch (capstone_delegate_group_of((uint64_t)n)) {
  case CAPSTONE_GROUP_DELEGATED: {
    long rc = dl_call((uint64_t)n, raw);
    if (rc == -ENOSYS)
      __capstone_hc_note_unserved(n);
    return rc;
  }
  case CAPSTONE_GROUP_SIGNAL:
    /* rt_sigreturn: no kernel frame exists here; handlers return to the
       dispatcher that invoked them */
    __capstone_hc_note_unserved(n);
    return -ENOSYS;
  case CAPSTONE_GROUP_MEMORY:
  case CAPSTONE_GROUP_PROCESS:
  case CAPSTONE_GROUP_RUNTIME:
  case CAPSTONE_GROUP_UNKNOWN:
  default:
    __capstone_hc_note_unserved(n);
    return -ENOSYS;
  }
}

/* posix_spawn's request: the packed block crosses as an IN buffer. */
long __capstone_delegate_spawn(const void *block, unsigned long bytes) {
  syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS] = {(syscall_arg_t)block, (syscall_arg_t)bytes, 0, 0, 0, 0};
  if (!__capstone_delegate_ready())
    return -EIO;
  if (bytes > dl_capacity - 16)
    return -E2BIG;
  return dl_call(CAPSTONE_NR_SPAWN, raw);
}

/* Context requests (docs/plans/delegation-threads.md): two integers in, an
   optional 48-byte event out. */
long __capstone_delegate_context(uint64_t nr, uint64_t a, uint64_t b, void *event) {
  syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS] = {(syscall_arg_t)a, (syscall_arg_t)b,
                                               (syscall_arg_t)event, 0, 0, 0};
  if (!__capstone_delegate_ready())
    return -EIO;
  return dl_call(nr, raw);
}

/* The unserved report's writer: two is the task's stderr. */
void __capstone_delegate_write2(const char *buf, unsigned long n) {
  syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS] = {(syscall_arg_t)2, (syscall_arg_t)buf,
                                               (syscall_arg_t)n, 0, 0, 0};
  while (n) {
    long k = dl_call(CAPSTONE_SYS_write, raw);
    if (k <= 0)
      return;
    buf += k;
    n -= (unsigned long)k;
    raw[1] = (syscall_arg_t)buf;
    raw[2] = (syscall_arg_t)n;
  }
}

/* musl's cancellation point (src/thread/pthread_cancel.c calls it for every
 * call that is one). Linked in only with pthread_cancel.o, which alone calls
 * it: __cancel is a weak reference. A pending cancellation is taken at entry,
 * as musl's own __cp_begin does; during the call signals.c shows a
 * cancellation handler where the call stands. */
__attribute__((__weak__)) long __cancel(void);
long __syscall_cp_asm(volatile int *cancel, syscall_arg_t nr, syscall_arg_t u, syscall_arg_t v,
                      syscall_arg_t w, syscall_arg_t x, syscall_arg_t y, syscall_arg_t z) {
  if (*cancel)
    return __cancel();
  int saved = __capstone_cp_enter();
  long r = __capstone_delegate_call((long)nr, u, v, w, x, y, z);
  /* abandoned for a cancellation: what musl's __cp_cancel does, whatever the
     call (a close included, which __syscall_cp_c does not cancel on EINTR) */
  if (__capstone_cp_leave(saved))
    return __cancel();
  return r;
}
