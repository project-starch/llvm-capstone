#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "signals.h"
#include <errno.h>
#include <pthread.h>
#include <string.h>
#include <sys/syscall.h>
#include <ucontext.h>
#include <unistd.h>

/* The process's dispositions; a state never initialized (all zero) has its
   own, all default. */
static struct capstone_signal_table *table_of(const struct capstone_signal_state *s) {
  return s->table ? s->table : (struct capstone_signal_table *)&s->own;
}

/* The state of the context this thread serves (capstone_signals_attach). */
static __thread struct capstone_signal_state *active;

/* Signals a launcher thread took before it attached: only the C library's
   own two can reach it then (a new thread unblocks them; every other signal
   is blocked from its creation), and they wait here, blocked, until attach
   records them in the context's ring. No system call: the thread's seccomp
   filter allows none that would put them back. */
#define EARLY_SIGNALS 8
static __thread struct { int signo; siginfo_t info; } early[EARLY_SIGNALS];
static __thread unsigned early_count;

#if defined(__riscv) && __riscv_xlen == 64
extern char capstone_stub_begin[], capstone_stub_ecall[], capstone_stub_after[], capstone_stub_retry[];
long capstone_stub_syscall(long nr, const long args[6], const _Atomic uint64_t *head,
                           const uint64_t *tail, int can_publish);
#endif

static uint64_t bit(int signo) { return signo >= 1 && signo <= 64 ? UINT64_C(1) << (signo - 1) : 0; }

/* A signal into a sigset the kernel reads: glibc's sigaddset refuses its
   internal signals (32 and 33), which a domain's musl uses. Signals 1 to 64
   are the set's first word on Linux. */
static void mask_add(sigset_t *set, int signo) {
  uint64_t m;
  memcpy(&m, set, sizeof m);
  m |= bit(signo);
  memcpy(set, &m, sizeof m);
}

static uint64_t mask_of(const sigset_t *set) {
  uint64_t m = 0;
  for (int s = 1; s <= CAPSTONE_SIGNAL_MAX; ++s)
    if (sigismember(set, s) == 1) m |= bit(s);
  return m;
}

#if !(defined(__riscv) && __riscv_xlen == 64)   /* the native tests' mask and actions */
static void set_of(uint64_t m, sigset_t *set) {
  sigemptyset(set);
  for (int s = 1; s <= CAPSTONE_SIGNAL_MAX; ++s)
    if (m & bit(s)) sigaddset(set, s);
}
#endif

/* Backpressure covers caught signals, including standard signals installed
 * with SA_NODEFER. Linux retains their pending records while the ring drains. */
static uint64_t caught_mask(const struct capstone_signal_state *s) {
  uint64_t mask = 0;
  for (int sig = 1; sig <= CAPSTONE_SIGNAL_MAX; ++sig)
    if (table_of(s)->class[sig] == CAPSTONE_SIGNAL_CAUGHT) mask |= bit(sig);
  return mask;
}

/* The calling thread's kernel mask. The C library filters its own internal
 * signals (32 and 33 in glibc) out of every mask it sets, and a domain's musl
 * uses them (its timer and cancel signals), so on the guest the mask goes to
 * the kernel directly; the launcher uses neither glibc feature behind them. */
uint64_t capstone_signals_set_kernel_mask(uint64_t mask) {
  uint64_t old = 0;
#if defined(__riscv) && __riscv_xlen == 64
  syscall(SYS_rt_sigprocmask, SIG_SETMASK, &mask, &old, sizeof mask);
#else
  sigset_t set, previous;
  set_of(mask, &set);
  pthread_sigmask(SIG_SETMASK, &set, &previous);
  old = mask_of(&previous);
#endif
  return old;
}

/* The kernel mask is the union; SIGKILL and SIGSTOP the kernel drops itself. */
static void apply(struct capstone_signal_state *s) {
  capstone_signals_set_kernel_mask(s->logical | atomic_load(&s->inflight) | s->backpressure);
}

/* The kernel's sigaction on the guest, for the same reason as the mask: glibc
 * refuses its internal signals. RISC-V has no sa_restorer; the kernel returns
 * through its vDSO. */
static int set_action(int signo, void (*sigaction_fn)(int, siginfo_t *, void *), void *disposition,
                      unsigned long flags, uint64_t mask) {
#if defined(__riscv) && __riscv_xlen == 64
  struct { void *handler; unsigned long flags; uint64_t mask; } k = {
      sigaction_fn ? (void *)sigaction_fn : disposition, flags, mask};
  return syscall(SYS_rt_sigaction, signo, &k, NULL, sizeof k.mask) ? -errno : 0;
#else
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  if (sigaction_fn) sa.sa_sigaction = sigaction_fn;
  else sa.sa_handler = (void (*)(int))disposition;
  sa.sa_flags = (int)flags;
  set_of(mask, &sa.sa_mask);
  return sigaction(signo, &sa, NULL) ? -errno : 0;
#endif
}

static int fatal_from_kernel(int signo, const siginfo_t *si) {
  return (signo == SIGSEGV || signo == SIGBUS || signo == SIGILL || signo == SIGFPE) &&
         si->si_code > 0;
}

/* Record an accepted signal in s's ring. uc is the interrupted context,
 * whose mask on return blocks the signal while its handler is due (NULL at
 * attach, which applies the thread's mask itself afterwards). */
static void record(struct capstone_signal_state *s, int signo, const siginfo_t *si, ucontext_t *uc) {
  uint64_t head = atomic_load(&s->head), tail = s->tail;
  uint32_t flags = table_of(s)->flags[signo];   /* once: another thread may change it */
  int realtime = signo >= SIGRTMIN;
  uintptr_t pc = 0;
#if defined(__riscv) && __riscv_xlen == 64
  if (uc) pc = uc->uc_mcontext.__gregs[0];
#endif
  if (!realtime) {
    /* a standard signal coalesces: the first pending event is the one kept */
    for (uint64_t i = tail; i < head; ++i)
      if (s->ring[i % CAPSTONE_SIGNAL_RING].signo == (uint32_t)signo) {
        if (uc && !(flags & SA_NODEFER)) mask_add(&uc->uc_sigmask, signo);
        goto redirect;
      }
  }
  struct capstone_signal_event *ev;
  if (head - tail >= CAPSTONE_SIGNAL_RING) {
    /* An in-progress temporary-mask wait may accept one last event. Preserve
       its original siginfo in the spare slot and block caught signals before
       returning to the interrupted code. A second spill means our mask
       invariant failed; terminate instead of losing or fabricating an event. */
    if (s->overflow_valid) {
      syscall(SYS_exit_group, 125);
      __builtin_trap();
    }
    s->overflow_valid = 1;
    ++s->spilled;
    ev = &s->overflow;
  } else {
    ev = &s->ring[head % CAPSTONE_SIGNAL_RING];
  }
  int in_wait = uc && s->wait_active;
#if defined(__riscv) && __riscv_xlen == 64
  in_wait = in_wait && pc >= (uintptr_t)capstone_stub_begin && pc <= (uintptr_t)capstone_stub_after;
#endif
  ev->seq = head + 1;
  ev->signo = (uint32_t)signo;
  ev->flags = (in_wait ? CAPSTONE_SIGNAL_WAIT : 0) |
              ((flags & SA_NODEFER) ? 0 : CAPSTONE_SIGNAL_DEFER);
  ev->mask = in_wait ? s->wait_mask : uc ? mask_of(&uc->uc_sigmask) : s->logical;
  ev->generation = table_of(s)->generation[signo];
  memcpy(ev->info, si, sizeof ev->info < sizeof *si ? sizeof ev->info : sizeof *si);
  if (!(flags & SA_NODEFER)) {
    if (uc) mask_add(&uc->uc_sigmask, signo);
    atomic_fetch_or(&s->inflight, bit(signo));
  }
  atomic_store(&s->head, head + 1);
  if (s->block) s->block->recorded = head + 1;
  if (head + 1 - tail >= CAPSTONE_SIGNAL_BACKPRESSURE) {
    s->backpressure = caught_mask(s);
    if (uc)
      for (int sig = 1; sig <= CAPSTONE_SIGNAL_MAX; ++sig)
        if (s->backpressure & bit(sig)) mask_add(&uc->uc_sigmask, sig);
  }
redirect:
#if defined(__riscv) && __riscv_xlen == 64
  if (uc && s->open_count < CAPSTONE_SIGNAL_RING &&
      pc >= (uintptr_t)capstone_stub_begin && pc <= (uintptr_t)capstone_stub_ecall)
    uc->uc_mcontext.__gregs[0] = (uintptr_t)capstone_stub_retry;
#endif
  (void)pc;
}

/* Async-signal-safe: plain stores, atomic increments, sigaction on the fatal
 * path only. The main flow never writes ring[head]. */
static void trampoline(int signo, siginfo_t *si, void *context) {
  struct capstone_signal_state *s = active;
  ucontext_t *uc = context;
  if (signo < 1 || signo > CAPSTONE_SIGNAL_MAX) return;
  if (!s) {
    mask_add(&uc->uc_sigmask, signo);
    if (early_count == EARLY_SIGNALS) {
      syscall(SYS_exit_group, 125);   /* the invariant above failed */
      __builtin_trap();
    }
    early[early_count].signo = signo;
    early[early_count].info = *si;
    ++early_count;
    return;
  }
  if (fatal_from_kernel(signo, si)) {
    /* The launcher itself faulted: not the domain's signal. Die as before. */
    struct sigaction dfl = {.sa_handler = SIG_DFL};
    sigaction(signo, &dfl, NULL);
    return;
  }
  record(s, signo, si, uc);
}

/* Whether sig's disposition is SIG_IGN now: the kernel's own answer, since
   glibc refuses to report its internal signals. */
static int ignored_now(int sig) {
#if defined(__riscv) && __riscv_xlen == 64
  struct { void *handler; unsigned long flags; uint64_t mask; } k;
  return syscall(SYS_rt_sigaction, sig, NULL, &k, sizeof k.mask) == 0 && k.handler == (void *)SIG_IGN;
#else
  struct sigaction action;
  return sigaction(sig, NULL, &action) == 0 && action.sa_handler == SIG_IGN;
#endif
}

/* glibc sets up its internal signals once per process, at its first
   pthread_create, and installs its own handler on 33 then, over whatever the
   domain installed. Have it happen now, before any domain disposition, and
   put back what the process was started with. */
static void *nothing(void *arg) { return arg; }
void capstone_signals_settle_libc(void) {
#if defined(__riscv) && __riscv_xlen == 64
  struct { void *handler; unsigned long flags; uint64_t mask; } k[2];
  int got[2];
  for (int i = 0; i < 2; ++i)
    got[i] = syscall(SYS_rt_sigaction, 32 + i, NULL, &k[i], sizeof k[i].mask) == 0;
  pthread_t t;
  if (!pthread_create(&t, NULL, nothing, NULL))
    pthread_join(t, NULL);
  for (int i = 0; i < 2; ++i)
    if (got[i])
      syscall(SYS_rt_sigaction, 32 + i, &k[i], NULL, sizeof k[i].mask);
#endif
}

void capstone_signals_init(struct capstone_signal_state *s, struct capstone_signal_block *block) {
  memset(s, 0, sizeof *s);
  s->table = &s->own;
  s->block = block;
  if (block) memset(block, 0, sizeof *block);
  sigset_t now;
  sigprocmask(SIG_BLOCK, NULL, &now);
  s->logical = mask_of(&now);
  for (int sig = 1; sig <= CAPSTONE_SIGNAL_MAX; ++sig)
    if (ignored_now(sig))
      table_of(s)->class[sig] = CAPSTONE_SIGNAL_IGNORE;
  if (block) {
    block->initial_mask = s->logical;
    block->initial_ignored = capstone_signals_ignored(s);
  }
  active = s;
}

void capstone_signals_init_context(struct capstone_signal_state *s, struct capstone_signal_block *block,
                                   const struct capstone_signal_state *creator) {
  memset(s, 0, sizeof *s);
  s->table = table_of(creator);
  s->block = block;
  s->logical = creator->logical;
  if (block) {
    /* the table is the first context's to install; a further one takes the
       mask only (the ignored set is not read under the owner's lock here) */
    memset(block, 0, sizeof *block);
    block->initial_mask = s->logical;
  }
}

void capstone_signals_attach(struct capstone_signal_state *s) {
  /* no trampoline on this thread while the early ones move (raw: glibc's
     mask calls would leave its own two open) */
  capstone_signals_set_kernel_mask(~UINT64_C(0));
  active = s;
  for (unsigned i = 0; i < early_count; ++i)
    record(s, early[i].signo, &early[i].info, NULL);
  early_count = 0;
  apply(s);
}

void capstone_signals_detach(void) {
  capstone_signals_set_kernel_mask(~UINT64_C(0));
  active = NULL;
}

long capstone_signals_action(struct capstone_signal_state *s, int signo, unsigned cls,
                             unsigned flags) {
  long r;
  if (signo < 1 || signo > CAPSTONE_SIGNAL_MAX || signo == SIGKILL || signo == SIGSTOP ||
      cls > CAPSTONE_SIGNAL_CAUGHT)
    return -EINVAL;
  if (cls == CAPSTONE_SIGNAL_CAUGHT)
    /* Linux decides restart against EINTR, resets a one-shot disposition and
       applies the child options; the trampoline records under all signals
       blocked so that recording is atomic. */
    r = set_action(signo, trampoline, NULL,
                   SA_SIGINFO | (flags & (SA_RESTART | SA_RESETHAND | SA_NOCLDSTOP | SA_NOCLDWAIT)),
                   ~UINT64_C(0));
  else
    r = set_action(signo, NULL, cls == CAPSTONE_SIGNAL_IGNORE ? (void *)SIG_IGN : (void *)SIG_DFL,
                   flags & (SA_NOCLDSTOP | SA_NOCLDWAIT), 0);
  if (r)
    return r;
  table_of(s)->class[signo] = (uint8_t)cls;
  table_of(s)->flags[signo] = flags;
  ++table_of(s)->generation[signo];
  if (s->backpressure) {
    s->backpressure = caught_mask(s);
    apply(s);
  }
  return 0;
}

long capstone_signals_procmask(struct capstone_signal_state *s, int how, const uint64_t *set,
                               uint64_t *old) {
  uint64_t previous = s->logical;
  if (set) {
    uint64_t m = *set & ~(bit(SIGKILL) | bit(SIGSTOP));
    switch (how) {
    case SIG_BLOCK: s->logical |= m; break;
    case SIG_UNBLOCK: s->logical &= ~m; break;
    case SIG_SETMASK: s->logical = m; break;
    default: return -EINVAL;
    }
    apply(s);
  }
  if (old) *old = previous;
  return 0;
}

long capstone_signals_done(struct capstone_signal_state *s, uint64_t seq) {
  int signo = 0;
  int deferred = 0;
  for (unsigned i = 0; i < s->open_count; ++i)
    if (s->open[i].seq == seq) {
      signo = s->open[i].signo;
      deferred = s->open[i].deferred;
      s->open[i] = s->open[--s->open_count];
      break;
    }
  if (!signo)
    return -EINVAL;
  if (!deferred) return 0;
  /* another deferred event of the same signal keeps it blocked */
  for (unsigned i = 0; i < s->open_count; ++i)
    if (s->open[i].signo == signo && s->open[i].deferred) return 0;
  atomic_fetch_and(&s->inflight, ~bit(signo));
  apply(s);
  return 0;
}

int capstone_signals_waiting(const struct capstone_signal_state *s) {
  return atomic_load(&((struct capstone_signal_state *)s)->head) != s->tail;
}

int capstone_signals_restartable(const struct capstone_signal_state *s) {
  uint64_t head = atomic_load(&((struct capstone_signal_state *)s)->head);
  for (uint64_t i = s->tail; i < head; ++i) {
    const struct capstone_signal_event *ev = &s->ring[i % CAPSTONE_SIGNAL_RING];
    if (!(table_of(s)->flags[ev->signo] & SA_RESTART)) return 0;
  }
  return head != s->tail;
}

void capstone_signals_publish(struct capstone_signal_state *s, struct capstone_delegate_entry *entry,
                              int retry) {
  uint64_t head = atomic_load(&s->head), pending = 0;
  unsigned count = 0;
  if (s->block) {
    while (s->tail < head && count < CAPSTONE_SIGNAL_EVENTS && s->open_count < CAPSTONE_SIGNAL_RING) {
      const struct capstone_signal_event *ev =
          s->overflow_valid && s->overflow.seq == s->tail + 1
              ? &s->overflow : &s->ring[s->tail % CAPSTONE_SIGNAL_RING];
      s->block->events[count++] = *ev;
      pending |= bit((int)ev->signo);
      s->open[s->open_count++] = (typeof(s->open[0])){
          ev->seq, (int)ev->signo, !!(ev->flags & CAPSTONE_SIGNAL_DEFER)};
      if (ev == &s->overflow) s->overflow_valid = 0;
      ++s->tail;
    }
    s->block->count = count;
    s->block->published = s->tail;
  }
  entry->pending = pending;
  entry->status = retry ? CAPSTONE_ROUND_RETRY : CAPSTONE_ROUND_DONE;
  /* Release pending kernel signals as the ring drains. A trampoline may have
     installed this mask between rounds before publication could run. */
  uint64_t want = head - s->tail >= CAPSTONE_SIGNAL_BACKPRESSURE ? caught_mask(s) : 0;
  if (want != s->backpressure) {
    s->backpressure = want;
    apply(s);
  }
}

uint64_t capstone_signals_ignored(const struct capstone_signal_state *s) {
  uint64_t m = 0;
  for (int sig = 1; sig <= CAPSTONE_SIGNAL_MAX; ++sig)
    if (table_of(s)->class[sig] == CAPSTONE_SIGNAL_IGNORE) m |= bit(sig);
  return m;
}

long capstone_signals_call(struct capstone_signal_state *s, long nr, const long args[6],
                           int has_wait_mask, uint64_t wait_mask) {
  long r;
  s->wait_active = has_wait_mask;
  s->wait_mask = wait_mask;
#if defined(__riscv) && __riscv_xlen == 64
  r = capstone_stub_syscall(nr, args, &s->head, &s->tail,
                             s->open_count < CAPSTONE_SIGNAL_RING);
#else
  /* Native tests: the ring check without the program-counter redirect. */
  if (s->open_count < CAPSTONE_SIGNAL_RING && capstone_signals_waiting(s))
    r = CAPSTONE_STUB_RETRY;
  else {
    r = syscall(nr, args[0], args[1], args[2], args[3], args[4], args[5]);
    if (r == -1) r = -errno;
  }
#endif
  s->wait_active = 0;
  return r;
}
