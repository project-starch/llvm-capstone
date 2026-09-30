#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "signals.h"
#include <errno.h>
#include <string.h>
#include <sys/syscall.h>
#include <ucontext.h>
#include <unistd.h>

static struct capstone_signal_state *active;

#if defined(__riscv) && __riscv_xlen == 64
extern char capstone_stub_begin[], capstone_stub_ecall[], capstone_stub_after[], capstone_stub_retry[];
long capstone_stub_syscall(long nr, const long args[6], const _Atomic uint64_t *head,
                           const uint64_t *tail, int can_publish);
#endif

static uint64_t bit(int signo) { return signo >= 1 && signo <= 64 ? UINT64_C(1) << (signo - 1) : 0; }

static uint64_t mask_of(const sigset_t *set) {
  uint64_t m = 0;
  for (int s = 1; s <= CAPSTONE_SIGNAL_MAX; ++s)
    if (sigismember(set, s) == 1) m |= bit(s);
  return m;
}

static void set_of(uint64_t m, sigset_t *set) {
  sigemptyset(set);
  for (int s = 1; s <= CAPSTONE_SIGNAL_MAX; ++s)
    if (m & bit(s)) sigaddset(set, s);
}

/* Backpressure covers caught signals, including standard signals installed
 * with SA_NODEFER. Linux retains their pending records while the ring drains. */
static uint64_t caught_mask(const struct capstone_signal_state *s) {
  uint64_t mask = 0;
  for (int sig = 1; sig <= CAPSTONE_SIGNAL_MAX; ++sig)
    if (s->class[sig] == CAPSTONE_SIGNAL_CAUGHT) mask |= bit(sig);
  return mask;
}

/* The kernel mask is the union; SIGKILL and SIGSTOP the kernel drops itself. */
static void apply(struct capstone_signal_state *s) {
  sigset_t set;
  set_of(s->logical | atomic_load(&s->inflight) | s->backpressure, &set);
  sigprocmask(SIG_SETMASK, &set, NULL);
}

static int fatal_from_kernel(int signo, const siginfo_t *si) {
  return (signo == SIGSEGV || signo == SIGBUS || signo == SIGILL || signo == SIGFPE) &&
         si->si_code > 0;
}

/* Async-signal-safe: plain stores, one atomic increment, sigaction on the
 * fatal path only. The main flow never writes ring[head]. */
static void trampoline(int signo, siginfo_t *si, void *context) {
  struct capstone_signal_state *s = active;
  ucontext_t *uc = context;
  if (!s || signo < 1 || signo > CAPSTONE_SIGNAL_MAX) return;
  if (fatal_from_kernel(signo, si)) {
    /* The launcher itself faulted: not the domain's signal. Die as before. */
    struct sigaction dfl = {.sa_handler = SIG_DFL};
    sigaction(signo, &dfl, NULL);
    return;
  }
  uint64_t head = atomic_load(&s->head), tail = s->tail;
  int realtime = signo >= SIGRTMIN;
  uintptr_t pc = 0;
#if defined(__riscv) && __riscv_xlen == 64
  pc = uc->uc_mcontext.__gregs[0];
#endif
  if (!realtime) {
    /* a standard signal coalesces: the first pending event is the one kept */
    for (uint64_t i = tail; i < head; ++i)
      if (s->ring[i % CAPSTONE_SIGNAL_RING].signo == (uint32_t)signo) {
        if (!(s->flags[signo] & SA_NODEFER)) sigaddset(&uc->uc_sigmask, signo);
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
  int in_wait = s->wait_active;
#if defined(__riscv) && __riscv_xlen == 64
  in_wait = in_wait && pc >= (uintptr_t)capstone_stub_begin && pc <= (uintptr_t)capstone_stub_after;
#endif
  ev->seq = head + 1;
  ev->signo = (uint32_t)signo;
  ev->flags = (in_wait ? CAPSTONE_SIGNAL_WAIT : 0) |
              ((s->flags[signo] & SA_NODEFER) ? 0 : CAPSTONE_SIGNAL_DEFER);
  ev->mask = in_wait ? s->wait_mask : mask_of(&uc->uc_sigmask);
  ev->generation = s->generation[signo];
  memcpy(ev->info, si, sizeof ev->info < sizeof *si ? sizeof ev->info : sizeof *si);
  if (!(s->flags[signo] & SA_NODEFER)) {
    sigaddset(&uc->uc_sigmask, signo);
    atomic_fetch_or(&s->inflight, bit(signo));
  }
  atomic_store(&s->head, head + 1);
  if (s->block) s->block->recorded = head + 1;
  if (head + 1 - tail >= CAPSTONE_SIGNAL_BACKPRESSURE) {
    s->backpressure = caught_mask(s);
    for (int sig = 1; sig <= CAPSTONE_SIGNAL_MAX; ++sig)
      if (s->backpressure & bit(sig)) sigaddset(&uc->uc_sigmask, sig);
  }
redirect:
#if defined(__riscv) && __riscv_xlen == 64
  if (s->open_count < CAPSTONE_SIGNAL_RING &&
      pc >= (uintptr_t)capstone_stub_begin && pc <= (uintptr_t)capstone_stub_ecall)
    uc->uc_mcontext.__gregs[0] = (uintptr_t)capstone_stub_retry;
#endif
  (void)pc;
}

void capstone_signals_init(struct capstone_signal_state *s, struct capstone_signal_block *block) {
  memset(s, 0, sizeof *s);
  s->block = block;
  if (block) memset(block, 0, sizeof *block);
  sigset_t now;
  sigprocmask(SIG_BLOCK, NULL, &now);
  s->logical = mask_of(&now);
  for (int sig = 1; sig <= CAPSTONE_SIGNAL_MAX; ++sig) {
    struct sigaction action;
    if (sigaction(sig, NULL, &action) == 0 && action.sa_handler == SIG_IGN)
      s->class[sig] = CAPSTONE_SIGNAL_IGNORE;
  }
  if (block) {
    block->initial_mask = s->logical;
    block->initial_ignored = capstone_signals_ignored(s);
  }
  active = s;
}

long capstone_signals_action(struct capstone_signal_state *s, int signo, unsigned cls,
                             unsigned flags) {
  struct sigaction sa;
  if (signo < 1 || signo > CAPSTONE_SIGNAL_MAX || signo == SIGKILL || signo == SIGSTOP ||
      cls > CAPSTONE_SIGNAL_CAUGHT)
    return -EINVAL;
  memset(&sa, 0, sizeof sa);
  if (cls == CAPSTONE_SIGNAL_CAUGHT) {
    sa.sa_sigaction = trampoline;
    /* Linux decides restart against EINTR, resets a one-shot disposition and
       applies the child options; the trampoline records under all signals
       blocked so that recording is atomic. */
    sa.sa_flags = SA_SIGINFO |
        (int)(flags & (SA_RESTART | SA_RESETHAND | SA_NOCLDSTOP | SA_NOCLDWAIT));
    sigfillset(&sa.sa_mask);
  } else {
    sa.sa_handler = cls == CAPSTONE_SIGNAL_IGNORE ? SIG_IGN : SIG_DFL;
    sa.sa_flags = (int)(flags & (SA_NOCLDSTOP | SA_NOCLDWAIT));
  }
  if (sigaction(signo, &sa, NULL))
    return -errno;
  s->class[signo] = (uint8_t)cls;
  s->flags[signo] = flags;
  ++s->generation[signo];
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
    if (!(s->flags[ev->signo] & SA_RESTART)) return 0;
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

void capstone_signals_logical_set(const struct capstone_signal_state *s, sigset_t *set) {
  set_of(s->logical, set);
}

uint64_t capstone_signals_ignored(const struct capstone_signal_state *s) {
  uint64_t m = 0;
  for (int sig = 1; sig <= CAPSTONE_SIGNAL_MAX; ++sig)
    if (s->class[sig] == CAPSTONE_SIGNAL_IGNORE) m |= bit(sig);
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
