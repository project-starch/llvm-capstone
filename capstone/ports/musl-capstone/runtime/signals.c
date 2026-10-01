/* The domain's half of delegated signals: handlers live here, everything else
 * is Linux's. See docs/plans/delegation-signals.md.
 *
 * rt_sigaction becomes a SIGACTION request carrying a class, never a pointer;
 * rt_sigprocmask is delegated and mirrored here, so the domain always knows
 * its logical mask; every round brings the events Linux delivered to the task
 * meanwhile, and they run here, after the round's data is back, under the
 * mask Linux would have used, and are acknowledged when their handler is done.
 */
#define _GNU_SOURCE
#include "capstone/delegate.h"
#include <errno.h>
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include "ksigaction.h"

#pragma clang diagnostic ignored "-Wcapstone-pointer-roundtrip"
#pragma clang diagnostic ignored "-Wint-to-void-pointer-cast"
#pragma clang diagnostic ignored "-Wvoid-pointer-to-int-cast"
#pragma clang diagnostic ignored "-Wpointer-to-int-cast"

/* delegate.c: one round with up to three arguments, no marshalling */
long __capstone_delegate_ints(uint64_t nr, uint64_t a, uint64_t b, uint64_t c);
long __capstone_delegate_procmask(int how, const uint64_t *set, uint64_t *old);
void __capstone_call_on_stack(void (*fn)(void *), void *arg, void *top);

#define SIGNALS 64
#define EVENTS 256

struct action {
  void *handler;          /* SIG_DFL, SIG_IGN or the handler; a domain capability */
  unsigned long flags;
  uint64_t mask;          /* sa_mask, bit n-1 = signal n */
  uint64_t generation;    /* counts SIGACTION requests, like the launcher's */
  unsigned char cls;
};
static struct action table[SIGNALS + 1], previous[SIGNALS + 1];

enum { FREE, PENDING, RUNNING };
struct pending {
  struct capstone_signal_event ev;
  unsigned char state, on_alt;
  uintptr_t frame;        /* a local of the frame that runs it, or alt stack top */
};
static struct pending events[EVENTS];
static unsigned event_count;

static uint64_t logical;  /* the mask, as the kernel holds it for this task */
/* The calling context's handover block: the first context's is in transport
   0; a further context has none yet (docs/plans/delegation-threads.md). */
static __thread struct capstone_signal_block *block;
static __thread uint64_t seen_published;
static stack_t alt;
static int alt_enabled;
static int transition;    /* a management round of delivery is in progress */

static uint64_t bit(int sig) { return sig >= 1 && sig <= SIGNALS ? UINT64_C(1) << (sig - 1) : 0; }
static uint64_t of_set(const sigset_t *set) { uint64_t m; memcpy(&m, set, sizeof m); return m; }
static void to_set(uint64_t m, sigset_t *set) { memset(set, 0, sizeof *set); memcpy(set, &m, sizeof m); }

void __capstone_signals_regions(void *meta, unsigned long bytes) {
  block = bytes >= CAPSTONE_DELEGATE_META_BYTES
      ? (struct capstone_signal_block *)((char *)meta + CAPSTONE_SIGNAL_OFFSET) : 0;
  seen_published = 0;
  if (!block) return;
  logical = block->initial_mask;
  for (int sig = 1; sig <= SIGNALS; ++sig) {
    table[sig].handler = block->initial_ignored & bit(sig) ? SIG_IGN : SIG_DFL;
    table[sig].cls = block->initial_ignored & bit(sig)
        ? CAPSTONE_SIGNAL_IGNORE : CAPSTONE_SIGNAL_DEFAULT;
  }
}

int __capstone_signals_hint(void) {
  return block && block->recorded != seen_published;
}

/* Signals are the first context's for now (docs/plans/delegation-threads.md):
   only a context with a handover block takes events, runs handlers, or reads
   and changes the signal state. Another context's signal calls answer ENOSYS
   before they touch any of it; the launcher refuses them as well. */
static int signal_context(void) { return block != 0; }

/* Round end: the launcher published block->count events; keep them. */
void __capstone_signals_take(void) {
  if (!block) return;
  for (uint32_t i = 0; i < block->count && i < CAPSTONE_SIGNAL_EVENTS; ++i) {
    unsigned slot;
    for (slot = 0; slot < EVENTS && events[slot].state != FREE; ++slot) ;
    if (slot == EVENTS) break;   /* 256 unacknowledged events: the rest wait in the launcher */
    events[slot].ev = block->events[i];
    events[slot].state = PENDING;
    events[slot].on_alt = 0;
    if (slot >= event_count) event_count = slot + 1;
  }
  seen_published = block->published;
}

uint64_t __capstone_sigmask_current(void) { return logical; }

/* sigsetjmp with savemask: the buffer's __ss[0] holds the mask */
void __capstone_sigsetjmp_save(unsigned long *buf) { buf[29] = logical; }

/* musl's siglongjmp is a longjmp; here the saved mask comes back first. */
void siglongjmp(sigjmp_buf buf, int val) {
  unsigned long *raw = (unsigned long *)buf;
  if (raw[28]) {
    uint64_t saved = raw[29];
    transition++;
    __capstone_delegate_procmask(SIG_SETMASK, &saved, 0);
    transition--;
  }
  longjmp(buf, val);
}

static int on_altstack(void) {
  volatile char here;
  uintptr_t sp = (uintptr_t)&here, base = (uintptr_t)alt.ss_sp;
  return alt_enabled && sp >= base && sp < base + alt.ss_size;
}

long __capstone_sigaltstack(const stack_t *ss, stack_t *old) {
  if (!signal_context()) return -ENOSYS;
  if (old) {
    *old = alt;
    old->ss_flags = alt_enabled ? (on_altstack() ? SS_ONSTACK : 0) : SS_DISABLE;
  }
  if (ss) {
    if (on_altstack()) return -EPERM;
    if (ss->ss_flags & ~(SS_DISABLE | SS_AUTODISARM)) return -EINVAL;
    if (ss->ss_flags & SS_DISABLE) { alt_enabled = 0; return 0; }
    if (ss->ss_size < MINSIGSTKSZ) return -ENOMEM;
    alt = *ss;
    alt.ss_flags = 0;
    alt_enabled = 1;
  }
  return 0;
}

/* The kernel's RV64 siginfo, field by field into musl's, whose pointers are
 * capabilities. Only what Linux defines for the signal is copied. */
void __capstone_siginfo_translate(const unsigned char *raw, int sig, siginfo_t *si) {
  int32_t i32[8];
  int64_t i64[16];
  memcpy(i32, raw, sizeof i32);
  memcpy(i64, raw, sizeof i64);
  memset(si, 0, sizeof *si);
  si->si_signo = i32[0];
  si->si_errno = i32[1];
  si->si_code = i32[2];
  if (sig == SIGCHLD) {
    si->si_pid = i32[4]; si->si_uid = (uid_t)i32[5]; si->si_status = i32[6];
    si->si_utime = (clock_t)i64[4]; si->si_stime = (clock_t)i64[5];
  } else if (si->si_code == SI_TIMER) {
    si->si_timerid = i32[4]; si->si_overrun = i32[5]; si->si_value.sival_int = i32[6];
  } else if (si->si_code <= 0) {   /* SI_USER, SI_QUEUE, SI_TKILL, SI_MESGQ, ... */
    si->si_pid = i32[4]; si->si_uid = (uid_t)i32[5]; si->si_value.sival_int = i32[6];
  } else if (sig == SIGSEGV || sig == SIGBUS || sig == SIGILL || sig == SIGFPE || sig == SIGTRAP) {
    si->si_addr = (void *)(uintptr_t)i64[2];
  } else if (sig == SIGPOLL) {
    si->si_band = (long)i64[2]; si->si_fd = i32[6];
  }
}

struct call { struct action *a; int sig; siginfo_t *si; ucontext_t *uc; };
static void invoke(void *arg) {
  struct call *c = arg;
  if (c->a->flags & SA_SIGINFO)
    ((void (*)(int, siginfo_t *, void *))c->a->handler)(c->sig, c->si, c->uc);
  else
    ((void (*)(int))c->a->handler)(c->sig);
}

static struct action *action_for(const struct capstone_signal_event *ev) {
  int sig = (int)ev->signo;
  if (sig < 1 || sig > SIGNALS) return 0;
  if (table[sig].generation == ev->generation) return &table[sig];
  if (previous[sig].generation == ev->generation) return &previous[sig];
  return 0;
}

static int caught(const struct action *a) {
  return a && a->cls == CAPSTONE_SIGNAL_CAUGHT && a->handler != SIG_DFL && a->handler != SIG_IGN;
}

static void acknowledge(struct pending *p) {
  p->state = FREE;
  transition++;
  __capstone_delegate_ints(CAPSTONE_NR_SIGDONE, p->ev.seq, 0, 0);
  transition--;
}

/* longjmp restores the saved sp. A running handler below that sp has been
 * abandoned, even if a later call grows the stack below its old frame. The
 * alternate stack is a separate stack: jumping from it to the normal stack
 * abandons every handler currently on it. */
void __capstone_longjmp_reap(void *buf) {
  void *target = *(void **)((char *)buf + 208); /* sp: setjmp.S slot 13 */
  uintptr_t sp = (uintptr_t)target, base = (uintptr_t)alt.ss_sp;
  int target_alt = alt_enabled && sp >= base && sp < base + alt.ss_size;
  for (unsigned i = 0; i < event_count; ++i) {
    struct pending *p = &events[i];
    if (p->state != RUNNING) continue;
    int gone = p->on_alt == target_alt ? p->frame < sp : p->on_alt;
    if (gone) acknowledge(p);
  }
}

static void run_event(struct pending *p) {
  volatile char marker;
  int sig = (int)p->ev.signo;
  struct action *a = action_for(&p->ev);
  uint64_t saved = logical, base, handler_mask;
  siginfo_t si;
  ucontext_t uc;
  if (!caught(a)) {           /* the installation is gone: Linux would have run it then */
    acknowledge(p);
    return;
  }
  p->state = RUNNING;
  p->frame = (uintptr_t)&marker;
  base = (p->ev.flags & CAPSTONE_SIGNAL_WAIT) ? p->ev.mask : logical;
  handler_mask = base | a->mask | ((a->flags & SA_NODEFER) ? 0 : bit(sig));
  handler_mask &= ~(bit(SIGKILL) | bit(SIGSTOP));
  transition++;
  __capstone_delegate_procmask(SIG_SETMASK, &handler_mask, 0);
  transition--;
  if ((a->flags & SA_RESETHAND) && a == &table[sig]) {
    /* the kernel reset its side at acceptance; the generation stays, no request was sent */
    previous[sig] = table[sig];
    table[sig].handler = SIG_DFL;
    table[sig].cls = CAPSTONE_SIGNAL_DEFAULT;
    a = &previous[sig];
  }
  __capstone_siginfo_translate(p->ev.info, sig, &si);
  memset(&uc, 0, sizeof uc);
  to_set(base, &uc.uc_sigmask);
  struct call c = {a, sig, &si, &uc};
  /* frame and on_alt identify which running handlers a longjmp leaves. */
  if ((a->flags & SA_ONSTACK) && alt_enabled && !on_altstack()) {
    char *top = (char *)alt.ss_sp + (alt.ss_size & ~(size_t)15);
    p->on_alt = 1;
    p->frame = (uintptr_t)top;
    __capstone_call_on_stack(invoke, &c, top);
  } else {
    p->on_alt = on_altstack();
    invoke(&c);
  }
  transition++;
  __capstone_delegate_procmask(SIG_SETMASK, &saved, 0);
  transition--;
  acknowledge(p);
}

static struct pending *next_runnable(int only_sig) {
  struct pending *best = 0;
  for (unsigned i = 0; i < event_count; ++i) {
    struct pending *p = &events[i];
    if (p->state != PENDING) continue;
    if (only_sig && (int)p->ev.signo != only_sig) continue;
    if (!(p->ev.flags & CAPSTONE_SIGNAL_WAIT) && (logical & bit((int)p->ev.signo))) continue;
    if (!best || p->ev.seq < best->ev.seq) best = p;
  }
  return best;
}

/* After every round: run what is runnable, in acceptance order. */
void __capstone_signals_deliver(void) {
  struct pending *p;
  if (transition || !signal_context()) return;
  while ((p = next_runnable(0)))
    run_event(p);
}

long __capstone_sigaction(int sig, const struct k_sigaction *new, struct k_sigaction *old) {
  struct pending *p;
  if (sig < 1 || sig > SIGNALS || sig == SIGKILL || sig == SIGSTOP) return -EINVAL;
  if (!signal_context()) return -ENOSYS;
  if (new && !transition)
    while ((p = next_runnable(sig)))   /* accepted for the old installation: run it first */
      run_event(p);
  if (old) {
    memset(old, 0, sizeof *old);
    old->handler = table[sig].handler;
    old->flags = table[sig].flags;
    memcpy(old->mask, &table[sig].mask, sizeof table[sig].mask);
  }
  if (new) {
    unsigned cls = new->handler == SIG_DFL ? CAPSTONE_SIGNAL_DEFAULT
                 : new->handler == SIG_IGN ? CAPSTONE_SIGNAL_IGNORE : CAPSTONE_SIGNAL_CAUGHT;
    /* A management round: an event this round publishes may already belong
       to the new installation, so the table moves before anything runs. */
    transition++;
    long r = __capstone_delegate_ints(CAPSTONE_NR_SIGACTION, (uint64_t)sig, cls, (uint32_t)new->flags);
    transition--;
    if (r) {
      __capstone_signals_deliver();
      return r;
    }
    previous[sig] = table[sig];
    table[sig].handler = new->handler;
    table[sig].flags = new->flags;
    memcpy(&table[sig].mask, new->mask, sizeof table[sig].mask);
    table[sig].generation++;
    table[sig].cls = (unsigned char)cls;
  }
  __capstone_signals_deliver();
  return 0;
}

/* rt_sigprocmask, mirrored: the launcher answers with the old logical mask,
 * which is what this held; the mirror moves the way the kernel did. */
long __capstone_sigprocmask(int how, const sigset_t *set, sigset_t *old, unsigned long size) {
  uint64_t m = set ? of_set(set) : 0, before = 0;
  long r;
  if (size != 8) return -EINVAL;
  if (!signal_context()) return -ENOSYS;
  /* the round that unblocks a signal publishes it: the mirror must have
     moved before delivery decides what is runnable */
  transition++;
  r = __capstone_delegate_procmask(how, set ? &m : 0, &before);
  transition--;
  __capstone_signals_deliver();
  if (r) return r;
  /* the kernel writes sigsetsize bytes of the old mask and nothing more; musl's
     sigaction hands over an unsigned long[1] for it, not a sigset_t */
  if (old) memcpy(old, &before, 8);
  return 0;
}

/* delegate.c reports the new mask after a successful delegated procmask */
void __capstone_sigmask_note(int how, uint64_t m) {
  m &= ~(bit(SIGKILL) | bit(SIGSTOP));
  if (how == SIG_BLOCK) logical |= m;
  else if (how == SIG_UNBLOCK) logical &= ~m;
  else logical = m;
}
