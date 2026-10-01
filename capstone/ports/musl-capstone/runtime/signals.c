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
#include <stdlib.h>
#include <string.h>
#include <ucontext.h>
#include <capstone/lock.h>
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
/* The dispositions are the process's, like Linux's: one table for every
   context, changed under table_lock (the request and the table move together). */
static struct action table[SIGNALS + 1], previous[SIGNALS + 1];
static volatile int table_lock;
/* The entries themselves, read and written only under this leaf lock: a
   delivery copies its action here while another context may be changing it. */
static volatile int table_spin;

enum { FREE, PENDING, RUNNING };
struct pending {
  struct capstone_signal_event ev;
  unsigned char state, on_alt;
  uintptr_t frame;        /* a local of the frame that runs it, or alt stack top */
};
/* Everything below is each context's own (docs/plans/delegation-threads.md,
   B8), as Linux keeps a mask, pending signals and an alternate stack per
   thread. The events live on the heap, taken when the context attaches:
   256 of them do not belong in every TLS block. */
static __thread struct pending *events;
static __thread unsigned event_count;
static __thread uint64_t logical;  /* the mask, as the kernel holds it for this context's thread */
/* The context's handover block, in its transport's META block. */
static __thread struct capstone_signal_block *block;
static __thread uint64_t seen_published;
static __thread stack_t alt;
static __thread int alt_enabled;
static __thread int transition;    /* a management round of delivery is in progress */

static uint64_t bit(int sig) { return sig >= 1 && sig <= SIGNALS ? UINT64_C(1) << (sig - 1) : 0; }
static uint64_t of_set(const sigset_t *set) { uint64_t m; memcpy(&m, set, sizeof m); return m; }
static void to_set(uint64_t m, sigset_t *set) { memset(set, 0, sizeof *set); memcpy(set, &m, sizeof m); }

/* A context that ends takes no more events: its list is handed back (the
   caller frees it where it may take the heap lock). */
void *__capstone_signals_detach(void) {
  void *list = events;
  block = 0;
  events = 0;
  event_count = 0;
  return list;
}

/* A context takes its transport's handover block: its mask as the launcher
   set it (the first context's from the launch, a further context's its
   creator's), no events. Without room for its events it takes none, and its
   signal calls answer ENOSYS. */
void __capstone_signals_attach(void *meta) {
  seen_published = 0;
  event_count = 0;
  block = 0;
  if (!meta || !(events = calloc(EVENTS, sizeof *events)))
    return;
  block = (struct capstone_signal_block *)((char *)meta + CAPSTONE_SIGNAL_OFFSET);
  logical = block->initial_mask;
}

/* The first context: its block, and the process's table from the launch. */
void __capstone_signals_regions(void *meta, unsigned long bytes) {
  __capstone_signals_attach(bytes >= CAPSTONE_DELEGATE_META_BYTES ? meta : 0);
  if (!block) return;
  for (int sig = 1; sig <= SIGNALS; ++sig) {
    table[sig].handler = block->initial_ignored & bit(sig) ? SIG_IGN : SIG_DFL;
    table[sig].cls = block->initial_ignored & bit(sig)
        ? CAPSTONE_SIGNAL_IGNORE : CAPSTONE_SIGNAL_DEFAULT;
  }
}

int __capstone_signals_hint(void) {
  return block && block->recorded != seen_published;
}

/* Only a context with a handover block takes events, runs handlers, or reads
   and changes signal state; one without (no transport, or no room for its
   events) answers ENOSYS. */
static int signal_context(void) { return block != 0; }

/* Cancellation points (musl's pthread_cancel.c). musl's handler for SIGCANCEL
   decides by the interrupted pc: inside [__cp_begin, __cp_end) it moves the
   pc to __cp_cancel, and the call is abandoned for a cancellation. Here the
   call is delegated rounds, not an instruction: __syscall_cp_asm marks the
   context as inside one, and until the call's own round has its result (a
   delivery at the SIGPOLL round before it, or at a RETRY round) a handler is
   shown __cp_begin; one that moved it to __cp_cancel ends the call with
   EINTR, without making it if it was not made yet, and musl's __syscall_cp_c
   then cancels. After the result the handler is shown __cp_end, and musl
   sends the signal again for the next cancellation point. A handler's own
   cancellation point nests: the state is saved around it. The three symbols
   are addresses to compare, never code that runs (start-musl.S). */
extern const char __cp_begin[1], __cp_end[1], __cp_cancel[1];
static __thread int in_cp, cp_unfinished, cp_cancelled;

int __capstone_cp_enter(void) {
  int saved = in_cp | cp_unfinished << 1 | cp_cancelled << 2;
  in_cp = 1;
  cp_unfinished = 1;
  cp_cancelled = 0;
  return saved;
}
/* Restores the enclosing state; returns whether this cancellation point was
   cancelled. */
int __capstone_cp_leave(int saved) {
  int cancelled = in_cp && cp_cancelled;
  in_cp = saved & 1;
  cp_unfinished = saved >> 1 & 1;
  cp_cancelled = saved >> 2 & 1;
  return cancelled;
}
/* Out of a cancellation point for a call the runtime makes on its own inside
   one (a clock read): __capstone_cp_leave restores. */
int __capstone_cp_pause(void) {
  int saved = in_cp | cp_unfinished << 1 | cp_cancelled << 2;
  in_cp = 0;
  return saved;
}
/* delegate.c: the cancellation point's own round has its result */
void __capstone_cp_result(void) { cp_unfinished = 0; }
int __capstone_cp_cancelled(void) { return in_cp && cp_cancelled; }

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
  if (raw[28] && signal_context()) {
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
  if (!signal_context()) return;   /* a context without a handover block runs none */
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
  struct action copy, *a = &copy;   /* this delivery's: another context may change the table */
  uint64_t saved = logical, base, handler_mask;
  siginfo_t si;
  ucontext_t uc;
  capstone_spin_lock(&table_spin);
  struct action *found = action_for(&p->ev);
  int runs = caught(found);
  if (runs) {
    copy = *found;
    if ((copy.flags & SA_RESETHAND) && found == &table[sig]) {
      /* the kernel reset its side at acceptance; the generation stays, no request was sent */
      previous[sig] = table[sig];
      table[sig].handler = SIG_DFL;
      table[sig].cls = CAPSTONE_SIGNAL_DEFAULT;
    }
  }
  capstone_spin_unlock(&table_spin);
  if (!runs) {                /* the installation is gone: Linux would have run it then */
    acknowledge(p);
    return;
  }
  p->state = RUNNING;
  p->frame = (uintptr_t)&marker;
  /* where a cancellation point stands, then out of it for the delivery: the
     handler's calls and the mask changes around it are its own */
  uintptr_t shown = in_cp ? (uintptr_t)(cp_unfinished ? __cp_begin : __cp_end) : 0;
  int cp_saved = in_cp | cp_unfinished << 1 | cp_cancelled << 2;
  in_cp = 0;
  base = (p->ev.flags & CAPSTONE_SIGNAL_WAIT) ? p->ev.mask : logical;
  handler_mask = base | a->mask | ((a->flags & SA_NODEFER) ? 0 : bit(sig));
  handler_mask &= ~(bit(SIGKILL) | bit(SIGSTOP));
  transition++;
  __capstone_delegate_procmask(SIG_SETMASK, &handler_mask, 0);
  transition--;
  __capstone_siginfo_translate(p->ev.info, sig, &si);
  /* the mask the handler returns to, as the kernel shows it: the one from
     before the delivery (for a sigsuspend, from before the wait) */
  memset(&uc, 0, sizeof uc);
  to_set(saved, &uc.uc_sigmask);
  uc.uc_mcontext.__gregs[0] = shown;
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
  /* sigreturn's mask: what the handler left in uc_sigmask */
  uint64_t restore = of_set(&uc.uc_sigmask) & ~(bit(SIGKILL) | bit(SIGSTOP));
  transition++;
  __capstone_delegate_procmask(SIG_SETMASK, &restore, 0);
  transition--;
  acknowledge(p);
  (void)__capstone_cp_leave(cp_saved);
  if (in_cp && cp_unfinished && uc.uc_mcontext.__gregs[0] == (uintptr_t)__cp_cancel)
    cp_cancelled = 1;
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
int __capstone_lock_depth(void);   /* lock.c */

void __capstone_signals_deliver(void) {
  struct pending *p;
  /* never while this context holds a runtime-internal lock (Q6): the handler
     could want it; the events run when the last one is released */
  if (transition || !signal_context() || __capstone_lock_depth()) return;
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
  long r = 0;
  capstone_lock(&table_lock);   /* one change at a time, its request included */
  if (old) {
    memset(old, 0, sizeof *old);
    capstone_spin_lock(&table_spin);
    old->handler = table[sig].handler;
    old->flags = table[sig].flags;
    memcpy(old->mask, &table[sig].mask, sizeof table[sig].mask);
    capstone_spin_unlock(&table_spin);
  }
  if (new) {
    unsigned cls = new->handler == SIG_DFL ? CAPSTONE_SIGNAL_DEFAULT
                 : new->handler == SIG_IGN ? CAPSTONE_SIGNAL_IGNORE : CAPSTONE_SIGNAL_CAUGHT;
    /* A management round: an event this round publishes may already belong
       to the new installation, so the table moves before anything runs. */
    transition++;
    r = __capstone_delegate_ints(CAPSTONE_NR_SIGACTION, (uint64_t)sig, cls, (uint32_t)new->flags);
    transition--;
    if (!r) {
      capstone_spin_lock(&table_spin);
      previous[sig] = table[sig];
      table[sig].handler = new->handler;
      table[sig].flags = new->flags;
      memcpy(&table[sig].mask, new->mask, sizeof table[sig].mask);
      table[sig].generation++;
      table[sig].cls = (unsigned char)cls;
      capstone_spin_unlock(&table_spin);
    }
  }
  capstone_unlock(&table_lock);   /* its release delivers what waited (lock.c) */
  __capstone_signals_deliver();
  return r;
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
