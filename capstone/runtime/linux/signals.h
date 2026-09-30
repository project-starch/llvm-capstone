/* The launcher's half of delegated signals: Linux delivers to this task, the
 * runtime carries the event into the domain. See docs/plans/delegation-signals.md,
 * and docs/plans/delegation-threads.md (B8) for contexts.
 *
 * One state per context, served on that context's launcher thread: its ring,
 * its mask (the thread's kernel mask is the context's logical mask with the
 * in-flight and backpressure signals added), its handover block. The
 * dispositions are the process's, one table shared by every context's state,
 * as Linux keeps them per process. So Linux itself picks the thread for a
 * process-directed signal, and a thread-directed one reaches its context.
 *
 * Linux keeps dispositions (as classes: default, ignore, caught), the blocked
 * mask, the pending set and every default action. What lives here is the one
 * state Linux does not have: an accepted event on its way to a domain handler.
 * The trampoline records it into a private ring; the round end publishes it
 * into the META region's handover block; the domain acknowledges it when its
 * handler is done. Until then the signal stays blocked in the kernel mask
 * unless the domain asked for SA_NODEFER, exactly as Linux blocks a signal
 * for the duration of its handler.
 */
#ifndef CAPSTONE_LINUX_SIGNALS_H
#define CAPSTONE_LINUX_SIGNALS_H

#include "capstone/delegate.h"
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>

#define CAPSTONE_SIGNAL_RING 256u
#define CAPSTONE_SIGNAL_BACKPRESSURE 192u /* leave room before blocking caught signals */
#define CAPSTONE_SIGNAL_MAX 64

/* The stub's answer when the call did not run because events wait, or because
 * Linux prepared a restart that must not happen before the handlers ran. Not
 * an errno: those are -1..-4095. */
#define CAPSTONE_STUB_RETRY (-0x10000L)

struct capstone_signal_table {
  uint8_t class[CAPSTONE_SIGNAL_MAX + 1];
  uint32_t flags[CAPSTONE_SIGNAL_MAX + 1];
  uint64_t generation[CAPSTONE_SIGNAL_MAX + 1];
};

struct capstone_signal_state {
  struct capstone_signal_event ring[CAPSTONE_SIGNAL_RING];
  struct capstone_signal_event overflow; /* one emergency record beyond the ring */
  unsigned overflow_valid;
  _Atomic uint64_t head;        /* events recorded; the trampoline advances it */
  uint64_t tail;                /* events published; the round end advances it */
  _Atomic uint64_t inflight;    /* accepted, handler not yet acknowledged, no SA_NODEFER */
  uint64_t logical;             /* the context's mask, as delegated */
  uint64_t backpressure;        /* caught signals blocked while the ring is nearly full */
  struct capstone_signal_table *table;   /* the process's dispositions */
  struct capstone_signal_table own;      /* the table, in the first context's state */
  /* Every published event occupies one domain slot until SIGDONE. */
  struct { uint64_t seq; int signo; unsigned char deferred; } open[CAPSTONE_SIGNAL_RING];
  unsigned open_count;
  /* the wait the dispatcher is executing, for the event class */
  int wait_active;
  uint64_t wait_mask;
  uint64_t spilled;             /* emergency slot use, never expected in normal flow */
  struct capstone_signal_block *block;  /* the META region's handover block, or NULL */
};

/* The first context's state: the process's table, read from the kernel with
 * the mask; the calling thread serves it (capstone_signals_attach). */
void capstone_signals_init(struct capstone_signal_state *s, struct capstone_signal_block *block);

/* A further context's state, made by its creator: the creator's table, and
 * the creator's mask, which a new thread inherits on Linux. The block tells
 * the domain both. The context's own thread attaches it. */
void capstone_signals_init_context(struct capstone_signal_state *s, struct capstone_signal_block *block,
                                   const struct capstone_signal_state *creator);

/* The calling thread serves s from now on: the trampoline records there, and
 * the thread's kernel mask becomes the context's. A signal that reaches a
 * launcher thread before it attached stays pending, blocked there if Linux
 * sent it to that thread, sent back to the process otherwise. */
void capstone_signals_attach(struct capstone_signal_state *s);

/* The calling thread serves no context from now on: every signal blocked. */
void capstone_signals_detach(void);

/* Once, before any domain disposition: glibc's one-time setup of its internal
   signals (its first pthread_create installs a handler on 33), with the
   dispositions the process started with put back. */
void capstone_signals_settle_libc(void);

/* The calling thread's kernel mask, bypassing the C library, which keeps two
 * of the signals a domain's libc uses (32 and 33, musl's timer and cancel
 * signals) to itself. Every launcher mask change goes through these. */
uint64_t capstone_signals_set_kernel_mask(uint64_t mask);   /* returns the previous mask */

/* SIGACTION: install the class in the kernel. Returns 0 or -errno. */
long capstone_signals_action(struct capstone_signal_state *s, int signo, unsigned cls,
                             unsigned flags);

/* rt_sigprocmask on the logical mask; the kernel gets logical | inflight |
 * backpressure. `set` may be NULL; `old` receives the previous logical mask. */
long capstone_signals_procmask(struct capstone_signal_state *s, int how, const uint64_t *set,
                               uint64_t *old);

/* SIGDONE: the handler for event `seq` ran; unblock its signal. */
long capstone_signals_done(struct capstone_signal_state *s, uint64_t seq);

/* True when the ring holds events the domain has not been given. */
int capstone_signals_waiting(const struct capstone_signal_state *s);

/* Every waiting event's action has SA_RESTART: a compound operation may
 * answer RETRY instead of -EINTR. */
int capstone_signals_restartable(const struct capstone_signal_state *s);

/* Round end: move waiting events into the handover block, set entry->pending
 * and entry->status (RETRY when `retry`). */
void capstone_signals_publish(struct capstone_signal_state *s, struct capstone_delegate_entry *entry,
                              int retry);

/* The ignored signals, for a spawned child to inherit. */
uint64_t capstone_signals_ignored(const struct capstone_signal_state *s);


/* Run one Linux system call for the domain through the stub: skipped when
 * events wait, turned into RETRY when the trampoline redirected it. `nr` is
 * the host's number. For rt_sigsuspend and ppoll with a mask, `wait_mask`
 * classifies the events accepted meanwhile. Returns the kernel's value
 * (negative errno on failure) or CAPSTONE_STUB_RETRY. */
long capstone_signals_call(struct capstone_signal_state *s, long nr, const long args[6],
                           int has_wait_mask, uint64_t wait_mask);

#endif
