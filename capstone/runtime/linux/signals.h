/* The launcher's half of delegated signals: Linux delivers to this task, the
 * runtime carries the event into the domain. See docs/plans/delegation-signals.md.
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

struct capstone_signal_state {
  struct capstone_signal_event ring[CAPSTONE_SIGNAL_RING];
  struct capstone_signal_event overflow; /* one emergency record beyond the ring */
  unsigned overflow_valid;
  _Atomic uint64_t head;        /* events recorded; the trampoline advances it */
  uint64_t tail;                /* events published; the round end advances it */
  _Atomic uint64_t inflight;    /* accepted, handler not yet acknowledged, no SA_NODEFER */
  uint64_t logical;             /* the domain's mask, as delegated */
  uint64_t backpressure;        /* caught signals blocked while the ring is nearly full */
  uint8_t class[CAPSTONE_SIGNAL_MAX + 1];
  uint32_t flags[CAPSTONE_SIGNAL_MAX + 1];
  uint64_t generation[CAPSTONE_SIGNAL_MAX + 1];
  /* Every published event occupies one domain slot until SIGDONE. */
  struct { uint64_t seq; int signo; unsigned char deferred; } open[CAPSTONE_SIGNAL_RING];
  unsigned open_count;
  /* the wait the dispatcher is executing, for the event class */
  int wait_active;
  uint64_t wait_mask;
  uint64_t spilled;             /* emergency slot use, never expected in normal flow */
  struct capstone_signal_block *block;  /* the META region's handover block, or NULL */
};

/* The trampoline serves one launcher; this names its state. */
void capstone_signals_init(struct capstone_signal_state *s, struct capstone_signal_block *block);

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
