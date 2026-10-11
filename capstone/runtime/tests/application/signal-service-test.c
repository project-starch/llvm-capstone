/* Native test of the launcher's signal adapter: the trampoline records, the
 * round end publishes, the domain acknowledges, the physical mask follows.
 * The stub's program-counter redirect is RV64-only and is covered by the
 * signal contract in the guest; here the C fallback checks the ring. */
#include "../../linux/delegate-service.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define EXCHANGE 4096
static char exchange[EXCHANGE];
static struct capstone_delegate_host host = {.exchange = exchange, .exchange_bytes = EXCHANGE};
static struct capstone_signal_block *block;

static struct capstone_delegate_entry entry(uint64_t nr, uint64_t a, uint64_t b, uint64_t c,
                                            uint64_t d, uint64_t e, uint64_t f) {
  struct capstone_delegate_entry x;
  uint64_t args[CAPSTONE_DELEGATE_ARGS] = {a, b, c, d, e, f};
  assert(!capstone_delegate_pack(&x, nr, args));
  return x;
}
static long serve(struct capstone_delegate_entry *x) {
  capstone_delegate_serve(&host, x);
  return (long)x->result;
}
static int blocked(int sig) {
  sigset_t now;
  sigprocmask(SIG_BLOCK, NULL, &now);
  return sigismember(&now, sig);
}
static uint64_t bit(int sig) { return UINT64_C(1) << (sig - 1); }

int main(void) {
  struct capstone_delegate_entry x;
  block = calloc(1, sizeof *block);
  capstone_signals_init(&host.signals, block);

  /* a caught class installs the trampoline; SIGUSR1 delivered to us lands in the ring */
  x = entry(CAPSTONE_NR_SIGACTION, SIGUSR1, CAPSTONE_SIGNAL_CAUGHT, 0, 0, 0, 0);
  assert(serve(&x) == 0 && x.status == CAPSTONE_ROUND_DONE && !x.pending);
  assert(!blocked(SIGUSR1));
  raise(SIGUSR1);
  assert(blocked(SIGUSR1));                     /* in flight: blocked until acknowledged */
  assert(capstone_signals_waiting(&host.signals));
  /* the next call is not made: RETRY, and the event is published */
  int fds[2];
  assert(!pipe(fds));
  x = entry(CAPSTONE_SYS_read, (uint64_t)fds[0], 64, 8, 0, 0, 0);
  serve(&x);
  assert(x.status == CAPSTONE_ROUND_RETRY && x.result == 0);
  assert(x.pending == bit(SIGUSR1) && block->count == 1 && block->published == 1);
  assert(block->events[0].seq == 1 && block->events[0].signo == SIGUSR1 &&
         block->events[0].flags == CAPSTONE_SIGNAL_DEFER);
  siginfo_t info;
  memcpy(&info, block->events[0].info, sizeof info);
  assert(info.si_signo == SIGUSR1 && info.si_pid == getpid());
  assert(!capstone_signals_waiting(&host.signals));
  /* the retried call runs and completes; nothing new is pending */
  assert(write(fds[1], "abc", 3) == 3);
  x = entry(CAPSTONE_SYS_read, (uint64_t)fds[0], 64, 8, 0, 0, 0);
  assert(serve(&x) == 3 && x.status == CAPSTONE_ROUND_DONE && !x.pending && block->count == 0);
  /* acknowledgement unblocks */
  x = entry(CAPSTONE_NR_SIGDONE, 1, 0, 0, 0, 0, 0);
  assert(serve(&x) == 0 && !blocked(SIGUSR1));
  x = entry(CAPSTONE_NR_SIGDONE, 1, 0, 0, 0, 0, 0);
  assert(serve(&x) == -EINVAL);                 /* twice is an error */

  /* the logical mask: rt_sigprocmask returns the old one, the kernel follows */
  uint64_t set = bit(SIGUSR2), old = ~UINT64_C(0);
  memcpy(exchange + 128, &set, sizeof set);
  x = entry(CAPSTONE_SYS_rt_sigprocmask, SIG_BLOCK, 128, 256, 8, 0, 0);
  assert(serve(&x) == 0);
  memcpy(&old, exchange + 256, sizeof old);
  assert(old == 0 && blocked(SIGUSR2) && host.signals.logical == bit(SIGUSR2));
  /* a blocked caught signal stays in the kernel, not in the ring */
  x = entry(CAPSTONE_NR_SIGACTION, SIGUSR2, CAPSTONE_SIGNAL_CAUGHT, SA_RESTART, 0, 0, 0);
  assert(serve(&x) == 0);
  raise(SIGUSR2);
  assert(!capstone_signals_waiting(&host.signals));
  /* unblocking delivers it on the way out of that very call: published in the same round */
  x = entry(CAPSTONE_SYS_rt_sigprocmask, SIG_UNBLOCK, 128, 0, 8, 0, 0);
  assert(serve(&x) == 0 && x.pending == bit(SIGUSR2) && block->count == 1);
  assert(blocked(SIGUSR2));                     /* in flight again */
  assert(capstone_signals_ignored(&host.signals) == 0);
  x = entry(CAPSTONE_NR_SIGDONE, block->events[0].seq, 0, 0, 0, 0, 0);
  assert(serve(&x) == 0 && !blocked(SIGUSR2));

  /* ignore and default are the kernel's own */
  x = entry(CAPSTONE_NR_SIGACTION, SIGPIPE, CAPSTONE_SIGNAL_IGNORE, 0, 0, 0, 0);
  assert(serve(&x) == 0 && capstone_signals_ignored(&host.signals) == bit(SIGPIPE));
  int closed[2];
  assert(!pipe(closed));
  close(closed[0]);
  assert(write(closed[1], "x", 1) == -1 && errno == EPIPE);   /* ignored: EPIPE, not death */
  x = entry(CAPSTONE_NR_SIGACTION, SIGKILL, CAPSTONE_SIGNAL_CAUGHT, 0, 0, 0, 0);
  assert(serve(&x) == -EINVAL);

  /* a wait with a temporary mask classifies the event it accepts */
  {
    uint64_t block_all = bit(SIGUSR1) | bit(SIGUSR2);
    memcpy(exchange + 128, &block_all, sizeof block_all);
    x = entry(CAPSTONE_SYS_rt_sigprocmask, SIG_SETMASK, 128, 0, 8, 0, 0);
    assert(serve(&x) == 0 && blocked(SIGUSR1));
    pid_t child = fork();
    if (!child) { usleep(100000); kill(getppid(), SIGUSR1); _exit(0); }
    uint64_t empty = 0;
    memcpy(exchange + 512, &empty, sizeof empty);
    x = entry(CAPSTONE_SYS_rt_sigsuspend, 512, 8, 0, 0, 0, 0);
    long r = serve(&x);
    assert(r == -EINTR && x.status == CAPSTONE_ROUND_DONE);
    assert(x.pending == bit(SIGUSR1) && block->count == 1);
    assert(block->events[0].flags == (CAPSTONE_SIGNAL_WAIT | CAPSTONE_SIGNAL_DEFER) &&
           block->events[0].mask == 0);
    assert(host.signals.logical == block_all && blocked(SIGUSR2));
    waitpid(child, NULL, 0);
    x = entry(CAPSTONE_NR_SIGDONE, block->events[0].seq, 0, 0, 0, 0, 0);
    assert(serve(&x) == 0);
  }

  /* restartable: every waiting event's action has SA_RESTART */
  {
    uint64_t none = 0;
    memcpy(exchange + 128, &none, sizeof none);
    x = entry(CAPSTONE_SYS_rt_sigprocmask, SIG_SETMASK, 128, 0, 8, 0, 0);
    assert(serve(&x) == 0);
    raise(SIGUSR2);                             /* SA_RESTART */
    assert(capstone_signals_restartable(&host.signals));
    raise(SIGUSR1);                             /* no SA_RESTART */
    assert(!capstone_signals_restartable(&host.signals));
    raise(SIGUSR1);                             /* coalesces: one event */
    x = entry(CAPSTONE_NR_SIGPOLL, 0, 0, 0, 0, 0, 0);
    assert(serve(&x) == 0 && block->count == 2);
    assert(block->events[0].signo == SIGUSR2 && block->events[1].signo == SIGUSR1);
  }
  puts("signal service passed");
  return 0;
}
