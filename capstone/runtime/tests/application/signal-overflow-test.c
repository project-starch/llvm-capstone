/* A realtime burst must keep every payload while the launcher's ring applies
 * backpressure. SA_NODEFER makes the kernel willing to deliver the burst. */
#include "../../linux/signals.h"
#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static struct capstone_signal_state state;
static struct capstone_signal_block block;

int main(void) {
  int rt = SIGRTMIN + 3;
  sigset_t original, held;
  struct sigaction before, pipe_before;
  assert(!sigprocmask(SIG_BLOCK, NULL, &original));
  assert(!sigaction(rt, NULL, &before));
  assert(!sigaction(SIGPIPE, NULL, &pipe_before));
  assert(signal(SIGPIPE, SIG_IGN) != SIG_ERR);
  sigemptyset(&held);
  sigaddset(&held, SIGUSR2);
  assert(!sigprocmask(SIG_BLOCK, &held, NULL));
  capstone_signals_init(&state, &block);
  assert(block.initial_mask & (UINT64_C(1) << (SIGUSR2 - 1)));
  assert(block.initial_ignored & (UINT64_C(1) << (SIGPIPE - 1)));
  assert(capstone_signals_ignored(&state) & (UINT64_C(1) << (SIGPIPE - 1)));
  assert(!capstone_signals_action(&state, rt, CAPSTONE_SIGNAL_CAUGHT,
                                  SA_SIGINFO | SA_NODEFER));
  sigemptyset(&held);
  sigaddset(&held, rt);
  assert(!sigprocmask(SIG_BLOCK, &held, NULL));
  for (int i = 1; i <= 400; ++i)
    assert(!sigqueue(getpid(), rt, (union sigval){.sival_int = i}));
  assert(!sigprocmask(SIG_UNBLOCK, &held, NULL));

  unsigned received = 0;
  for (unsigned round = 0; round < 100 && received < 400; ++round) {
    struct capstone_delegate_entry entry = {0};
    capstone_signals_publish(&state, &entry, 0);
    for (unsigned i = 0; i < block.count; ++i) {
      siginfo_t info;
      memcpy(&info, block.events[i].info, sizeof info);
      assert(block.events[i].signo == (unsigned)rt);
      assert(info.si_code == SI_QUEUE && info.si_value.sival_int == (int)++received);
      assert(!capstone_signals_done(&state, block.events[i].seq));
    }
  }
  assert(received == 400 && state.spilled == 0 && state.open_count == 0);

  /* Exercise the spare record explicitly: the ordinary threshold prevents a
     full ring, so seed one here and require the extra signal's original value. */
  for (unsigned i = 0; i < CAPSTONE_SIGNAL_RING; ++i) {
    struct capstone_signal_event *ev = &state.ring[(400 + i) % CAPSTONE_SIGNAL_RING];
    siginfo_t info = {.si_signo = rt, .si_code = SI_QUEUE};
    info.si_value.sival_int = 401 + (int)i;
    ev->seq = 401 + i;
    ev->signo = (uint32_t)rt;
    ev->flags = 0;
    memcpy(ev->info, &info, sizeof info);
  }
  atomic_store(&state.head, 400 + CAPSTONE_SIGNAL_RING);
  assert(!sigqueue(getpid(), rt, (union sigval){.sival_int = 999}));
  assert(state.spilled == 1 && state.overflow_valid);
  for (unsigned round = 0; round < 10 && received < 657; ++round) {
    struct capstone_delegate_entry entry = {0};
    capstone_signals_publish(&state, &entry, 0);
    for (unsigned i = 0; i < block.count; ++i) {
      siginfo_t info;
      memcpy(&info, block.events[i].info, sizeof info);
      ++received;
      assert(info.si_value.sival_int == (received == 657 ? 999 : (int)received));
      assert(!capstone_signals_done(&state, block.events[i].seq));
    }
  }
  assert(received == 657 && !state.overflow_valid && state.open_count == 0);
  assert(!sigaction(rt, &before, NULL));
  assert(!sigaction(SIGPIPE, &pipe_before, NULL));
  assert(!sigprocmask(SIG_SETMASK, &original, NULL));
  puts("signal overflow passed");
  return 0;
}
