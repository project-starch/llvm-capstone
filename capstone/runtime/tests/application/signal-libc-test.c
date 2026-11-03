/* Run as a native RV64 Linux process, with --wrap=pthread_create. Each case
 * starts before libc's first pthread and installs the launcher's real filter. */
#define _GNU_SOURCE
#include "../../linux/delegate-service.h"
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__riscv) && __riscv_xlen == 64
static unsigned creates;
static int fail_next;
static uint64_t worker_mask;
static void *(*worker_entry)(void *);
static void *worker_arg;
int __real_pthread_create(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);

static void *inspect_worker(void *unused) {
  (void)unused;
  assert(!syscall(SYS_rt_sigprocmask, SIG_BLOCK, NULL, &worker_mask, sizeof worker_mask));
  return worker_entry(worker_arg);
}

int __wrap_pthread_create(pthread_t *t, const pthread_attr_t *a, void *(*fn)(void *), void *arg) {
  ++creates;
  if (fail_next) { fail_next = 0; return EAGAIN; }
  /* The test joins every worker before starting another. */
  worker_entry = fn;
  worker_arg = arg;
  return __real_pthread_create(t, a, inspect_worker, NULL);
}

static uint64_t bit(int sig) { return UINT64_C(1) << (sig - 1); }
struct action { void *handler; unsigned long flags; uint64_t mask; };
static struct action action_now(int sig) {
  struct action a;
  assert(!syscall(SYS_rt_sigaction, sig, NULL, &a, sizeof a.mask));
  return a;
}
static void action_set(int sig, void *handler) {
  struct action a = {handler, 0, 0};
  assert(!syscall(SYS_rt_sigaction, sig, &a, NULL, sizeof a.mask));
}
static uint64_t mask_now(void) {
  uint64_t mask;
  assert(!syscall(SYS_rt_sigprocmask, SIG_BLOCK, NULL, &mask, sizeof mask));
  return mask;
}
static void *nothing(void *p) { return p; }

static void check_case(int which) {
  struct capstone_signal_state s;
  for (int sig = 32; sig <= 34; ++sig) action_set(sig, (void *)SIG_DFL);
  capstone_signals_set_kernel_mask(0);
  if (which == 1) capstone_signals_set_kernel_mask(bit(33));
  if (which == 2) action_set(33, (void *)SIG_IGN);
  assert(!capstone_delegate_seccomp());
  assert(!capstone_signals_prepare_libc());
  assert(creates == (unsigned)(which == 1 || which == 2));
  assert(mask_now() == (which == 1 ? bit(33) : 0));
  assert(action_now(33).handler == (which == 2 ? (void *)SIG_IGN : (void *)SIG_DFL));
  capstone_signals_init(&s, NULL);
  if (which == 1 || which == 2) return;

  if (which == 0) {
    /* Ordinary signal handling and masking do not create a worker. Later
     * initialization cannot deliver a caught ordinary signal to that worker. */
    assert(!capstone_signals_action(&s, SIGUSR1, CAPSTONE_SIGNAL_CAUGHT, 0));
    assert(!capstone_signals_action(&s, SIGUSR2, CAPSTONE_SIGNAL_CAUGHT, 0));
    uint64_t mask = bit(SIGUSR1);
    assert(!capstone_signals_procmask(&s, SIG_BLOCK, &mask, NULL));
    assert(!creates);
    uint64_t reserved = bit(33);
    assert(capstone_signals_procmask(&s, -1, &reserved, NULL) == -EINVAL);
    assert(!creates && mask_now() == bit(SIGUSR1));
    assert(!capstone_signals_settle_libc());
    assert(!capstone_signals_settle_libc());
    assert(creates == 1 && (worker_mask & bit(SIGUSR1)) && (worker_mask & bit(SIGUSR2)));
    assert(mask_now() == bit(SIGUSR1));
    assert(action_now(33).handler == (void *)SIG_DFL);
    return;
  }

  if (which == 3) {
    fail_next = 1;
    assert(capstone_signals_action(&s, 33, CAPSTONE_SIGNAL_CAUGHT, 0) == -EAGAIN);
    assert(creates == 1 && action_now(33).handler == (void *)SIG_DFL);
    assert(!mask_now());
    assert(!capstone_signals_action(&s, 33, CAPSTONE_SIGNAL_CAUGHT, 0));
    assert(creates == 2);
  }
  uint64_t mask = bit(33);
  assert(!capstone_signals_procmask(&s, SIG_BLOCK, &mask, NULL));
  assert(creates == (unsigned)(which == 3 ? 2 : 1));
  assert(!capstone_signals_action(&s, 33, CAPSTONE_SIGNAL_CAUGHT, 0));
  void *handler = action_now(33).handler;
  assert(handler != (void *)SIG_DFL && handler != (void *)SIG_IGN);
  pthread_t worker;
  assert(!pthread_create(&worker, NULL, nothing, NULL));
  assert(!pthread_join(worker, NULL));
  assert(action_now(33).handler == handler && mask_now() == bit(33));
  assert(!syscall(SYS_tgkill, getpid(), syscall(SYS_gettid), 33));
  assert(!atomic_load(&s.head));
  assert(!capstone_signals_procmask(&s, SIG_UNBLOCK, &mask, NULL));
  assert(atomic_load(&s.head) == 1 && s.ring[0].signo == 33);
}
#endif

int main(void) {
#if defined(__riscv) && __riscv_xlen == 64
  for (int which = 0; which < 5; ++which) {
    pid_t child = fork();
    assert(child >= 0);
    if (!child) { check_case(which); _exit(0); }
    int status;
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  }
  puts("SIGNAL_LIBC_OK cases=5");
  return 0;
#else
  puts("signal libc initialization requires native RV64 Linux");
  return 77;
#endif
}
