#define _POSIX_C_SOURCE 200809L
#include "capstone/linux-domain-fault.h"
#include <pthread.h>
#include <signal.h>
#include <stdlib.h>

void capstone_domain_exit_on_fault(unsigned long result,
                                   void (*cleanup)(void *), void *context) {
  if (result != CAPSTONE_DOMAIN_FAULT_RETVAL)
    return;
  if (cleanup)
    cleanup(context);
  /* No diagnostic I/O here: a closed/full output pipe must not turn process
   * termination into SIGPIPE or an indefinitely blocked write. */
  struct sigaction action = {0};
  sigset_t signals;
  action.sa_handler = SIG_DFL;
  sigemptyset(&action.sa_mask);
  sigaction(SIGSEGV, &action, NULL);
  sigemptyset(&signals);
  sigaddset(&signals, SIGSEGV);
  pthread_sigmask(SIG_UNBLOCK, &signals, NULL);
  raise(SIGSEGV);
  /* Fail closed if signal setup/delivery unexpectedly fails. Tests require
   * actual signal death, not this numerically similar exit status. */
  _Exit(128 + SIGSEGV);
}
