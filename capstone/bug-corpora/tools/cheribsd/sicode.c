/* What kind of CHERI fault ended a CheriBSD process, as one line on stderr.
 *
 * A purecap process that dies on a capability fault gets SIGPROT, and the shell sees only the
 * signal. What separates the cases a corpus cares about is si_code: PROT_CHERI_TAG for an access
 * through a capability whose tag is gone (a revocation sweep cleared it, which is how a
 * use-after-free is caught), PROT_CHERI_BOUNDS for an access outside the capability's bounds.
 *
 *   default             a shared object; LD_PRELOAD it in front of the program. Its constructor
 *                       installs the handler, which writes
 *                         SICODE signal=<n> si_code=<c> (<name>) addr=<address>
 *                       and then lets the signal kill the process as it would have.
 *   -DSICODE_SELFTEST   the positive control, a program with no handler of its own: run it
 *                       with the shared object preloaded, exactly as the program under test
 *                       runs. `bounds` reads past a 16-byte allocation, `revoked` frees one,
 *                       forces a revocation pass with malloc_revoke_quarantine_force_flush()
 *                       and reads through the stale pointer. Each must end with the matching
 *                       line; a run that prints nothing has shown that the handler or its
 *                       preload cannot be trusted, not that nothing faulted. With revocation
 *                       switched off for the process, `revoked` must complete instead.  */
#include <cheri/cheric.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef SICODE_SELFTEST
static const char *name(int code) {
  switch (code) {
#ifdef PROT_CHERI_BOUNDS
  case PROT_CHERI_BOUNDS: return "PROT_CHERI_BOUNDS";
#endif
#ifdef PROT_CHERI_TAG
  case PROT_CHERI_TAG: return "PROT_CHERI_TAG";
#endif
#ifdef PROT_CHERI_SEALED
  case PROT_CHERI_SEALED: return "PROT_CHERI_SEALED";
#endif
#ifdef PROT_CHERI_PERM
  case PROT_CHERI_PERM: return "PROT_CHERI_PERM";
#endif
  default: return "OTHER";
  }
}

static void report(int sig, siginfo_t *info, void *context) {
  (void)context;
  char line[160];
  int n = snprintf(line, sizeof line, "SICODE signal=%d si_code=%d (%s) addr=%#lx\n", sig,
                   info->si_code, sig == SIGPROT ? name(info->si_code) : "not SIGPROT",
                   (unsigned long)cheri_getaddress(info->si_addr));
  if (n > 0) write(2, line, (size_t)n);
  signal(sig, SIG_DFL);
  raise(sig);
}

__attribute__((constructor)) static void install(void) {
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_sigaction = report;
  sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
  sigaction(SIGPROT, &sa, NULL);
  sigaction(SIGSEGV, &sa, NULL);
  sigaction(SIGBUS, &sa, NULL);
}
#endif

#ifdef SICODE_SELFTEST
#include <malloc_np.h>  /* malloc_revoke_enabled */
int main(int argc, char **argv) {
  volatile char *p = malloc(16);
  if (argc != 2 || !p) return 2;
  if (!strcmp(argv[1], "bounds")) {
    printf("read %d\n", p[32]);
  } else if (!strcmp(argv[1], "revoked")) {
    /* With revocation off (_RUNTIME_REVOCATION_DISABLE=1) the same read must complete: that
       is the control that the per-process knob is live. */
    int on = malloc_revoke_enabled();
    printf("SELFTEST revocation is %s\n", on ? "on" : "off");
    p[0] = 1;
    free((void *)p);
    if (on && malloc_revoke_quarantine_force_flush() != 0) return 3;
    printf("read %d\n", p[0]);
  } else {
    return 2;
  }
  printf("SELFTEST no fault\n");
  return 0;
}
#endif
