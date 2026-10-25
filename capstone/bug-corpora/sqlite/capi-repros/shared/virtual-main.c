/* The reach marker for a capi row on the virtual profile, linked with -Wl,--wrap=main.
 *
 * A row's before.c is a whole program with no harness, and this corpus does not edit its rows. So
 * the runner wraps main: the runtime's call to main lands here, which prints `CAPI BEGIN` before the
 * row runs and `CAPI RETURNED <rc>` if main returns. BEGIN marks main's ENTRY, not the defective
 * access, and every reading says so (shared/run-virtual.py).
 */
#include <stdio.h>

int __real_main(int argc, char **argv, char **envp);

int __wrap_main(int argc, char **argv, char **envp) {
  printf("CAPI BEGIN\n");
  fflush(stdout);
  int rc = __real_main(argc, argv, envp);
  printf("CAPI RETURNED %d\n", rc);
  fflush(stdout);
  return rc;
}
