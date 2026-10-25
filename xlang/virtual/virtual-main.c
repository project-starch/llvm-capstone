/* The reach marker for an xlang shim on the virtual profile, linked with -Wl,--wrap=main.
 *
 * A shim is a whole program whose only print (MOCK ...) comes after its stale access, and its rows are not edited here. So
 * the runner wraps main: the runtime's call to main lands here, which prints `XLANG BEGIN` before the
 * row runs and `XLANG RETURNED <rc>` if main returns. BEGIN marks main's ENTRY, not the defective
 * access, and every reading says so (xlang/virtual/run-virtual.py).
 */
#include <stdio.h>

int __real_main(int argc, char **argv, char **envp);

int __wrap_main(int argc, char **argv, char **envp) {
  printf("XLANG BEGIN\n");
  fflush(stdout);
  int rc = __real_main(argc, argv, envp);
  printf("XLANG RETURNED %d\n", rc);
  fflush(stdout);
  return rc;
}
