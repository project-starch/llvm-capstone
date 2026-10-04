/* B0 (docs/plans/b0-silicon-delegated-runtime.md): the smallest delegated application, built for silicon.
 * One write through the delegated runtime and a normal exit. The line is the whole result: a launcher that sees it
 * has run a gp-captable musl application end to end. */
#include <stdio.h>

int main(void) {
  puts("B0: hello from a gp-captable delegated application");
  return 0;
}
