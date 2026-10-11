/* Cost of one delegated round: N calls of the cheapest syscall. The launcher
 * counts rounds and ticks; this program only makes the calls. */
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
  long n = argc > 1 ? atol(argv[1]) : 10000;
  volatile long sink = 0;
  for (long i = 0; i < n; ++i)
    sink += getpid();
  return sink ? 0 : 1;
}
