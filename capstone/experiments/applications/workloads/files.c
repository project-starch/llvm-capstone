/* Calibration for the common file service: more than eight descriptors,
 * capacity exhaustion, data integrity and reuse after closing every handle. */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include "capstone/hostcall.h"
int main(int argc, char **argv) {
  if (argc != 2) return 64;
  int fds[HC_V0_FILE_SLOTS];
  for (int round = 0; round < 2; ++round) {
    for (int i = 0; i < HC_V0_FILE_SLOTS; ++i) {
      char c = 0;
      fds[i] = open(argv[1], O_RDONLY);
      if (fds[i] < 0 || read(fds[i], &c, 1) != 1 || c != 'x') return 1;
    }
    errno = 0;
    if (open(argv[1], O_RDONLY) != -1 || errno != EMFILE) return 2;
    for (int i = 0; i < HC_V0_FILE_SLOTS; ++i) if (close(fds[i])) return 3;
  }
  puts("EXP-OK file-capacity");
  return 0;
}
