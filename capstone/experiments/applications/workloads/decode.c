/* Decode complete independent streams repeatedly in one process. The stock
 * native FFmpeg framemd5 output supplies the external content oracle. */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "ffapp_decode.h"
static void phase(const char *kind, int epoch) {
  char line[80];
  int n = snprintf(line, sizeof line, "MEMPHASE %s-%d\n", kind, epoch);
  write(2, line, n);
}
int main(int argc, char **argv) {
  if (argc != 3) return 64;
  int batches = atoi(argv[2]);
  if (batches < 1) return 64;
  for (int epoch = 0; epoch < batches; ++epoch) {
    phase("before", epoch);
    if (ffapp_run(argv[1], FFAPP_M5_ALL) != FFAPP_M5_ALL) return 1;
    phase("released", epoch);
  }
  printf("EXP-OK ffmpeg %d\n", batches);
  return 0;
}
