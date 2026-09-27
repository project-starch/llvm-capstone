/* Whole FFmpeg decoder with the existing PoisonCap pool adapter.
 * Mode 0 and mode 2 use the same image and 4 MiB pool mapping. stdout remains
 * the decoded-frame oracle; pool policy and storage accounting go to stderr. */
#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "ffapp_decode.h"
#include "trace.h"

static void phase(const char *name, int batch) {
  char marker[80];
  int len = snprintf(marker, sizeof marker, "MEMPHASE %s-%d\n", name, batch);
  if (len <= 0 || len >= (int)sizeof marker || write(2, marker, (size_t)len) != len)
    exit(67);
  struct ff2_header h = {0};
  ff2_memory_report(&h);
  fprintf(stderr, "FFPOOL-MEM phase=%s-%d payload_used=%llu snapshots=%llu sweeps=%llu poisoned=%llu\n",
          name, batch, (unsigned long long)h.payload_used,
          (unsigned long long)h.reserved[2], (unsigned long long)h.reserved[0],
          (unsigned long long)h.reserved[1]);
}

int main(int argc, char **argv) {
  if (argc != 4) return 64;
  int batches = atoi(argv[2]);
  int mode = atoi(argv[3]);
  if (batches < 1 || (mode != 0 && mode != 2)) return 64;
  void *pool = mmap(NULL, 4UL*1024*1024, PROT_READ|PROT_WRITE,
                    MAP_PRIVATE|MAP_ANON, -1, 0);
  if (pool == MAP_FAILED) return 65;
  ff2_payload_init(pool, 4UL*1024*1024);
  ff2_set_mode((unsigned)mode);
  fprintf(stderr, "FFPOOL-POLICY mode=%d payload_reservation=4194304\n", mode);
  for (int batch = 0; batch < batches; batch++) {
    phase("before", batch);
    if (ffapp_run(argv[1], FFAPP_M5_ALL) != FFAPP_M5_ALL) return 1;
    phase("released", batch);
  }
  printf("EXP-OK ffmpeg %d\n", batches);
  return munmap(pool, 4UL*1024*1024) == 0 ? 0 : 66;
}
