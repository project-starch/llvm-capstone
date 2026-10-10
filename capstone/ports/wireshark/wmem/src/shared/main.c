#include "port.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
_Noreturn void wm_fail(unsigned code) {
  fprintf(stderr, "WM failure %u\n", code);
  exit(1);
}
int main(int argc, char **argv) {
  if (argc != 3)
    return 2;
  FILE *f = fopen(argv[1], "rb");
  struct wm_header *trace = malloc(WM_TRACE_BYTES);
  if (!f || !trace)
    return 2;
  size_t n = fread(trace, 1, WM_TRACE_BYTES, f);
  if (ferror(f) || fgetc(f) != EOF || n < sizeof *trace ||
      trace->magic != WM_MAGIC ||
      trace->count >
          (WM_TRACE_BYTES - sizeof *trace) / sizeof(struct wm_event) ||
      n != sizeof *trace + trace->count * sizeof(struct wm_event))
    return 3;
  fclose(f);
  struct wm_header report = {0};
  wm_replay(trace, &report);
  f = fopen(argv[2], "wb");
  if (!f || fwrite(&report, sizeof report, 1, f) != 1 || fclose(f))
    return 5;
  printf("WM completed=%llu allocs=%llu checksum=%llu system=%llu peak=%llu\n",
         (unsigned long long)report.completed,
         (unsigned long long)report.allocs,
         (unsigned long long)report.checksum,
         (unsigned long long)report.system_allocs,
         (unsigned long long)report.system_peak);
  free(trace);
  return 0;
}
