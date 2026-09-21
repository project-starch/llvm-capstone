#include "port.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef WM_POISONCAP
#include "poisoncap.h"
#endif
_Noreturn void wm_fail(unsigned code) {
  fprintf(stderr, "WM failure %u\n", code);
  exit(1);
}
int main(int argc, char **argv) {
  unsigned mode = 0;
#ifdef WM_POISONCAP
  /* The PoisonCap replay defaults to the protected mode; an explicit final
   * argument selects the spatial control. */
  mode = 1;
  if (argc == 4) {
    if (strcmp(argv[3], "0") && strcmp(argv[3], "1"))
      return 2;
    mode = argv[3][0] - '0';
  } else if (argc != 3)
    return 2;
#else
  if (argc != 3)
    return 2;
#endif
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
#ifdef WM_POISONCAP
  void *payload = NULL;
  wm_init_backing(NULL, NULL, mode);
#else
  void *payload = aligned_alloc(16, WM_PAYLOAD_BYTES);
  if (!payload)
    return 4;
  wm_init_backing(NULL, payload, mode);
#endif
  struct wm_header report = {0};
  report.mode = mode;
  wm_replay(trace, &report);
  f = fopen(argv[2], "wb");
  if (!f || fwrite(&report, sizeof report, 1, f) != 1 || fclose(f))
    return 5;
  printf("WM completed=%llu allocs=%llu checksum=%llu regions=%llu peak=%llu\n",
         (unsigned long long)report.completed,
         (unsigned long long)report.allocs,
         (unsigned long long)report.checksum,
         (unsigned long long)report.regions_created,
         (unsigned long long)report.regions_peak);
#ifdef WM_POISONCAP
  wm_poisoncap_report();
#endif
  free(payload);
  free(trace);
  return 0;
}
