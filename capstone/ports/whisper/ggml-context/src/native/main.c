#include "port.h"
#include <stdio.h>
#include <stdlib.h>
_Noreturn void wg_fail(unsigned code) {
  fprintf(stderr, "WG failure %u\n", code);
  exit(1);
}
int main(int argc, char **argv) {
  if (argc != 3)
    return 2;
  FILE *f = fopen(argv[1], "rb");
  struct wg_header *trace = malloc(WG_TRACE_BYTES);
  if (!f || !trace)
    return 2;
  size_t n = fread(trace, 1, WG_TRACE_BYTES, f);
  if (ferror(f) || fgetc(f) != EOF || n < sizeof *trace ||
      trace->magic != WG_MAGIC ||
      trace->count >
          (WG_TRACE_BYTES - sizeof *trace) / sizeof(struct wg_event) ||
      n != sizeof *trace + trace->count * sizeof(struct wg_event))
    return 3;
  fclose(f);
  void *payload = aligned_alloc(16, WG_PAYLOAD_BYTES);
  if (!payload)
    return 4;
  wg_init_backing(NULL, payload, 0);
  struct wg_header report = {0};
  wg_replay(trace, &report);
  f = fopen(argv[2], "wb");
  if (!f || fwrite(&report, sizeof report, 1, f) != 1 || fclose(f))
    return 5;
  printf(
      "WG completed=%llu objects=%llu checksum=%llu peak=%llu layout=%llu\n",
      (unsigned long long)report.completed, (unsigned long long)report.objects,
      (unsigned long long)report.checksum, (unsigned long long)report.peak_used,
      (unsigned long long)report.layout_checksum);
  free(payload);
  free(trace);
  return 0;
}
