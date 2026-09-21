/* libcapstone's public header expects size_t to be declared by its caller. */
#include <stddef.h>

#include "libcapstone.h"
#include "port.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  if (argc != 5 || (strcmp(argv[4], "0") && strcmp(argv[4], "1")))
    return 2;
  setbuf(stdout, NULL);
  FILE *f = fopen(argv[2], "rb");
  struct aprp_header *staging = malloc(APRP_FILE_BYTES);
  if (!f || !staging)
    return 2;
  size_t bytes = fread(staging, 1, APRP_FILE_BYTES, f);
  if (ferror(f) || fgetc(f) != EOF || bytes < sizeof *staging ||
      staging->magic != APRP_MAGIC ||
      staging->count >
          (APRP_FILE_BYTES - sizeof *staging) / sizeof(struct aprp_event) ||
      bytes != sizeof *staging + staging->count * sizeof(struct aprp_event))
    return 3;
  fclose(f);
  if (capstone_init())
    return 4;
  dom_id_t dom = create_dom(argv[1], NULL);
  if ((long)dom < 0)
    return 5;
  region_id_t ro = create_region(4096), rm = create_region(APRP_META_BYTES);
  region_id_t rt = create_region(APRP_FILE_BYTES),
              ra = create_region(APRP_PAYLOAD_BYTES);
  if ((long)ro < 0 || (long)rm < 0 || (long)rt < 0 || (long)ra < 0)
    return 6;
  struct aprp_header *out = map_region(ro, 4096);
  void *input = map_region(rt, APRP_FILE_BYTES);
  if (!out || out == (void *)-1 || !input || input == (void *)-1)
    return 7;
  memset(out, 0, sizeof *out);
  out->mode = strtoul(argv[4], NULL, 10);
  memcpy(input, staging, bytes);
  shared_region_annotated(dom, ro, 1, 0);
  shared_region_annotated(dom, rm, 1, 0);
  shared_region_annotated(dom, rt, 1, 0);
  shared_region_annotated(dom, ra, 1, 1);
  unsigned long result = call_dom(dom);
  printf("APRP return=%lu status=%llu completed=%llu nodes=%llu "
         "reuses=%llu releases=%llu\n",
         result, (unsigned long long)out->status,
         (unsigned long long)out->completed, (unsigned long long)out->nodes,
         (unsigned long long)out->node_reuses,
         (unsigned long long)out->node_releases);
  f = fopen(argv[3], "wb");
  if (!f || fwrite(out, sizeof *out, 1, f) != 1 || fclose(f))
    return 8;
  int ok = result == 42046 && out->magic == APRP_MAGIC && !out->status &&
           out->completed == staging->count;
  capstone_cleanup();
  free(staging);
  return ok ? 0 : 1;
}
