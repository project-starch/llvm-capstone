#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "sqlite_hostcall.h"

/* Controller for the row3 fork B2 matched pair. Like sqlite_host_row3.c it
 * shares the two host-call regions (metadata + payload, REV_SHARED) plus a THIRD
 * region -- the linear arena -- handed over REV_TRANSFERRED / PERM_INOUT.
 *
 * The difference from B1's host is the arena size. In B1 the arena backs only a
 * single carved column-name buffer (4 KiB is plenty). In B2 the arena is
 * SQLite's ENTIRE heap, and the allocator never coalesces, so it must hold
 * row3's whole CUMULATIVE allocation, not its peak. Size it generously; the
 * kernel backs a region with __get_free_pages, whose practical ceiling is a few
 * MiB, so this is a few MiB, not more. If a heavier workload exhausts it,
 * rof_malloc returns 0 and SQLite reports SQLITE_NOMEM -- the honest
 * non-coalescing limit measured in Phase 2.
 *
 * The arena region is NEVER touched by the host after call_dom() (task-007 "Gap
 * found"). We only read the host-call payload region, whose node stays live.
 */

#define ROW3_ANNOTATION_PERM_INOUT 0x1UL
#define ROW3_ANNOTATION_REV_SHARED 0x2UL
#define ROW3_ANNOTATION_REV_TRANSFERRED 0x3UL

#ifndef ROW3_B2_ARENA_SIZE
#define ROW3_B2_ARENA_SIZE (4UL * 1024UL * 1024UL)
#endif

static int fail_cleanup(const char *message, unsigned long value) {
  fprintf(stderr, "sqlite-row3-b2-host: %s (observed=%lu)\n", message, value);
  capstone_cleanup();
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <sqlite-row3-b2-domain.dom>\n", argv[0]);
    return 2;
  }
  if (capstone_init()) {
    fprintf(stderr, "sqlite-row3-b2-host: failed to initialize Capstone\n");
    return 1;
  }

  dom_id_t domain = create_dom(argv[1], NULL);
  if ((long)domain < 0)
    return fail_cleanup("create_dom failed", (unsigned long)domain);

  region_id_t metadata_region = create_region(SQLITE_HC_REGION_SIZE);
  region_id_t payload_region = create_region(SQLITE_HC_REGION_SIZE);
  region_id_t arena_region = create_region(ROW3_B2_ARENA_SIZE);
  if ((long)arena_region < 0)
    return fail_cleanup("create_region(arena) failed", (unsigned long)arena_region);

  struct sqlite_hostcall_v0 *metadata =
      (struct sqlite_hostcall_v0 *)map_region(metadata_region,
                                              SQLITE_HC_REGION_SIZE);
  char *payload = (char *)map_region(payload_region, SQLITE_HC_REGION_SIZE);
  if (!metadata || !payload)
    return fail_cleanup("map_region failed", 0);

  memset(metadata, 0, SQLITE_HC_REGION_SIZE);
  memset(payload, 0, SQLITE_HC_REGION_SIZE);

  /* Share order is the capture order in domain_main: 0=metadata, 1=payload,
   * 2=arena. */
  shared_region_annotated(domain, metadata_region, ROW3_ANNOTATION_PERM_INOUT,
                          ROW3_ANNOTATION_REV_SHARED);
  shared_region_annotated(domain, payload_region, ROW3_ANNOTATION_PERM_INOUT,
                          ROW3_ANNOTATION_REV_SHARED);
  shared_region_annotated(domain, arena_region, ROW3_ANNOTATION_PERM_INOUT,
                          ROW3_ANNOTATION_REV_TRANSFERRED);

  unsigned long result = call_dom(domain);

  if (metadata->length > 0 && metadata->length <= SQLITE_HC_REGION_SIZE) {
    (void)write(STDOUT_FILENO, payload, (size_t)metadata->length);
    fflush(stdout);
  }
  printf("sqlite-row3-b2-host: call retval = 0x%08lx\n", result);
  fflush(stdout);

  capstone_cleanup();
  return 0;
}
