/* Host controller for the xlang Capstone column. Modelled on
 * sqlite_host_row3.c; one host serves all 14 rows because the domain is
 * parameterised rather than per-row.
 *
 * Shares three regions, IN THIS ORDER (the domain captures them by share
 * order): 0 = host-call metadata, 1 = text payload, 2 = the linear arena the
 * allocator carves from.
 *
 * The arena is handed over REV_TRANSFERRED so the domain can MREV it and owns
 * the revoke outright. It is NEVER touched by the host after call_dom(): once
 * the domain revokes a REV_TRANSFERRED lineage the monitor holds a stale
 * untagged duplicate, and a host access to those pages aborts QEMU in
 * swap_cpmp()/cap_base() (task-007). We only read the payload region.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../../capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "sqlite_hostcall.h"

#define XLANG_ANNOTATION_PERM_INOUT 0x1UL
#define XLANG_ANNOTATION_REV_SHARED 0x2UL
#define XLANG_ANNOTATION_REV_TRANSFERRED 0x3UL

/* Row 10 carves ~5.3 KiB (state + context + 1024 stack + wedge + a 4096
 * realloc); the 2048-byte rows carve ~10.4 KiB. rof never coalesces, so the
 * arena only shrinks -- size for the widest row, with headroom. */
#ifndef XLANG_ARENA_SIZE
#define XLANG_ARENA_SIZE 65536UL
#endif

static int fail_cleanup(const char *message, unsigned long value) {
  fprintf(stderr, "xlang-host: %s (observed=%lu)\n", message, value);
  capstone_cleanup();
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <xlang-row.dom>\n", argv[0]);
    return 2;
  }
  if (capstone_init()) {
    fprintf(stderr, "xlang-host: failed to initialize Capstone\n");
    return 1;
  }

  dom_id_t domain = create_dom(argv[1], NULL);
  if ((long)domain < 0)
    return fail_cleanup("create_dom failed", (unsigned long)domain);

  region_id_t metadata_region = create_region(SQLITE_HC_REGION_SIZE);
  region_id_t payload_region = create_region(SQLITE_HC_REGION_SIZE);
  region_id_t arena_region = create_region(XLANG_ARENA_SIZE);

  struct sqlite_hostcall_v0 *metadata =
      (struct sqlite_hostcall_v0 *)map_region(metadata_region,
                                              SQLITE_HC_REGION_SIZE);
  char *payload = (char *)map_region(payload_region, SQLITE_HC_REGION_SIZE);
  if (!metadata || !payload)
    return fail_cleanup("map_region failed", 0);

  memset(metadata, 0, SQLITE_HC_REGION_SIZE);
  memset(payload, 0, SQLITE_HC_REGION_SIZE);

  shared_region_annotated(domain, metadata_region, XLANG_ANNOTATION_PERM_INOUT,
                          XLANG_ANNOTATION_REV_SHARED);
  shared_region_annotated(domain, payload_region, XLANG_ANNOTATION_PERM_INOUT,
                          XLANG_ANNOTATION_REV_SHARED);
  shared_region_annotated(domain, arena_region, XLANG_ANNOTATION_PERM_INOUT,
                          XLANG_ANNOTATION_REV_TRANSFERRED);

  unsigned long result = call_dom(domain);

  /* Flush whatever the domain wrote before it returned or faulted. Payload
   * only -- never the arena. */
  if (metadata->length > 0 && metadata->length <= SQLITE_HC_REGION_SIZE) {
    (void)write(STDOUT_FILENO, payload, (size_t)metadata->length);
    fflush(stdout);
  }

  /* Read this together with the output above:
   *   fault  + no XLANG line   -> the mechanism caught the stale access
   *   return + XLANG ... use-after-free-survived -> it did not
   * The no-revoke control MUST produce the second, or the first proves nothing
   * (ALLOCATOR-CONTRACT.md §4). */
  printf("xlang-host: call retval = 0x%08lx\n", result);
  fflush(stdout);

  capstone_cleanup();
  return 0;
}
