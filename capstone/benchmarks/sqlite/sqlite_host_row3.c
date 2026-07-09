#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "sqlite_hostcall.h"

/* Controller for the row3 matched pair. Shares the two host-call regions
 * (metadata + payload, REV_SHARED) plus a THIRD region -- the linear arena the
 * domain carves its revocable column-name buffer from -- handed over
 * REV_TRANSFERRED / PERM_INOUT so the domain can MREV it and owns the revoke
 * outright. See sqlite_row3_domain.c.
 *
 * The arena region is NEVER touched by the host after call_dom(): once the
 * domain revokes a REV_TRANSFERRED lineage, the monitor's regions[] holds a
 * stale untagged duplicate and a host access to those pages aborts QEMU in
 * swap_cpmp()/cap_base() (task-007, "Gap found"). We only ever read the
 * host-call payload region, whose node stays live. */

#define ROW3_ANNOTATION_PERM_INOUT 0x1UL
#define ROW3_ANNOTATION_REV_SHARED 0x2UL
#define ROW3_ANNOTATION_REV_TRANSFERRED 0x3UL
#define ROW3_ARENA_SIZE 4096UL

static int fail_cleanup(const char *message, unsigned long value) {
  fprintf(stderr, "sqlite-row3-host: %s (observed=%lu)\n", message, value);
  capstone_cleanup();
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <sqlite-row3-domain.dom>\n", argv[0]);
    return 2;
  }
  if (capstone_init()) {
    fprintf(stderr, "sqlite-row3-host: failed to initialize Capstone\n");
    return 1;
  }

  dom_id_t domain = create_dom(argv[1], NULL);
  if ((long)domain < 0)
    return fail_cleanup("create_dom failed", (unsigned long)domain);

  region_id_t metadata_region = create_region(SQLITE_HC_REGION_SIZE);
  region_id_t payload_region = create_region(SQLITE_HC_REGION_SIZE);
  region_id_t arena_region = create_region(ROW3_ARENA_SIZE);
  struct sqlite_hostcall_v0 *metadata =
      (struct sqlite_hostcall_v0 *)map_region(metadata_region,
                                              SQLITE_HC_REGION_SIZE);
  char *payload = (char *)map_region(payload_region, SQLITE_HC_REGION_SIZE);
  if (!metadata || !payload)
    return fail_cleanup("map_region failed", 0);

  memset(metadata, 0, SQLITE_HC_REGION_SIZE);
  memset(payload, 0, SQLITE_HC_REGION_SIZE);

  /* Order matters: the domain captures regions in share order
   * (0=metadata, 1=payload, 2=arena). */
  shared_region_annotated(domain, metadata_region, ROW3_ANNOTATION_PERM_INOUT,
                          ROW3_ANNOTATION_REV_SHARED);
  shared_region_annotated(domain, payload_region, ROW3_ANNOTATION_PERM_INOUT,
                          ROW3_ANNOTATION_REV_SHARED);
  /* The arena: LINEAR (MREV-able) + RW (cstighten won't delinearise it), no
   * monitor-retained handle -- the single-domain shape. */
  shared_region_annotated(domain, arena_region, ROW3_ANNOTATION_PERM_INOUT,
                          ROW3_ANNOTATION_REV_TRANSFERRED);

  unsigned long result = call_dom(domain);

  /* Flush whatever the domain wrote before it either returned or faulted. Only
   * the payload region -- never the arena. */
  if (metadata->length > 0 && metadata->length <= SQLITE_HC_REGION_SIZE) {
    (void)write(STDOUT_FILENO, payload, (size_t)metadata->length);
    fflush(stdout);
  }
  /* The fault variant never returns here (the domain halts and QEMU exits); the
   * evidence is the monitor's fault line. The no-revoke control returns DONE. */
  printf("sqlite-row3-host: call retval = 0x%08lx\n", result);
  fflush(stdout);

  capstone_cleanup();
  return 0;
}
