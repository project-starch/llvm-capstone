/* Host controller for the real-Lua Capstone domain (lua_domain.c).
 *
 * A near-verbatim copy of xlang/capstone/xlang_shim_host.c: shares three regions
 * IN ORDER — 0 = hostcall metadata, 1 = text payload, 2 = the linear arena the
 * revoking allocator carves from — calls the domain, then flushes whatever the
 * domain wrote into the payload (the Lua result line, or a fault leaves it empty).
 *
 * The only substantive change is XLANG_ARENA_SIZE: real Lua's newstate + base lib
 * + the script churns far more heap than a hand C shim, and the rof allocator never
 * coalesces (the arena only ever shrinks), so the arena must cover the SUM of every
 * allocation the run makes, not the peak live set. 4 MiB is generous headroom on the
 * 8 GiB guest.
 *
 * The arena is REV_TRANSFERRED so the domain owns the revoke outright; the host must
 * never touch the arena pages after call_dom() (a revoked REV_TRANSFERRED lineage
 * leaves the monitor holding a stale untagged alias). We only read the payload.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../../../capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "sqlite_hostcall.h"

#define XLANG_ANNOTATION_PERM_INOUT 0x1UL
#define XLANG_ANNOTATION_REV_SHARED 0x2UL
#define XLANG_ANNOTATION_REV_TRANSFERRED 0x3UL

#ifndef XLANG_ARENA_SIZE
#define XLANG_ARENA_SIZE (4UL * 1024UL * 1024UL)
#endif

static int fail_cleanup(const char *message, unsigned long value) {
  fprintf(stderr, "lua-host: %s (observed=%lu)\n", message, value);
  capstone_cleanup();
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <lua-domain.dom>\n", argv[0]);
    return 2;
  }
  if (capstone_init()) {
    fprintf(stderr, "lua-host: failed to initialize Capstone\n");
    return 1;
  }

  dom_id_t domain = create_dom(argv[1], NULL);
  if ((long)domain < 0)
    return fail_cleanup("create_dom failed", (unsigned long)domain);
  fprintf(stderr, "HOST: create_dom ok id=%ld\n", (long)domain); /* stderr = unbuffered */

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

  fprintf(stderr, "HOST: share metadata (enters domain)\n");
  shared_region_annotated(domain, metadata_region, XLANG_ANNOTATION_PERM_INOUT,
                          XLANG_ANNOTATION_REV_SHARED);
  fprintf(stderr, "HOST: share payload (enters domain)\n");
  shared_region_annotated(domain, payload_region, XLANG_ANNOTATION_PERM_INOUT,
                          XLANG_ANNOTATION_REV_SHARED);
  fprintf(stderr, "HOST: share arena (enters domain)\n");
  shared_region_annotated(domain, arena_region, XLANG_ANNOTATION_PERM_INOUT,
                          XLANG_ANNOTATION_REV_TRANSFERRED);

  fprintf(stderr, "HOST: call_dom (run entry)\n");
  unsigned long result = call_dom(domain);
  fprintf(stderr, "HOST: call_dom returned %lu\n", result);

  /* Flush whatever the domain wrote before it returned or faulted. Payload only —
   * never the arena. A fault with an empty payload means the run never reached the
   * result line; a "LUA-OK result=400" line means the interpreter ran and the
   * realloc-moved table kept its data (and, for GC objects, its tags). */
  if (metadata->length > 0 && metadata->length <= SQLITE_HC_REGION_SIZE) {
    (void)write(STDOUT_FILENO, payload, (size_t)metadata->length);
    fflush(stdout);
  }

  printf("lua-host: call retval = 0x%08lx\n", result);
  fflush(stdout);

  capstone_cleanup();
  return 0;
}
