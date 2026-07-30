#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "sqlite_hostcall.h"

static int fail_cleanup(const char *message, unsigned long value) {
  fprintf(stderr, "sqlite-host: %s (observed=%lu)\n", message, value);
  capstone_cleanup();
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <sqlite-domain.dom>\n", argv[0]);
    return 2;
  }
  if (capstone_init()) {
    fprintf(stderr, "sqlite-host: failed to initialize Capstone\n");
    return 1;
  }

  dom_id_t domain = create_dom(argv[1], NULL);
  if ((long)domain < 0)
    return fail_cleanup("create_dom failed", (unsigned long)domain);

  /* PHASE MARKERS. On silicon a monitor fault is C_PRINT + while(1), and C_PRINT
     goes to the RTL trace, not the UART -- so a wedge in create_dom and a wedge in
     the domain's entry glue look identical from the console: silence after
     libcapstone's last line. These two lines separate them, which is the whole
     difference between debugging the monitor and debugging the glue. */
  fprintf(stderr, "sqlite-host: create_dom ok (id=%lu)\n", (unsigned long)domain);

  region_id_t metadata_region = create_region(SQLITE_HC_REGION_SIZE);
  region_id_t payload_region = create_region(SQLITE_HC_REGION_SIZE);
  struct sqlite_hostcall_v0 *metadata =
      (struct sqlite_hostcall_v0 *)map_region(metadata_region,
                                              SQLITE_HC_REGION_SIZE);
  char *payload = (char *)map_region(payload_region, SQLITE_HC_REGION_SIZE);
  if (!metadata || !payload)
    return fail_cleanup("map_region failed", 0);

  memset(metadata, 0, SQLITE_HC_REGION_SIZE);
  memset(payload, 0, SQLITE_HC_REGION_SIZE);
  shared_region_annotated(domain, metadata_region,
                          SQLITE_HC_ANNOTATION_PERM_INOUT,
                          SQLITE_HC_ANNOTATION_REV_SHARED);
  shared_region_annotated(domain, payload_region,
                          SQLITE_HC_ANNOTATION_PERM_INOUT,
                          SQLITE_HC_ANNOTATION_REV_SHARED);

  fprintf(stderr, "sqlite-host: regions shared, entering domain\n");
  unsigned long result = call_dom(domain);
  fprintf(stderr, "sqlite-host: domain returned\n");
  if (metadata->length > 0 && metadata->length <= SQLITE_HC_REGION_SIZE) {
    (void)write(STDOUT_FILENO, payload, (size_t)metadata->length);
    fflush(stdout);
  }
  if (result != SQLITE_HC_RET_DONE)
    return fail_cleanup("unexpected domain return", result);

  if (capstone_cleanup()) {
    fprintf(stderr, "sqlite-host: failed to clean up Capstone\n");
    return 1;
  }
  return 0;
}
