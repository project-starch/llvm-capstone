#include <stdio.h>
#include <stdlib.h>

#include "../../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "shared_region_probe.h"

#define print_nobuf(...)         \
  do {                           \
    printf(__VA_ARGS__);         \
    fflush(stdout);              \
  } while (0)

#define PROBE_ANNOTATION_PERM_INOUT 0x1
#define PROBE_ANNOTATION_REV_SHARED 0x2

static int fail_cleanup(const char *message, unsigned long observed) {
  fprintf(stderr, "shared-region-probe: %s (observed=0x%016lx)\n", message,
          observed);
  capstone_cleanup();
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <smode-domain-path>\n", argv[0]);
    return 2;
  }

  if (capstone_init()) {
    fprintf(stderr, "shared-region-probe: failed to initialize Capstone\n");
    return 1;
  }

  dom_id_t dom_id = create_dom("/test-domains/sbi.dom", argv[1]);
  if ((long)dom_id < 0)
    return fail_cleanup("create_dom failed", (unsigned long)dom_id);

  region_id_t region_id = create_region(SHARED_REGION_PROBE_REGION_SIZE);
  unsigned long *region_words =
      (unsigned long *)map_region(region_id, SHARED_REGION_PROBE_REGION_SIZE);
  if (!region_words)
    return fail_cleanup("map_region failed", 0);

  region_words[0] = 0;

  print_nobuf("shared-region-probe: created domain ID = %lu\n", dom_id);
  print_nobuf("shared-region-probe: created shared region ID = %lu\n", region_id);
  print_nobuf("shared-region-probe: initial word = 0x%016lx\n", region_words[0]);

  shared_region_annotated(dom_id, region_id, PROBE_ANNOTATION_PERM_INOUT,
                          PROBE_ANNOTATION_REV_SHARED);
  print_nobuf("shared-region-probe: region shared via annotated path\n");

  unsigned long ret1 = call_dom(dom_id);
  print_nobuf("shared-region-probe: call 1 retval = %lu\n", ret1);
  print_nobuf("shared-region-probe: word after call 1 = 0x%016lx\n",
              region_words[0]);
  if (region_words[0] != SHARED_REGION_PROBE_SENTINEL_STAGE1)
    return fail_cleanup("stage 1 sentinel mismatch", region_words[0]);

  unsigned long ret2 = call_dom(dom_id);
  print_nobuf("shared-region-probe: call 2 retval = %lu\n", ret2);
  print_nobuf("shared-region-probe: word after call 2 = 0x%016lx\n",
              region_words[0]);
  if (region_words[0] != SHARED_REGION_PROBE_SENTINEL_STAGE2)
    return fail_cleanup("stage 2 sentinel mismatch", region_words[0]);

  print_nobuf("shared-region-probe: success\n");

  if (capstone_cleanup()) {
    fprintf(stderr, "shared-region-probe: failed to clean up Capstone\n");
    return 1;
  }

  return 0;
}

