#include <stdio.h>

#include "../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"

#define BEEBS_RET_CORRECT 0xC171C0DEUL

static int fail_cleanup(const char *msg, unsigned long v) {
  fprintf(stderr, "beebs-tarai-host: %s (observed=0x%lx)\n", msg, v);
  capstone_cleanup();
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <beebs-tarai-domain.dom>\n", argv[0]);
    return 2;
  }

  if (capstone_init()) {
    fprintf(stderr, "beebs-tarai-host: failed to initialize Capstone\n");
    return 1;
  }

  dom_id_t dom_id = create_dom(argv[1], NULL);
  if ((long)dom_id < 0)
    return fail_cleanup("create_dom failed", (unsigned long)dom_id);

  unsigned long ret = call_dom(dom_id);
  if (ret != BEEBS_RET_CORRECT)
    return fail_cleanup("unexpected correctness marker", ret);

  printf("beebs-tarai-host: correctness marker validated (retval=0x%lx)\n",
         ret);

  if (capstone_cleanup()) {
    fprintf(stderr, "beebs-tarai-host: failed to clean up Capstone\n");
    return 1;
  }

  return 0;
}
