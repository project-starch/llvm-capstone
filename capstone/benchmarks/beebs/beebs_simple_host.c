#include <stdio.h>

#include "../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"

#define BEEBS_RET_CORRECT 0xC171C0DEUL

static int fail_cleanup(const char *bench, const char *msg, unsigned long v) {
  fprintf(stderr, "beebs-%s-host: %s (observed=0x%lx)\n", bench, msg, v);
  capstone_cleanup();
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <benchmark-name> <beebs-domain.dom>\n", argv[0]);
    return 2;
  }

  const char *bench = argv[1];
  const char *domain = argv[2];

  if (capstone_init()) {
    fprintf(stderr, "beebs-%s-host: failed to initialize Capstone\n", bench);
    return 1;
  }

  dom_id_t dom_id = create_dom(domain, NULL);
  if ((long)dom_id < 0)
    return fail_cleanup(bench, "create_dom failed", (unsigned long)dom_id);

  unsigned long ret = call_dom(dom_id);
  if (ret != BEEBS_RET_CORRECT)
    return fail_cleanup(bench, "unexpected correctness marker", ret);

  printf("beebs-%s-host: correctness marker validated (retval=0x%lx)\n",
         bench, ret);

  if (capstone_cleanup()) {
    fprintf(stderr, "beebs-%s-host: failed to clean up Capstone\n", bench);
    return 1;
  }

  return 0;
}
