#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "coremark_hostcall.h"

#define print_nobuf(...)   \
  do {                     \
    printf(__VA_ARGS__);   \
    fflush(stdout);        \
  } while (0)

static int fail_cleanup(const char *msg, unsigned long v) {
  fprintf(stderr, "coremark-host: %s (observed=%lu)\n", msg, v);
  capstone_cleanup();
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <coremark-domain.dom>\n", argv[0]);
    return 2;
  }

  if (capstone_init()) {
    fprintf(stderr, "coremark-host: failed to initialize Capstone\n");
    return 1;
  }

  /* Pure C-domain: no smode payload. */
  dom_id_t dom_id = create_dom(argv[1], NULL);
  if ((long)dom_id < 0)
    return fail_cleanup("create_dom failed", (unsigned long)dom_id);

  region_id_t meta_id    = create_region(HC_REGION_SIZE);
  region_id_t payload_id = create_region(HC_REGION_SIZE);

  struct hostcall_v0 *meta =
      (struct hostcall_v0 *)map_region(meta_id, HC_REGION_SIZE);
  char *pay = (char *)map_region(payload_id, HC_REGION_SIZE);

  if (!meta || !pay)
    return fail_cleanup("map_region failed", 0);

  memset(meta, 0, HC_REGION_SIZE);
  memset(pay,  0, HC_REGION_SIZE);

  /*
   * Both regions are INOUT+SHARED so the domain retains read/write access
   * across all dom_return_pending() yields (multiple ee_printf calls).
   */
  shared_region_annotated(dom_id, meta_id,    HC_ANNOTATION_PERM_INOUT, HC_ANNOTATION_REV_SHARED);
  shared_region_annotated(dom_id, payload_id, HC_ANNOTATION_PERM_INOUT, HC_ANNOTATION_REV_SHARED);

  print_nobuf("coremark-host: domain created, regions shared, starting benchmark\n");

  unsigned long ret;
  while ((ret = call_dom(dom_id)) == HC_V0_RET_PENDING) {
    /* Snapshot metadata before acting — it is INOUT and mutable. */
    hostcall_u64_t op  = meta->opcode;
    hostcall_u64_t off = meta->offset;
    hostcall_u64_t len = meta->length;

    if (op == HC_V0_OP_WRITE_STDOUT) {
      if (off <= HC_REGION_SIZE && len <= HC_REGION_SIZE &&
          off + len <= HC_REGION_SIZE) {
        ssize_t n = write(STDOUT_FILENO, pay + off, (size_t)len);
        meta->result = (hostcall_s64_t)n;
        meta->error  = (n < 0) ? (hostcall_s64_t)errno : 0LL;
      } else {
        meta->result = -1;
        meta->error  = 1;
      }
    }
    meta->phase = HC_V0_PHASE_RESP;
  }

  /* Flush the payload buffer accumulated by ee_printf during the run.
   * Do this before the return-value check so output is visible on error too. */
  hostcall_u64_t total = meta->length;
  if (total > 0 && total <= HC_REGION_SIZE) {
    ssize_t n = write(STDOUT_FILENO, pay, (size_t)total);
    (void)n;
    fflush(stdout);
  }

  /* ret = HC_V0_RET_DONE = 0 when CoreMark exits cleanly. */
  if (ret != HC_V0_RET_DONE)
    return fail_cleanup("unexpected final retval", ret);

  print_nobuf("coremark-host: benchmark complete (retval=%lu)\n", ret);

  if (capstone_cleanup()) {
    fprintf(stderr, "coremark-host: failed to clean up Capstone\n");
    return 1;
  }

  return 0;
}
