#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "linear_uninit_corpus_probe.h"

/* Controller for the LINEAR (row11) and UNINIT (row14) corpus probes.
 *
 * It does one thing: create a region and TRANSFER it, linear and RW, into a
 * single domain. Everything the two rows are about happens inside that domain --
 * it derives an uninitialised handle from its own grant (row14), or carves a
 * move-only statement capability and drops it (row11). No monitor op was added
 * for either row; the controller is the same one the held-cap probe uses, minus
 * its arena-lifecycle comment.
 *
 * The post-call read-back ("read-arena") is opt-in, for the row11 probes only,
 * never for row14 (which revokes the arena). It reads arena[CORPUS_OFFSET]
 * through the DOMAIN'S own surviving alias, via a second call_dom, exactly as
 * the sibling intra-domain-mrev-revoke controller does since Q-05. It used to
 * read the host's Linux mmap directly, which only ever worked because QEMU left
 * a stale tagged duplicate of a transferred region behind a linear move; Q-05
 * (2026-09-07, sbi_capstone.c REV_TRANSFERRED -> make_hole) removed that
 * duplicate, so a host mmap read of a transferred region now takes a monitor
 * cpmp-miss fault (CAPSTONE_NO_CPMP_REGION). The observer moved into the domain;
 * the probes' semantics are untouched. See probe_domain.h and
 * ../intra-domain-mrev-revoke-probe/README.md, "Gap found".
 *
 * Domain payloads are domain_main-style .dom images built with the Capstone
 * clang, loaded with create_dom(path, NULL). A probe that takes a capability
 * fault halts the domain and QEMU exits, so for the fault probes this program
 * never reaches its own exit; the evidence is the fault line in the serial log.
 */

#define print_nobuf(...) \
  do {                   \
    printf(__VA_ARGS__); \
    fflush(stdout);      \
  } while (0)

#define TAG "linear-uninit-corpus-probe"

static int fail_cleanup(const char *message, unsigned long observed) {
  fprintf(stderr, "%s: %s (observed=0x%016lx)\n", TAG, message, observed);
  fflush(stderr);
  capstone_cleanup();
  return 1;
}

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    fprintf(stderr, "usage: %s <domain.dom> [read-arena]\n", argv[0]);
    return 2;
  }
  int read_arena = (argc == 3 && strcmp(argv[2], "read-arena") == 0);

  if (capstone_init()) {
    fprintf(stderr, "%s: failed to initialize Capstone\n", TAG);
    return 1;
  }

  dom_id_t dom_id = create_dom(argv[1], NULL);
  if ((long)dom_id < 0)
    return fail_cleanup("create_dom failed", (unsigned long)dom_id);
  print_nobuf("%s: created domain ID = %lu\n", TAG, dom_id);

  region_id_t region_id = create_region(CORPUS_REGION_SIZE);
  unsigned char *region_bytes =
      (unsigned char *)map_region(region_id, CORPUS_REGION_SIZE);
  if (!region_bytes)
    return fail_cleanup("map_region failed", 0);
  memset(region_bytes, 0, CORPUS_REGION_SIZE);
  print_nobuf("%s: created region ID = %lu\n", TAG, region_id);

  /* LINEAR so the domain can mrev/split it, RW so cstighten does not silently
   * delinearise it, TRANSFERRED so the monitor keeps no revocation handle. */
  shared_region_annotated(dom_id, region_id, CORPUS_ANNOTATION_PERM_INOUT,
                          CORPUS_ANNOTATION_REV_TRANSFERRED);
  print_nobuf("%s: region transferred to domain (LIN, RW)\n", TAG);

  unsigned long retval = call_dom(dom_id);
  print_nobuf("%s: call retval = 0x%08lx\n", TAG, retval);
  /* Q-05: the arena was TRANSFERRED to the domain; reading it here through the
   * host's Linux mmap has no authority and only worked through a QEMU-only
   * duplicate the monitor no longer keeps. Ask the domain instead: a second call
   * returns arena[CORPUS_OFFSET] read through its own alias (see probe_domain.h,
   * probe_calls). (void)region_bytes: the mapping is kept only for the memset. */
  if (read_arena) {
    unsigned long byte = call_dom(dom_id);
    print_nobuf("%s: arena[%u] after call = 0x%02x (read back by the domain)\n", TAG,
                CORPUS_OFFSET, (unsigned)(byte & 0xffu));
  }
  (void)region_bytes;

  capstone_cleanup();
  return 0;
}
