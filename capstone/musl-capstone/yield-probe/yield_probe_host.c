/* Host side of the resumable-yield probe: create the domain, share the two
 * HostCall v0 regions, then service requests until the domain says DONE.
 *
 * Snapshot discipline, per the HostCall v0 design note: the request fields are
 * copied out of the shared block immediately after call_dom() returns and the
 * host acts only on the copies. The metadata region stays INOUT+SHARED and the
 * domain may write it at any time, so re-reading it mid-service is a TOCTOU.
 *
 * The round loop is BOUNDED. A domain that never reaches DONE is a failure to
 * report, not a reason to hang the QEMU run until the harness timeout, which
 * would produce a log that cannot distinguish "wedged" from "looping".
 */
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "../../tests/runtime-qemu/hostcall-stdout-probe/hostcall_stdout_probe.h"

#define YIELD_PROBE_MAX_ROUNDS 8
#define YIELD_PROBE_REGION_SIZE HOSTCALL_STDOUT_PROBE_REGION_SIZE

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <yield-probe.dom>\n", argv[0]);
    return 2;
  }
  if (capstone_init()) {
    fprintf(stderr, "yield-probe: capstone_init failed\n");
    return 1;
  }

  dom_id_t domain = create_dom(argv[1], NULL);
  if ((long)domain < 0) {
    fprintf(stderr, "yield-probe: create_dom failed (%ld)\n", (long)domain);
    capstone_cleanup();
    return 1;
  }

  region_id_t metadata_region = create_region(YIELD_PROBE_REGION_SIZE);
  region_id_t payload_region = create_region(YIELD_PROBE_REGION_SIZE);
  struct hostcall_v0 *metadata =
      (struct hostcall_v0 *)map_region(metadata_region, YIELD_PROBE_REGION_SIZE);
  char *payload = (char *)map_region(payload_region, YIELD_PROBE_REGION_SIZE);
  if (!metadata || !payload) {
    fprintf(stderr, "yield-probe: map_region failed\n");
    capstone_cleanup();
    return 1;
  }
  memset(metadata, 0, YIELD_PROBE_REGION_SIZE);
  memset(payload, 0, YIELD_PROBE_REGION_SIZE);

  shared_region_annotated(domain, metadata_region,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED);
  shared_region_annotated(domain, payload_region,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED);

  unsigned serviced = 0;
  for (unsigned round = 0; round < YIELD_PROBE_MAX_ROUNDS; ++round) {
    (void)call_dom(domain);

    /* Snapshot before acting. */
    hostcall_u64_t phase = metadata->phase;
    hostcall_u64_t opcode = metadata->opcode;
    hostcall_u64_t offset = metadata->offset;
    hostcall_u64_t length = metadata->length;
    hostcall_s64_t result = metadata->result;

    if (phase == HC_V0_PHASE_DONE) {
      printf("yield-probe: DONE after %u serviced request(s), "
             "domain entered domain_main %lld time(s)\n",
             serviced, (long long)result);
      if (serviced == 2 && result == 1) {
        printf("__CAPSTONE_YIELD_PROBE_PASSED__\n");
        fflush(stdout);
        capstone_cleanup();
        return 0;
      }
      fprintf(stderr, "yield-probe: FAILED, expected 2 requests and 1 entry\n");
      break;
    }

    if (phase != HC_V0_PHASE_REQ || opcode != HC_V0_OP_WRITE_STDOUT) {
      fprintf(stderr, "yield-probe: unexpected phase=%llu opcode=%llu\n",
              (unsigned long long)phase, (unsigned long long)opcode);
      break;
    }

    if (offset > YIELD_PROBE_REGION_SIZE ||
        length > YIELD_PROBE_REGION_SIZE - offset) {
      fprintf(stderr, "yield-probe: request out of bounds\n");
      break;
    }

    ssize_t written = write(STDOUT_FILENO, payload + offset, (size_t)length);
    fflush(stdout);
    metadata->result = (hostcall_s64_t)written;
    metadata->error = written == (ssize_t)length ? 0 : 1;
    metadata->phase = HC_V0_PHASE_RESP;
    ++serviced;
  }

  fprintf(stderr, "yield-probe: did not reach DONE\n");
  capstone_cleanup();
  return 1;
}
