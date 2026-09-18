/* Host side of the musl write probe: create the pure-capability domain, share
 * the two HostCall v0 regions, and service what musl's write() asks for.
 *
 * DERIVED FROM yield_probe_host.c, which drives the same resumable-yield
 * mechanism. The existing hostcall-stdout-probe host cannot be reused: it
 * drives an S-mode payload in a nested domain and yields with an SBI ecall,
 * which is not pure-capability and so cannot carry musl (see start-musl.S).
 *
 * THIS HOST COMPARES, IT DOES NOT ONLY PRINT. The payload region is shared
 * memory that starts out zeroed, so "the host printed something" is a weaker
 * claim than it looks. The bytes are checked against the one string both sides
 * take from write_probe.h, and a mismatch fails the run.
 *
 * Snapshot discipline per the HostCall v0 note: request fields are copied out
 * immediately after call_dom() and the host acts on the copies, because the
 * metadata region stays INOUT+SHARED and re-reading it mid-service is a TOCTOU.
 *
 * The round loop is BOUNDED, so a domain that never reaches DONE is a reported
 * failure rather than a hang that the harness timeout cannot tell from a stall.
 */
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* By -I, not by relative path. A git worktree of this repository has the
 * buildroot submodule present but EMPTY, so "../../../caplifive-buildroot/..."
 * resolves to nothing there and the probe cannot be built outside the main
 * clone. The build script points these at CAPSTONE_BUILDROOT_DIR instead, which
 * is the variable a worktree is supposed to set anyway. */
#include "libcapstone.h"
#include "hostcall_stdout_probe.h"
#include "write_probe.h"

#define WRITE_PROBE_MAX_ROUNDS 8
#define WRITE_PROBE_REGION_SIZE HOSTCALL_STDOUT_PROBE_REGION_SIZE

static const char *status_name(long long s) {
  switch (s) {
  case WP_OK:                return "OK";
  case WP_SHORT_WRITE:       return "SHORT_WRITE";
  case WP_WRITE_FAILED:      return "WRITE_FAILED";
  case WP_BADFD_SUCCEEDED:   return "BADFD_SUCCEEDED (fd check did not refuse)";
  case WP_BADFD_WRONG_ERRNO: return "BADFD_WRONG_ERRNO (errno lost at boundary)";
  default:                   return "UNKNOWN";
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <write_probe.dom>\n", argv[0]);
    return 2;
  }
  if (capstone_init()) {
    fprintf(stderr, "write-probe: capstone_init failed\n");
    return 1;
  }

  dom_id_t domain = create_dom(argv[1], NULL);
  if ((long)domain < 0) {
    fprintf(stderr, "write-probe: create_dom failed (%ld)\n", (long)domain);
    capstone_cleanup();
    return 1;
  }

  region_id_t metadata_region = create_region(WRITE_PROBE_REGION_SIZE);
  region_id_t payload_region = create_region(WRITE_PROBE_REGION_SIZE);
  struct hostcall_v0 *metadata =
      (struct hostcall_v0 *)map_region(metadata_region, WRITE_PROBE_REGION_SIZE);
  char *payload = (char *)map_region(payload_region, WRITE_PROBE_REGION_SIZE);
  if (!metadata || !payload) {
    fprintf(stderr, "write-probe: map_region failed\n");
    capstone_cleanup();
    return 1;
  }
  memset(metadata, 0, WRITE_PROBE_REGION_SIZE);
  memset(payload, 0, WRITE_PROBE_REGION_SIZE);

  shared_region_annotated(domain, metadata_region,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED);
  shared_region_annotated(domain, payload_region,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED);

  unsigned serviced = 0;
  int payload_matched = 0;

  for (unsigned round = 0; round < WRITE_PROBE_MAX_ROUNDS; ++round) {
    (void)call_dom(domain);

    hostcall_u64_t phase = metadata->phase;
    hostcall_u64_t opcode = metadata->opcode;
    hostcall_u64_t offset = metadata->offset;
    hostcall_u64_t length = metadata->length;
    hostcall_s64_t result = metadata->result;

    if (phase == HC_V0_PHASE_DONE) {
      printf("write-probe: DONE, serviced %u request(s), capstone_main = %lld (%s)\n",
             serviced, (long long)result, status_name((long long)result));
      fflush(stdout);
      if (serviced == MUSL_WRITE_PROBE_EXPECTED_ROUNDS && result == WP_OK &&
          payload_matched) {
        /* All three, deliberately: the right number of rounds says the chunk
           loop and the fd check behaved, the status says musl saw a well-formed
           return AND that errno survived the boundary, and the comparison says
           the bytes that arrived are the bytes that were sent. */
        printf("__CAPSTONE_MUSL_WRITE_PROBE_PASSED__\n");
        fflush(stdout);
        capstone_cleanup();
        return 0;
      }
      fprintf(stderr,
              "write-probe: FAILED (rounds=%u want %d, status=%lld want %d, payload_matched=%d)\n",
              serviced, MUSL_WRITE_PROBE_EXPECTED_ROUNDS, (long long)result,
              WP_OK, payload_matched);
      break;
    }

    if (phase != HC_V0_PHASE_REQ || opcode != HC_V0_OP_WRITE_STDOUT) {
      fprintf(stderr, "write-probe: unexpected phase=%llu opcode=%llu\n",
              (unsigned long long)phase, (unsigned long long)opcode);
      break;
    }

    if (offset > WRITE_PROBE_REGION_SIZE ||
        length > WRITE_PROBE_REGION_SIZE - offset) {
      fprintf(stderr, "write-probe: request out of bounds\n");
      break;
    }

    if (length == MUSL_WRITE_PROBE_MESSAGE_LEN &&
        memcmp(payload + offset, MUSL_WRITE_PROBE_MESSAGE,
               MUSL_WRITE_PROBE_MESSAGE_LEN) == 0) {
      payload_matched = 1;
    } else {
      fprintf(stderr, "write-probe: payload mismatch, length=%llu want %llu\n",
              (unsigned long long)length,
              (unsigned long long)MUSL_WRITE_PROBE_MESSAGE_LEN);
    }

    ssize_t written = write(STDOUT_FILENO, payload + offset, (size_t)length);
    fflush(stdout);
    metadata->result = (hostcall_s64_t)written;
    metadata->error = written == (ssize_t)length ? 0 : 1;
    metadata->phase = HC_V0_PHASE_RESP;
    ++serviced;
  }

  fprintf(stderr, "write-probe: did not reach DONE\n");
  capstone_cleanup();
  return 1;
}
