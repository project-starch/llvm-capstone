/* Generic host servicer for a musl-in-a-domain program.
 *
 * Creates the domain, shares the two HostCall v0 regions in the order the
 * domain expects (metadata then payload), then services requests until the
 * domain reports DONE. Nothing here is program-specific, so one binary drives
 * any domain built on runtime/hostcall.c.
 *
 * Snapshot discipline, per the HostCall v0 design note: request fields are
 * copied out of the shared block immediately after call_dom() returns, because
 * the metadata region stays INOUT+SHARED and the domain may write it at any
 * time. Re-reading it mid-service is a TOCTOU.
 *
 * The round loop is BOUNDED. A domain that never reaches DONE must be reported
 * as such, not left to the harness timeout, which cannot distinguish a wedge
 * from a loop.
 */
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "../../tests/runtime-qemu/hostcall-stdout-probe/hostcall_stdout_probe.h"

#define HC_HOST_MAX_ROUNDS 256
#define HC_HOST_REGION_SIZE HOSTCALL_STDOUT_PROBE_REGION_SIZE

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <domain.dom>\n", argv[0]);
    return 2;
  }
  if (capstone_init()) {
    fprintf(stderr, "hostcall-host: capstone_init failed\n");
    return 1;
  }

  dom_id_t domain = create_dom(argv[1], NULL);
  if ((long)domain < 0) {
    fprintf(stderr, "hostcall-host: create_dom failed (%ld)\n", (long)domain);
    capstone_cleanup();
    return 1;
  }

  printf("hc-host: dom created id=%ld\n", (long)domain);
  fflush(stdout);

  region_id_t metadata_region = create_region(HC_HOST_REGION_SIZE);
  region_id_t payload_region = create_region(HC_HOST_REGION_SIZE);
  struct hostcall_v0 *metadata =
      (struct hostcall_v0 *)map_region(metadata_region, HC_HOST_REGION_SIZE);
  char *payload = (char *)map_region(payload_region, HC_HOST_REGION_SIZE);
  if (!metadata || !payload) {
    fprintf(stderr, "hostcall-host: map_region failed\n");
    capstone_cleanup();
    return 1;
  }
  printf("hc-host: regions mapped\n");
  fflush(stdout);
  memset(metadata, 0, HC_HOST_REGION_SIZE);
  memset(payload, 0, HC_HOST_REGION_SIZE);

  shared_region_annotated(domain, metadata_region,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED);
  shared_region_annotated(domain, payload_region,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED);

  /* PHASE MARKERS. A capability fault inside the monitor aborts QEMU outright,
     so the console shows an assertion and nothing about WHERE. libcapstone's
     last line ("Loadable size = ...") is printed BEFORE the create ioctl, which
     leaves create_dom, the two shares and call_dom indistinguishable -- four
     candidates for one bit of information. One line each turns that into a
     name, which is worth the noise. */
  printf("hc-host: shared, entering domain\n");
  fflush(stdout);

  unsigned serviced = 0;
  for (unsigned round = 0; round < HC_HOST_MAX_ROUNDS; ++round) {
    (void)call_dom(domain);

    hostcall_u64_t phase = metadata->phase;
    hostcall_u64_t opcode = metadata->opcode;
    hostcall_u64_t offset = metadata->offset;
    hostcall_u64_t length = metadata->length;
    hostcall_s64_t result = metadata->result;

    if (phase == HC_V0_PHASE_DONE) {
      printf("__CAPSTONE_HOSTCALL_HOST_DONE__ status=%lld serviced=%u\n",
             (long long)result, serviced);
      fflush(stdout);
      capstone_cleanup();
      return result == 0 ? 0 : 1;
    }

    if (phase != HC_V0_PHASE_REQ) {
      fprintf(stderr, "hostcall-host: unexpected phase=%llu\n",
              (unsigned long long)phase);
      break;
    }
    if (opcode != HC_V0_OP_WRITE_STDOUT) {
      /* Bound the trace: a displaced argument can feed our own return value back
         as a syscall number and loop. 24 lines is enough to see the pattern and
         short enough that the log stays readable. */
      static unsigned traced;
      if (++traced > 24) {
        fprintf(stderr, "hostcall-host: too many refusals, stopping\n");
        break;
      }
      /* Answer, do not guess. The domain's __capstone_hostcall only emits
         WRITE_STDOUT today; anything else means the two sides disagree, and an
         error is more useful than a plausible-looking success. */
      /* 0xE0/0xE1 are the domain saying "I refused a syscall"; `offset` carries
         the syscall number or fd it refused. Printed on stdout, not stderr, so
         it lands in the serial log next to the domain's own output. */
      printf("hc-host: kind=0x%llx nr/val=%lld arg0=%lld\n",
             (unsigned long long)opcode, (long long)offset, (long long)length);
      fflush(stdout);
      metadata->error = 1;
      metadata->result = -1;
      metadata->phase = HC_V0_PHASE_RESP;
      continue;
    }
    if (offset > HC_HOST_REGION_SIZE || length > HC_HOST_REGION_SIZE - offset) {
      fprintf(stderr, "hostcall-host: request out of bounds\n");
      break;
    }

    ssize_t written = write(STDOUT_FILENO, payload + offset, (size_t)length);
    fflush(stdout);
    metadata->result = (hostcall_s64_t)written;
    metadata->error = written == (ssize_t)length ? 0 : 1;
    metadata->phase = HC_V0_PHASE_RESP;
    ++serviced;
  }

  fprintf(stderr, "hostcall-host: did not reach DONE after %u serviced\n",
          serviced);
  capstone_cleanup();
  return 1;
}
