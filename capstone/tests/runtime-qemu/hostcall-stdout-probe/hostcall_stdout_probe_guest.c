#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "hostcall_stdout_probe.h"

/* Print immediately so QEMU serial logs show the exact runtime sequence. */
#define print_nobuf(...)         \
  do {                           \
    printf(__VA_ARGS__);         \
    fflush(stdout);              \
  } while (0)

/*
 * Centralized failure path used after Capstone has already been initialized.
 * Dumping the metadata snapshot makes protocol mismatches easier to debug from
 * the captured serial log.
 */
static int fail_cleanup(const char *message, unsigned long observed,
                        struct hostcall_v0 *metadata) {
  fprintf(stderr, "hostcall-stdout-probe: %s (observed=%lu)\n", message,
          observed);
  if (metadata) {
    fprintf(stderr,
            "hostcall-stdout-probe: metadata{phase=%llu opcode=%llu offset=%llu "
            "length=%llu result=%lld error=%lld}\n",
            metadata->phase, metadata->opcode, metadata->offset,
            metadata->length, metadata->result, metadata->error);
  }
  capstone_cleanup();
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <smode-domain-path>\n", argv[0]);
    return 2;
  }

  if (capstone_init()) {
    fprintf(stderr, "hostcall-stdout-probe: failed to initialize Capstone\n");
    return 1;
  }

  /*
   * Reuse the existing /test-domains/sbi.dom C-domain substrate and swap in a
   * custom .smode payload that speaks the tiny HostCall v0 protocol.
   */
  dom_id_t dom_id = create_dom("/test-domains/sbi.dom", argv[1]);
  if ((long)dom_id < 0)
    return fail_cleanup("create_dom failed", (unsigned long)dom_id, NULL);

  /*
   * Create one shared control block and one payload buffer.
   *
   * The metadata remains shared across both rounds because both sides mutate the
   * state machine. The payload is stricter: the domain writes it in round 1, the
   * helper consumes it after return, and the runtime should revoke the domain's
   * borrowed access when that round completes.
   */
  region_id_t metadata_region_id =
      create_region(HOSTCALL_STDOUT_PROBE_REGION_SIZE);
  region_id_t payload_region_id = create_region(HOSTCALL_STDOUT_PROBE_REGION_SIZE);
  struct hostcall_v0 *metadata = (struct hostcall_v0 *)map_region(
      metadata_region_id, HOSTCALL_STDOUT_PROBE_REGION_SIZE);
  char *payload =
      (char *)map_region(payload_region_id, HOSTCALL_STDOUT_PROBE_REGION_SIZE);
  if (!metadata || !payload)
    return fail_cleanup("map_region failed", 0, metadata);

  /* Start from a zeroed state so every observed field is written deliberately. */
  memset(metadata, 0, HOSTCALL_STDOUT_PROBE_REGION_SIZE);
  memset(payload, 0, HOSTCALL_STDOUT_PROBE_REGION_SIZE);

  print_nobuf("hostcall-stdout-probe: created domain ID = %lu\n", dom_id);
  print_nobuf("hostcall-stdout-probe: metadata region ID = %lu\n",
              metadata_region_id);
  print_nobuf("hostcall-stdout-probe: payload region ID = %lu\n",
              payload_region_id);

  /*
   * Share metadata as INOUT+SHARED because the protocol header stays live across
   * both rounds. Share the payload as OUT+BORROWED because only the domain needs
   * to produce bytes before the first return.
   */
  shared_region_annotated(dom_id, metadata_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED);
  shared_region_annotated(dom_id, payload_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_OUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf(
      "hostcall-stdout-probe: metadata shared, payload borrowed-out\n");

  /*
   * Round 1: the domain is expected to populate metadata/payload and return a
   * scalar "pending" status to hand control back to the host helper.
   */
  unsigned long ret1 = call_dom(dom_id);
  print_nobuf("hostcall-stdout-probe: first call retval = %lu\n", ret1);
  print_nobuf("hostcall-stdout-probe: metadata phase after first call = %llu\n",
              metadata->phase);
  if (ret1 != HC_V0_RET_PENDING)
    return fail_cleanup("unexpected first call retval", ret1, metadata);
  if (metadata->phase != HC_V0_PHASE_REQ)
    return fail_cleanup("unexpected phase after first call",
                        (unsigned long)metadata->phase, metadata);
  if (metadata->opcode != HC_V0_OP_WRITE_STDOUT)
    return fail_cleanup("unexpected opcode after first call",
                        (unsigned long)metadata->opcode, metadata);
  if (metadata->offset != 0)
    return fail_cleanup("unexpected payload offset",
                        (unsigned long)metadata->offset, metadata);
  if (metadata->length != HOSTCALL_STDOUT_PROBE_MESSAGE_LEN)
    return fail_cleanup("unexpected payload length",
                        (unsigned long)metadata->length, metadata);
  if (metadata->offset + metadata->length > HOSTCALL_STDOUT_PROBE_REGION_SIZE)
    return fail_cleanup("payload range exceeds shared region",
                        (unsigned long)(metadata->offset + metadata->length),
                        metadata);

  /*
   * Service the only currently supported host request: read the domain-produced
   * payload once and write it to stdout from the helper side.
   */
  print_nobuf("hostcall-stdout-probe: servicing HC_V0_OP_WRITE_STDOUT\n");
  ssize_t write_result = write(STDOUT_FILENO, payload + metadata->offset,
                               (size_t)metadata->length);
  if (write_result < 0) {
    metadata->result = -1;
    metadata->error = errno;
    metadata->phase = HC_V0_PHASE_ERROR;
    return fail_cleanup("write to stdout failed", (unsigned long)errno,
                        metadata);
  }

  /* Write the host response back into the shared metadata block. */
  metadata->result = write_result;
  metadata->error = 0;
  metadata->phase = HC_V0_PHASE_RESP;

  /*
   * Round 2: the domain should only need the shared metadata response now. The
   * payload handoff was for round 1 only.
   */
  unsigned long ret2 = call_dom(dom_id);
  print_nobuf("hostcall-stdout-probe: second call retval = %lu\n", ret2);
  print_nobuf("hostcall-stdout-probe: metadata phase after second call = %llu\n",
              metadata->phase);
  if (ret2 != HC_V0_RET_DONE)
    return fail_cleanup("unexpected second call retval", ret2, metadata);
  if (metadata->phase != HC_V0_PHASE_DONE)
    return fail_cleanup("unexpected phase after second call",
                        (unsigned long)metadata->phase, metadata);
  if (metadata->result != HOSTCALL_STDOUT_PROBE_MESSAGE_LEN)
    return fail_cleanup("unexpected result after second call",
                        (unsigned long)metadata->result, metadata);
  if (metadata->error != 0)
    return fail_cleanup("unexpected error after second call",
                        (unsigned long)metadata->error, metadata);

  print_nobuf("hostcall-stdout-probe: success\n");

  if (capstone_cleanup()) {
    fprintf(stderr, "hostcall-stdout-probe: failed to clean up Capstone\n");
    return 1;
  }

  return 0;
}


