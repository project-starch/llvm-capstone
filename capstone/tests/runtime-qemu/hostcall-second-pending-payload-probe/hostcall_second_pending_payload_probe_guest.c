#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "../hostcall-stdout-probe/hostcall_stdout_probe.h"

#define print_nobuf(...)         \
  do {                           \
    printf(__VA_ARGS__);         \
    fflush(stdout);              \
  } while (0)

#define HOSTCALL_SECOND_PENDING_PAYLOAD_STAGE1_RESULT 111LL
#define HOSTCALL_SECOND_PENDING_PAYLOAD_STAGE2_RESULT 222LL
#define HOSTCALL_SECOND_PENDING_PAYLOAD_STAGE1_TEXT "stage1-payload"
#define HOSTCALL_SECOND_PENDING_PAYLOAD_STAGE2_TEXT "stage2-payload"
#define HOSTCALL_SECOND_PENDING_PAYLOAD_STAGE1_LEN \
  (sizeof(HOSTCALL_SECOND_PENDING_PAYLOAD_STAGE1_TEXT) - 1)
#define HOSTCALL_SECOND_PENDING_PAYLOAD_STAGE2_LEN \
  (sizeof(HOSTCALL_SECOND_PENDING_PAYLOAD_STAGE2_TEXT) - 1)

static int fail_cleanup(const char *message, unsigned long observed,
                        const struct hostcall_v0 *metadata) {
  fprintf(stderr,
          "hostcall-second-pending-payload-probe: %s (observed=%lu)\n",
          message, observed);
  if (metadata) {
    fprintf(stderr,
            "hostcall-second-pending-payload-probe: metadata{phase=%llu opcode=%llu offset=%llu length=%llu result=%lld error=%lld}\n",
            metadata->phase, metadata->opcode, metadata->offset,
            metadata->length, metadata->result, metadata->error);
  }
  capstone_cleanup();
  return 1;
}

static void snapshot_request(struct hostcall_v0 *snapshot,
                             const struct hostcall_v0 *metadata) {
  snapshot->phase = metadata->phase;
  snapshot->opcode = metadata->opcode;
  snapshot->offset = metadata->offset;
  snapshot->length = metadata->length;
  snapshot->result = metadata->result;
  snapshot->error = metadata->error;
}

static int snapshot_payload(char *dest, const char *payload,
                            const struct hostcall_v0 *request) {
  hostcall_u64_t end = request->offset + request->length;
  hostcall_u64_t i;

  if (request->offset > HOSTCALL_STDOUT_PROBE_REGION_SIZE)
    return 0;
  if (request->length > HOSTCALL_STDOUT_PROBE_REGION_SIZE)
    return 0;
  if (end > HOSTCALL_STDOUT_PROBE_REGION_SIZE)
    return 0;

  for (i = 0; i < request->length; ++i)
    dest[i] = payload[request->offset + i];
  dest[request->length] = '\0';
  return 1;
}

static int payload_matches(const char *lhs, const char *rhs,
                           hostcall_u64_t len) {
  hostcall_u64_t i;
  for (i = 0; i < len; ++i) {
    if (lhs[i] != rhs[i])
      return 0;
  }
  return 1;
}

int main(int argc, char **argv) {
  dom_id_t dom_id;
  region_id_t metadata_region_id;
  region_id_t payload_region_id;
  struct hostcall_v0 *metadata;
  char *payload;
  struct hostcall_v0 request;
  char payload_snapshot[HOSTCALL_STDOUT_PROBE_REGION_SIZE + 1];
  unsigned long ret1;
  unsigned long ret2;
  unsigned long ret3;

  if (argc != 2) {
    fprintf(stderr, "usage: %s <smode-domain-path>\n", argv[0]);
    return 2;
  }

  if (capstone_init()) {
    fprintf(stderr,
            "hostcall-second-pending-payload-probe: failed to initialize Capstone\n");
    return 1;
  }

  dom_id = create_dom("/test-domains/sbi.dom", argv[1]);
  if ((long)dom_id < 0)
    return fail_cleanup("create_dom failed", (unsigned long)dom_id, NULL);

  metadata_region_id = create_region(HOSTCALL_STDOUT_PROBE_REGION_SIZE);
  payload_region_id = create_region(HOSTCALL_STDOUT_PROBE_REGION_SIZE);
  metadata = (struct hostcall_v0 *)map_region(metadata_region_id,
                                              HOSTCALL_STDOUT_PROBE_REGION_SIZE);
  payload = (char *)map_region(payload_region_id,
                               HOSTCALL_STDOUT_PROBE_REGION_SIZE);
  if (!metadata || !payload)
    return fail_cleanup("map_region failed", 0, metadata);

  memset(metadata, 0, HOSTCALL_STDOUT_PROBE_REGION_SIZE);
  memset(payload, 0, HOSTCALL_STDOUT_PROBE_REGION_SIZE);

  print_nobuf(
      "hostcall-second-pending-payload-probe: created domain ID = %lu\n", dom_id);
  print_nobuf(
      "hostcall-second-pending-payload-probe: metadata region ID = %lu\n",
      metadata_region_id);
  print_nobuf(
      "hostcall-second-pending-payload-probe: payload region ID = %lu\n",
      payload_region_id);

  shared_region_annotated(dom_id, metadata_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED);
  shared_region_annotated(dom_id, payload_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_OUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_DEFAULT);
  print_nobuf(
      "hostcall-second-pending-payload-probe: metadata shared, payload borrowed-out for round 1\n");

  ret1 = call_dom(dom_id);
  snapshot_request(&request, metadata);
  print_nobuf(
      "hostcall-second-pending-payload-probe: first call retval = %lu\n", ret1);
  print_nobuf(
      "hostcall-second-pending-payload-probe: snapped stage1 request{phase=%llu opcode=%llu offset=%llu length=%llu}\n",
      request.phase, request.opcode, request.offset, request.length);
  if (ret1 != HC_V0_RET_PENDING)
    return fail_cleanup("unexpected first call retval", ret1, &request);
  if (request.phase != HC_V0_PHASE_REQ)
    return fail_cleanup("unexpected stage1 phase", (unsigned long)request.phase,
                        &request);
  if (request.opcode != HC_V0_OP_SECOND_PENDING_PAYLOAD_STAGE1)
    return fail_cleanup("unexpected stage1 opcode", (unsigned long)request.opcode,
                        &request);
  if (request.offset != 0 ||
      request.length != HOSTCALL_SECOND_PENDING_PAYLOAD_STAGE1_LEN)
    return fail_cleanup("unexpected stage1 offset/length",
                        (unsigned long)request.length, &request);
  if (!snapshot_payload(payload_snapshot, payload, &request))
    return fail_cleanup("stage1 payload bounds error", 0, &request);
  if (!payload_matches(payload_snapshot, HOSTCALL_SECOND_PENDING_PAYLOAD_STAGE1_TEXT,
                       request.length)) {
    return fail_cleanup("unexpected stage1 payload contents", 0, &request);
  }

  metadata->result = HOSTCALL_SECOND_PENDING_PAYLOAD_STAGE1_RESULT;
  metadata->error = 0;
  metadata->phase = HC_V0_PHASE_RESP;

#ifdef HOSTCALL_SECOND_PENDING_PAYLOAD_DO_REVOKE
  print_nobuf(
      "hostcall-second-pending-payload-probe: revoking payload region before round 2 re-share\n");
  revoke_region(payload_region_id);
#endif
  print_nobuf(
      "hostcall-second-pending-payload-probe: about to re-share payload for round 2\n");
  shared_region_annotated(dom_id, payload_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_OUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_DEFAULT);
  print_nobuf(
      "hostcall-second-pending-payload-probe: payload re-shared as borrowed-out for round 2\n");

  ret2 = call_dom(dom_id);
  snapshot_request(&request, metadata);
  print_nobuf(
      "hostcall-second-pending-payload-probe: second call retval = %lu\n", ret2);
  print_nobuf(
      "hostcall-second-pending-payload-probe: snapped stage2 request{phase=%llu opcode=%llu offset=%llu length=%llu}\n",
      request.phase, request.opcode, request.offset, request.length);
  if (ret2 != HC_V0_RET_PENDING)
    return fail_cleanup("unexpected second call retval", ret2, &request);
  if (request.phase != HC_V0_PHASE_REQ)
    return fail_cleanup("unexpected stage2 phase", (unsigned long)request.phase,
                        &request);
  if (request.opcode != HC_V0_OP_SECOND_PENDING_PAYLOAD_STAGE2)
    return fail_cleanup("unexpected stage2 opcode", (unsigned long)request.opcode,
                        &request);
  if (request.offset != 0 ||
      request.length != HOSTCALL_SECOND_PENDING_PAYLOAD_STAGE2_LEN)
    return fail_cleanup("unexpected stage2 offset/length",
                        (unsigned long)request.length, &request);
  if (!snapshot_payload(payload_snapshot, payload, &request))
    return fail_cleanup("stage2 payload bounds error", 0, &request);
  if (!payload_matches(payload_snapshot, HOSTCALL_SECOND_PENDING_PAYLOAD_STAGE2_TEXT,
                       request.length)) {
    return fail_cleanup("unexpected stage2 payload contents", 0, &request);
  }

  print_nobuf(
      "hostcall-second-pending-payload-probe: second pending with payload reuse observed\n");
  metadata->result = HOSTCALL_SECOND_PENDING_PAYLOAD_STAGE2_RESULT;
  metadata->error = 0;
  metadata->phase = HC_V0_PHASE_RESP;

  ret3 = call_dom(dom_id);
  print_nobuf(
      "hostcall-second-pending-payload-probe: third call retval = %lu\n", ret3);
  print_nobuf(
      "hostcall-second-pending-payload-probe: final metadata phase = %llu\n",
      metadata->phase);
  if (ret3 != HC_V0_RET_DONE)
    return fail_cleanup("unexpected third call retval", ret3, metadata);
  if (metadata->phase != HC_V0_PHASE_DONE)
    return fail_cleanup("unexpected final phase", (unsigned long)metadata->phase,
                        metadata);
  if (metadata->result != HOSTCALL_SECOND_PENDING_PAYLOAD_STAGE2_RESULT)
    return fail_cleanup("unexpected final result", (unsigned long)metadata->result,
                        metadata);
  if (metadata->error != 0)
    return fail_cleanup("unexpected final error", (unsigned long)metadata->error,
                        metadata);

  print_nobuf("hostcall-second-pending-payload-probe: success\n");
  print_nobuf("__HOSTCALL_SECOND_PENDING_PAYLOAD_OK__\n");

  if (capstone_cleanup()) {
    fprintf(stderr,
            "hostcall-second-pending-payload-probe: failed to clean up Capstone\n");
    return 1;
  }

  return 0;
}



