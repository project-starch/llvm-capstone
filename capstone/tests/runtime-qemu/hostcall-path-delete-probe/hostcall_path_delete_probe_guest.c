#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "../hostcall-file-service-probe-common.h"

#define print_nobuf(...)         \
  do {                           \
    printf(__VA_ARGS__);         \
    fflush(stdout);              \
  } while (0)

static int fail_cleanup(const char *message, unsigned long observed,
                        struct hostcall_v0 *metadata) {
  fprintf(stderr, "hostcall-path-delete-probe: %s (observed=%lu)\n", message,
          observed);
  if (metadata) {
    fprintf(stderr,
            "hostcall-path-delete-probe: metadata{phase=%llu opcode=%llu offset=%llu length=%llu result=%lld error=%lld}\n",
            metadata->phase, metadata->opcode, metadata->offset,
            metadata->length, metadata->result, metadata->error);
  }
  capstone_cleanup();
  return 1;
}

static int service_path_delete(struct hostcall_v0 *metadata,
                               const struct hc_path_delete_req_v0 *request,
                               const char *path_snapshot) {
  if (request->flags != HC_PATH_DELETE_FLAG_NONE)
    return fail_cleanup("unexpected path delete flags",
                        (unsigned long)request->flags, metadata);

  print_nobuf(
      "hostcall-path-delete-probe: servicing HC_V0_OP_PATH_DELETE for %s\n",
      path_snapshot);

  if (unlink(path_snapshot) < 0) {
    metadata->offset = 0;
    metadata->length = 0;
    metadata->result = -1;
    metadata->error = errno;
    metadata->phase = HC_V0_PHASE_ERROR;
    return fail_cleanup("path delete request failed", (unsigned long)errno,
                        metadata);
  }

  metadata->offset = 0;
  metadata->length = 0;
  metadata->result = 0;
  metadata->error = 0;
  metadata->phase = HC_V0_PHASE_RESP;
  return 0;
}

static int service_path_access(struct hostcall_v0 *metadata,
                               const struct hc_path_access_req_v0 *request,
                               const char *path_snapshot) {
  int exists_result;

  if (request->flags != HC_PATH_ACCESS_FLAG_EXISTS)
    return fail_cleanup("unexpected path access flags",
                        (unsigned long)request->flags, metadata);

  print_nobuf(
      "hostcall-path-delete-probe: servicing HC_V0_OP_PATH_ACCESS for %s\n",
      path_snapshot);

  if (access(path_snapshot, F_OK) == 0) {
    exists_result = 1;
  } else if (errno == ENOENT || errno == ENOTDIR) {
    exists_result = 0;
  } else {
    metadata->offset = 0;
    metadata->length = 0;
    metadata->result = -1;
    metadata->error = errno;
    metadata->phase = HC_V0_PHASE_ERROR;
    return fail_cleanup("path access request failed", (unsigned long)errno,
                        metadata);
  }

  metadata->offset = 0;
  metadata->length = 0;
  metadata->result = (hostcall_s64_t)exists_result;
  metadata->error = 0;
  metadata->phase = HC_V0_PHASE_RESP;
  return 0;
}

int main(int argc, char **argv) {
  dom_id_t dom_id;
  region_id_t metadata_region_id;
  region_id_t payload_region_id;
  struct hostcall_v0 *metadata;
  char *payload;
  struct hostcall_v0 request;
  char path_snapshot[HOSTCALL_STDOUT_PROBE_REGION_SIZE + 1];
  struct hc_path_delete_req_v0 delete_request_snapshot;
  struct hc_path_access_req_v0 access_request_snapshot;
  unsigned long ret1;
  unsigned long ret2;
  unsigned long ret3;

  if (argc != 2) {
    fprintf(stderr, "usage: %s <smode-domain-path>\n", argv[0]);
    return 2;
  }

  if (capstone_init()) {
    fprintf(stderr,
            "hostcall-path-delete-probe: failed to initialize Capstone\n");
    return 1;
  }

  dom_id = create_dom("/test-domains/sbi.dom", argv[1]);
  if ((long)dom_id < 0)
    return fail_cleanup("create_dom failed", (unsigned long)dom_id, NULL);

  metadata_region_id = create_region(HOSTCALL_STDOUT_PROBE_REGION_SIZE);
  payload_region_id = create_region(HOSTCALL_STDOUT_PROBE_REGION_SIZE);
  metadata = (struct hostcall_v0 *)map_region(metadata_region_id,
                                              HOSTCALL_STDOUT_PROBE_REGION_SIZE);
  payload =
      (char *)map_region(payload_region_id, HOSTCALL_STDOUT_PROBE_REGION_SIZE);
  if (!metadata || !payload)
    return fail_cleanup("map_region failed", 0, metadata);

  memset(metadata, 0, HOSTCALL_STDOUT_PROBE_REGION_SIZE);
  memset(payload, 0, HOSTCALL_STDOUT_PROBE_REGION_SIZE);

  print_nobuf("hostcall-path-delete-probe: created domain ID = %lu\n", dom_id);
  print_nobuf("hostcall-path-delete-probe: metadata region ID = %lu\n",
              metadata_region_id);
  print_nobuf("hostcall-path-delete-probe: payload region ID = %lu\n",
              payload_region_id);

  shared_region_annotated(dom_id, metadata_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED);
  shared_region_annotated(dom_id, payload_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_OUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf(
      "hostcall-path-delete-probe: metadata shared, payload borrowed-out for path delete request\n");

  ret1 = call_dom(dom_id);
  hostcall_snapshot_request(&request, metadata);
  print_nobuf("hostcall-path-delete-probe: first call retval = %lu\n", ret1);
  if (ret1 != HC_V0_RET_PENDING)
    return fail_cleanup("unexpected first call retval", ret1, &request);
  if (request.phase != HC_V0_PHASE_REQ)
    return fail_cleanup("unexpected first request phase",
                        (unsigned long)request.phase, &request);
  if (request.opcode != HC_V0_OP_PATH_DELETE)
    return fail_cleanup("unexpected first request opcode",
                        (unsigned long)request.opcode, &request);
  if (request.offset != HC_PATH_DELETE_REQ_V0_PATH_OFFSET)
    return fail_cleanup("unexpected first path offset",
                        (unsigned long)request.offset, &request);
  if (!request.length)
    return fail_cleanup("unexpected empty delete path",
                        (unsigned long)request.length, &request);
  if (!hostcall_payload_range_valid(&request))
    return fail_cleanup("delete path exceeds shared region", 0, &request);

  memcpy(&delete_request_snapshot, payload, sizeof(delete_request_snapshot));
  memcpy(path_snapshot, payload + request.offset, (size_t)request.length);
  path_snapshot[request.length] = '\0';
  if (service_path_delete(metadata, &delete_request_snapshot, path_snapshot) != 0)
    return 1;

  revoke_region(payload_region_id);
  shared_region_annotated(dom_id, payload_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_OUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf(
      "hostcall-path-delete-probe: payload revoked and re-shared for path access request\n");

  ret2 = call_dom(dom_id);
  hostcall_snapshot_request(&request, metadata);
  print_nobuf("hostcall-path-delete-probe: second call retval = %lu\n", ret2);
  if (ret2 != HC_V0_RET_PENDING)
    return fail_cleanup("unexpected second call retval", ret2, &request);
  if (request.phase != HC_V0_PHASE_REQ)
    return fail_cleanup("unexpected second request phase",
                        (unsigned long)request.phase, &request);
  if (request.opcode != HC_V0_OP_PATH_ACCESS)
    return fail_cleanup("unexpected second request opcode",
                        (unsigned long)request.opcode, &request);
  if (request.offset != HC_PATH_ACCESS_REQ_V0_PATH_OFFSET)
    return fail_cleanup("unexpected second path offset",
                        (unsigned long)request.offset, &request);
  if (!request.length)
    return fail_cleanup("unexpected empty access path",
                        (unsigned long)request.length, &request);
  if (!hostcall_payload_range_valid(&request))
    return fail_cleanup("access path exceeds shared region", 0, &request);

  memcpy(&access_request_snapshot, payload, sizeof(access_request_snapshot));
  memcpy(path_snapshot, payload + request.offset, (size_t)request.length);
  path_snapshot[request.length] = '\0';
  if (service_path_access(metadata, &access_request_snapshot, path_snapshot) != 0)
    return 1;

  ret3 = call_dom(dom_id);
  print_nobuf("hostcall-path-delete-probe: third call retval = %lu\n", ret3);
  if (ret3 != HC_V0_RET_DONE)
    return fail_cleanup("unexpected third call retval", ret3, metadata);
  if (metadata->phase != HC_V0_PHASE_DONE)
    return fail_cleanup("unexpected final phase",
                        (unsigned long)metadata->phase, metadata);
  if (metadata->result != 0)
    return fail_cleanup("unexpected final result",
                        (unsigned long)metadata->result, metadata);
  if (metadata->error != 0)
    return fail_cleanup("unexpected final error",
                        (unsigned long)metadata->error, metadata);

  print_nobuf("hostcall-path-delete-probe: success\n");
  print_nobuf("__HOSTCALL_PATH_DELETE_OK__\n");

  if (capstone_cleanup()) {
    fprintf(stderr,
            "hostcall-path-delete-probe: failed to clean up Capstone\n");
    return 1;
  }

  return 0;
}

