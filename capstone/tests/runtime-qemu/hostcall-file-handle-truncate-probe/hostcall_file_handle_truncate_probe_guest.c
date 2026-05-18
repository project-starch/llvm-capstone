#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "../hostcall-file-service-probe-common.h"

#define print_nobuf(...)         \
  do {                           \
    printf(__VA_ARGS__);         \
    fflush(stdout);              \
  } while (0)

static struct hostcall_file_service_handle_slot
    handle_slots[HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES];

static int fail_cleanup(const char *message, unsigned long observed,
                        struct hostcall_v0 *metadata) {
  fprintf(stderr,
          "hostcall-file-handle-truncate-probe: %s (observed=%lu)\n",
          message, observed);
  if (metadata) {
    fprintf(stderr,
            "hostcall-file-handle-truncate-probe: metadata{phase=%llu opcode=%llu offset=%llu length=%llu result=%lld error=%lld}\n",
            metadata->phase, metadata->opcode, metadata->offset,
            metadata->length, metadata->result, metadata->error);
  }
  hostcall_cleanup_open_handles(handle_slots,
                                HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES);
  capstone_cleanup();
  return 1;
}

int main(int argc, char **argv) {
  dom_id_t dom_id;
  region_id_t metadata_region_id;
  region_id_t payload_region_id;
  struct hostcall_v0 *metadata;
  char *payload;
  struct hostcall_v0 request;
  char path_snapshot[HOSTCALL_STDOUT_PROBE_REGION_SIZE + 1];
  struct hc_file_truncate_req_v0 truncate_request_snapshot;
  struct hc_file_stat_basic_req_v0 stat_request_snapshot;
  struct stat helper_stat;
  unsigned long ret1;
  unsigned long ret2;
  unsigned long ret3;
  unsigned long ret4;
  unsigned long ret5;
  hostcall_u64_t handle_token;
  int fd;

  if (argc != 2) {
    fprintf(stderr, "usage: %s <smode-domain-path>\n", argv[0]);
    return 2;
  }

  if (capstone_init()) {
    fprintf(stderr,
            "hostcall-file-handle-truncate-probe: failed to initialize Capstone\n");
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
  memset(handle_slots, 0, sizeof(handle_slots));

  print_nobuf("hostcall-file-handle-truncate-probe: created domain ID = %lu\n",
              dom_id);
  print_nobuf("hostcall-file-handle-truncate-probe: metadata region ID = %lu\n",
              metadata_region_id);
  print_nobuf("hostcall-file-handle-truncate-probe: payload region ID = %lu\n",
              payload_region_id);

  shared_region_annotated(dom_id, metadata_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED);
  shared_region_annotated(dom_id, payload_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_OUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf(
      "hostcall-file-handle-truncate-probe: metadata shared, payload borrowed-out for open request\n");

  ret1 = call_dom(dom_id);
  hostcall_snapshot_request(&request, metadata);
  print_nobuf("hostcall-file-handle-truncate-probe: first call retval = %lu\n",
              ret1);
  if (ret1 != HC_V0_RET_PENDING)
    return fail_cleanup("unexpected first call retval", ret1, &request);
  if (request.phase != HC_V0_PHASE_REQ)
    return fail_cleanup("unexpected open phase", (unsigned long)request.phase,
                        &request);
  if (request.opcode != HC_V0_OP_FILE_OPEN)
    return fail_cleanup("unexpected open opcode", (unsigned long)request.opcode,
                        &request);
  if (request.offset != HC_FILE_OPEN_REQ_V0_PATH_OFFSET)
    return fail_cleanup("unexpected open path offset",
                        (unsigned long)request.offset, &request);
  if (!request.length)
    return fail_cleanup("unexpected empty open path",
                        (unsigned long)request.length, &request);
  if (!hostcall_payload_range_valid(&request))
    return fail_cleanup("open path exceeds shared region", 0, &request);

  memcpy(path_snapshot, payload + request.offset, (size_t)request.length);
  path_snapshot[request.length] = '\0';
  print_nobuf(
      "hostcall-file-handle-truncate-probe: servicing HC_V0_OP_FILE_OPEN for %s\n",
      path_snapshot);

  fd = open(path_snapshot, (int)((struct hc_file_open_req_v0 *)payload)->flags,
            (mode_t)((struct hc_file_open_req_v0 *)payload)->mode);
  if (fd < 0) {
    metadata->result = -1;
    metadata->error = errno;
    metadata->phase = HC_V0_PHASE_ERROR;
    return fail_cleanup("open request failed", (unsigned long)errno, metadata);
  }

  handle_token = hostcall_allocate_handle_token(
      handle_slots, HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES, fd);
  if (!handle_token) {
    int saved_errno = errno;
    close(fd);
    metadata->result = -1;
    metadata->error = saved_errno;
    metadata->phase = HC_V0_PHASE_ERROR;
    return fail_cleanup("handle allocation failed", (unsigned long)saved_errno,
                        metadata);
  }

  metadata->result = (hostcall_s64_t)handle_token;
  metadata->error = 0;
  metadata->phase = HC_V0_PHASE_RESP;
  print_nobuf(
      "hostcall-file-handle-truncate-probe: opened helper handle token = %llu\n",
      handle_token);

  revoke_region(payload_region_id);
  shared_region_annotated(dom_id, payload_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_OUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf(
      "hostcall-file-handle-truncate-probe: payload revoked and re-shared for truncate request\n");

  ret2 = call_dom(dom_id);
  hostcall_snapshot_request(&request, metadata);
  print_nobuf("hostcall-file-handle-truncate-probe: second call retval = %lu\n",
              ret2);
  if (ret2 != HC_V0_RET_PENDING)
    return fail_cleanup("unexpected second call retval", ret2, &request);
  if (request.phase != HC_V0_PHASE_REQ)
    return fail_cleanup("unexpected truncate phase", (unsigned long)request.phase,
                        &request);
  if (request.opcode != HC_V0_OP_FILE_TRUNCATE)
    return fail_cleanup("unexpected truncate opcode",
                        (unsigned long)request.opcode, &request);
  if (request.offset != 0 || request.length != 0)
    return fail_cleanup("unexpected truncate offset/length",
                        (unsigned long)request.offset, &request);
  memcpy(&truncate_request_snapshot, payload, sizeof(truncate_request_snapshot));
  if (truncate_request_snapshot.handle != handle_token)
    return fail_cleanup("unexpected truncate handle token",
                        (unsigned long)truncate_request_snapshot.handle,
                        &request);
  if (truncate_request_snapshot.flags != 0)
    return fail_cleanup("unexpected truncate flags",
                        (unsigned long)truncate_request_snapshot.flags,
                        &request);

  print_nobuf(
      "hostcall-file-handle-truncate-probe: servicing HC_V0_OP_FILE_TRUNCATE for token %llu\n",
      handle_token);
  fd = hostcall_lookup_handle_fd(handle_slots,
                                 HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES,
                                 handle_token);
  if (fd < 0) {
    metadata->result = -1;
    metadata->error = errno;
    metadata->phase = HC_V0_PHASE_ERROR;
    return fail_cleanup("lookup truncate handle failed", (unsigned long)errno,
                        metadata);
  }
  if (ftruncate(fd, (off_t)truncate_request_snapshot.size) < 0) {
    metadata->result = -1;
    metadata->error = errno;
    metadata->phase = HC_V0_PHASE_ERROR;
    return fail_cleanup("truncate request failed", (unsigned long)errno,
                        metadata);
  }

  metadata->offset = 0;
  metadata->length = 0;
  metadata->result = 0;
  metadata->error = 0;
  metadata->phase = HC_V0_PHASE_RESP;

  revoke_region(payload_region_id);
  shared_region_annotated(dom_id, payload_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_OUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf(
      "hostcall-file-handle-truncate-probe: payload revoked and re-shared for stat request\n");

  ret3 = call_dom(dom_id);
  hostcall_snapshot_request(&request, metadata);
  print_nobuf("hostcall-file-handle-truncate-probe: third call retval = %lu\n",
              ret3);
  if (ret3 != HC_V0_RET_PENDING)
    return fail_cleanup("unexpected third call retval", ret3, &request);
  if (request.phase != HC_V0_PHASE_REQ)
    return fail_cleanup("unexpected stat phase", (unsigned long)request.phase,
                        &request);
  if (request.opcode != HC_V0_OP_FILE_STAT_BASIC)
    return fail_cleanup("unexpected stat opcode", (unsigned long)request.opcode,
                        &request);
  if (request.offset != 0 || request.length != 0)
    return fail_cleanup("unexpected stat offset/length",
                        (unsigned long)request.offset, &request);
  memcpy(&stat_request_snapshot, payload, sizeof(stat_request_snapshot));
  if (stat_request_snapshot.handle != handle_token)
    return fail_cleanup("unexpected stat handle token",
                        (unsigned long)stat_request_snapshot.handle, &request);
  if (stat_request_snapshot.flags != 0)
    return fail_cleanup("unexpected stat flags",
                        (unsigned long)stat_request_snapshot.flags, &request);

  print_nobuf(
      "hostcall-file-handle-truncate-probe: servicing HC_V0_OP_FILE_STAT_BASIC for token %llu\n",
      handle_token);
  if (fstat(fd, &helper_stat) < 0) {
    metadata->result = -1;
    metadata->error = errno;
    metadata->phase = HC_V0_PHASE_ERROR;
    return fail_cleanup("stat request failed", (unsigned long)errno, metadata);
  }

  revoke_region(payload_region_id);
  shared_region_annotated(dom_id, payload_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf(
      "hostcall-file-handle-truncate-probe: payload revoked and re-shared for stat response plus close request\n");

  ((struct hc_file_stat_basic_resp_v0 *)payload)->file_size =
      (hostcall_u64_t)helper_stat.st_size;
  ((struct hc_file_stat_basic_resp_v0 *)payload)->mode =
      (hostcall_u64_t)helper_stat.st_mode;
  ((struct hc_file_stat_basic_resp_v0 *)payload)->reserved0 = 0;
  ((struct hc_file_stat_basic_resp_v0 *)payload)->reserved1 = 0;
  metadata->offset = 0;
  metadata->length = HC_FILE_STAT_BASIC_RESP_V0_SIZE;
  metadata->result = 0;
  metadata->error = 0;
  metadata->phase = HC_V0_PHASE_RESP;

  ret4 = call_dom(dom_id);
  hostcall_snapshot_request(&request, metadata);
  print_nobuf("hostcall-file-handle-truncate-probe: fourth call retval = %lu\n",
              ret4);
  if (ret4 != HC_V0_RET_PENDING)
    return fail_cleanup("unexpected fourth call retval", ret4, &request);
  if (request.phase != HC_V0_PHASE_REQ)
    return fail_cleanup("unexpected close phase", (unsigned long)request.phase,
                        &request);
  if (request.opcode != HC_V0_OP_FILE_CLOSE)
    return fail_cleanup("unexpected close opcode", (unsigned long)request.opcode,
                        &request);
  if (((struct hc_file_close_req_v0 *)payload)->handle != handle_token)
    return fail_cleanup("unexpected close handle token",
                        (unsigned long)((struct hc_file_close_req_v0 *)payload)
                            ->handle,
                        &request);

  print_nobuf(
      "hostcall-file-handle-truncate-probe: servicing HC_V0_OP_FILE_CLOSE for token %llu\n",
      handle_token);
  if (hostcall_close_handle_token(handle_slots,
                                  HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES,
                                  handle_token) < 0) {
    metadata->result = -1;
    metadata->error = errno;
    metadata->phase = HC_V0_PHASE_ERROR;
    return fail_cleanup("close request failed", (unsigned long)errno, metadata);
  }

  metadata->offset = 0;
  metadata->length = 0;
  metadata->result = 0;
  metadata->error = 0;
  metadata->phase = HC_V0_PHASE_RESP;

  ret5 = call_dom(dom_id);
  print_nobuf("hostcall-file-handle-truncate-probe: fifth call retval = %lu\n",
              ret5);
  if (ret5 != HC_V0_RET_DONE)
    return fail_cleanup("unexpected fifth call retval", ret5, metadata);
  if (metadata->phase != HC_V0_PHASE_DONE)
    return fail_cleanup("unexpected final phase",
                        (unsigned long)metadata->phase, metadata);
  if (metadata->result != 0)
    return fail_cleanup("unexpected final result",
                        (unsigned long)metadata->result, metadata);
  if (metadata->error != 0)
    return fail_cleanup("unexpected final error",
                        (unsigned long)metadata->error, metadata);

  print_nobuf("hostcall-file-handle-truncate-probe: success\n");
  print_nobuf("__HOSTCALL_FILE_HANDLE_TRUNCATE_OK__\n");

  if (capstone_cleanup()) {
    fprintf(stderr,
            "hostcall-file-handle-truncate-probe: failed to clean up Capstone\n");
    return 1;
  }

  return 0;
}

