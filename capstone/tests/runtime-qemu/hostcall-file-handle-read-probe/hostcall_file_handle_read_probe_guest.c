#include <errno.h>
#include <fcntl.h>
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

static struct hostcall_file_service_handle_slot
    handle_slots[HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES];

static int fail_cleanup(const char *message, unsigned long observed,
                        struct hostcall_v0 *metadata) {
  fprintf(stderr,
          "hostcall-file-handle-read-probe: %s (observed=%lu)\n",
          message, observed);
  if (metadata) {
    fprintf(stderr,
            "hostcall-file-handle-read-probe: metadata{phase=%llu opcode=%llu offset=%llu length=%llu result=%lld error=%lld}\n",
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
  struct hc_file_read_req_v0 read_request_snapshot;
  unsigned long ret1;
  unsigned long ret2;
  unsigned long ret3;
  hostcall_u64_t handle_token;
  int fd;
  ssize_t read_result;

  if (argc != 2) {
    fprintf(stderr, "usage: %s <smode-domain-path>\n", argv[0]);
    return 2;
  }

  if (capstone_init()) {
    fprintf(stderr,
            "hostcall-file-handle-read-probe: failed to initialize Capstone\n");
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

  print_nobuf("hostcall-file-handle-read-probe: created domain ID = %lu\n",
              dom_id);
  print_nobuf("hostcall-file-handle-read-probe: metadata region ID = %lu\n",
              metadata_region_id);
  print_nobuf("hostcall-file-handle-read-probe: payload region ID = %lu\n",
              payload_region_id);

  shared_region_annotated(dom_id, metadata_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED);
  shared_region_annotated(dom_id, payload_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_OUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf(
      "hostcall-file-handle-read-probe: metadata shared, payload borrowed-out for open request\n");

  ret1 = call_dom(dom_id);
  hostcall_snapshot_request(&request, metadata);
  print_nobuf("hostcall-file-handle-read-probe: first call retval = %lu\n",
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
      "hostcall-file-handle-read-probe: servicing HC_V0_OP_FILE_OPEN for %s\n",
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
      "hostcall-file-handle-read-probe: opened helper handle token = %llu\n",
      handle_token);

  revoke_region(payload_region_id);
  shared_region_annotated(dom_id, payload_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_OUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf(
      "hostcall-file-handle-read-probe: payload revoked and re-shared for read request\n");

  ret2 = call_dom(dom_id);
  hostcall_snapshot_request(&request, metadata);
  print_nobuf("hostcall-file-handle-read-probe: second call retval = %lu\n",
              ret2);
  if (ret2 != HC_V0_RET_PENDING)
    return fail_cleanup("unexpected second call retval", ret2, &request);
  if (request.phase != HC_V0_PHASE_REQ)
    return fail_cleanup("unexpected read phase", (unsigned long)request.phase,
                        &request);
  if (request.opcode != HC_V0_OP_FILE_READ)
    return fail_cleanup("unexpected read opcode", (unsigned long)request.opcode,
                        &request);
  if (request.offset != HC_FILE_READ_REQ_V0_DATA_OFFSET)
    return fail_cleanup("unexpected read data offset",
                        (unsigned long)request.offset, &request);
  if (!hostcall_payload_range_valid(&request))
    return fail_cleanup("read payload exceeds shared region", 0, &request);
  if (((struct hc_file_read_req_v0 *)payload)->handle != handle_token)
    return fail_cleanup("unexpected read handle token",
                        (unsigned long)((struct hc_file_read_req_v0 *)payload)
                            ->handle,
                        &request);

  memcpy(&read_request_snapshot, payload, sizeof(read_request_snapshot));
  print_nobuf(
      "hostcall-file-handle-read-probe: servicing HC_V0_OP_FILE_READ for token %llu\n",
      handle_token);
  fd = hostcall_lookup_handle_fd(handle_slots,
                                 HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES,
                                 handle_token);
  if (fd < 0) {
    metadata->result = -1;
    metadata->error = errno;
    metadata->phase = HC_V0_PHASE_ERROR;
    return fail_cleanup("lookup read handle failed", (unsigned long)errno,
                        metadata);
  }

  revoke_region(payload_region_id);
  shared_region_annotated(dom_id, payload_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_IN,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf(
      "hostcall-file-handle-read-probe: payload revoked and re-shared for read response\n");

  read_result = hostcall_read_into_buffer_at_offset(
      fd, payload + request.offset, (size_t)request.length,
      (off_t)read_request_snapshot.file_offset);
  if (read_result < 0) {
    metadata->result = -1;
    metadata->error = errno;
    metadata->phase = HC_V0_PHASE_ERROR;
    return fail_cleanup("read request failed", (unsigned long)errno, metadata);
  }

  metadata->result = (hostcall_s64_t)read_result;
  metadata->error = 0;
  metadata->phase = HC_V0_PHASE_RESP;

  ret3 = call_dom(dom_id);
  print_nobuf("hostcall-file-handle-read-probe: third call retval = %lu\n",
              ret3);
  if (ret3 != HC_V0_RET_DONE)
    return fail_cleanup("unexpected third call retval", ret3, metadata);
  if (metadata->phase != HC_V0_PHASE_DONE)
    return fail_cleanup("unexpected final phase",
                        (unsigned long)metadata->phase, metadata);
  if (metadata->result != HOSTCALL_FILE_HANDLE_READ_PROBE_MESSAGE_LEN)
    return fail_cleanup("unexpected final result",
                        (unsigned long)metadata->result, metadata);
  if (metadata->error != 0)
    return fail_cleanup("unexpected final error",
                        (unsigned long)metadata->error, metadata);

  print_nobuf("hostcall-file-handle-read-probe: success\n");
  print_nobuf("__HOSTCALL_FILE_HANDLE_READ_OK__\n");

  hostcall_cleanup_open_handles(handle_slots,
                                HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES);
  if (capstone_cleanup()) {
    fprintf(stderr,
            "hostcall-file-handle-read-probe: failed to clean up Capstone\n");
    return 1;
  }

  return 0;
}

