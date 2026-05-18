#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "../hostcall-stdout-probe/hostcall_stdout_probe.h"

#define print_nobuf(...)         \
  do {                           \
    printf(__VA_ARGS__);         \
    fflush(stdout);              \
  } while (0)

#define HOSTCALL_COMBINED_FILE_OBJECT_MAX_HANDLES 8

struct handle_slot {
  int in_use;
  int fd;
};

static struct handle_slot handle_slots[HOSTCALL_COMBINED_FILE_OBJECT_MAX_HANDLES];

static void cleanup_open_handles(void) {
  hostcall_u64_t slot_i;

  for (slot_i = 0; slot_i < HOSTCALL_COMBINED_FILE_OBJECT_MAX_HANDLES; ++slot_i) {
    if (handle_slots[slot_i].in_use) {
      close(handle_slots[slot_i].fd);
      handle_slots[slot_i].fd = -1;
      handle_slots[slot_i].in_use = 0;
    }
  }
}

static int fail_cleanup(const char *message, unsigned long observed,
                        struct hostcall_v0 *metadata) {
  fprintf(stderr,
          "hostcall-combined-file-object-probe: %s (observed=%lu)\n",
          message, observed);
  if (metadata) {
    fprintf(stderr,
            "hostcall-combined-file-object-probe: metadata{phase=%llu opcode=%llu offset=%llu length=%llu result=%lld error=%lld}\n",
            metadata->phase, metadata->opcode, metadata->offset,
            metadata->length, metadata->result, metadata->error);
  }
  cleanup_open_handles();
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

static int payload_range_valid(const struct hostcall_v0 *request) {
  hostcall_u64_t end = request->offset + request->length;

  if (request->offset > HOSTCALL_STDOUT_PROBE_REGION_SIZE)
    return 0;
  if (request->length > HOSTCALL_STDOUT_PROBE_REGION_SIZE)
    return 0;
  if (end > HOSTCALL_STDOUT_PROBE_REGION_SIZE)
    return 0;
  return 1;
}

static hostcall_u64_t allocate_handle_token(int fd) {
  hostcall_u64_t slot_i;

  for (slot_i = 0; slot_i < HOSTCALL_COMBINED_FILE_OBJECT_MAX_HANDLES; ++slot_i) {
    if (!handle_slots[slot_i].in_use) {
      handle_slots[slot_i].in_use = 1;
      handle_slots[slot_i].fd = fd;
      return slot_i + 1;
    }
  }

  errno = EMFILE;
  return 0;
}

static int lookup_handle_fd(hostcall_u64_t token) {
  if (token == 0 || token > HOSTCALL_COMBINED_FILE_OBJECT_MAX_HANDLES) {
    errno = EBADF;
    return -1;
  }
  if (!handle_slots[token - 1].in_use) {
    errno = EBADF;
    return -1;
  }
  return handle_slots[token - 1].fd;
}

static int close_handle_token(hostcall_u64_t token) {
  struct handle_slot *slot;
  int fd;

  if (token == 0 || token > HOSTCALL_COMBINED_FILE_OBJECT_MAX_HANDLES) {
    errno = EBADF;
    return -1;
  }

  slot = &handle_slots[token - 1];
  if (!slot->in_use) {
    errno = EBADF;
    return -1;
  }

  fd = slot->fd;
  slot->fd = -1;
  slot->in_use = 0;
  if (close(fd) < 0)
    return -1;
  return 0;
}

static ssize_t write_full_at_offset(int fd, const char *buffer, size_t len,
                                    off_t file_offset) {
  size_t written = 0;

  if (lseek(fd, file_offset, SEEK_SET) < 0)
    return -1;

  while (written < len) {
    ssize_t rc = write(fd, buffer + written, len - written);
    if (rc < 0)
      return -1;
    if (rc == 0) {
      errno = EIO;
      return -1;
    }
    written += (size_t)rc;
  }

  return (ssize_t)written;
}

static ssize_t read_into_buffer_at_offset(int fd, char *buffer, size_t len,
                                          off_t file_offset) {
  if (lseek(fd, file_offset, SEEK_SET) < 0)
    return -1;
  return read(fd, buffer, len);
}

int main(int argc, char **argv) {
  dom_id_t dom_id;
  region_id_t metadata_region_id;
  region_id_t payload_region_id;
  struct hostcall_v0 *metadata;
  char *payload;
  struct hostcall_v0 request;
  char path_snapshot[HOSTCALL_STDOUT_PROBE_REGION_SIZE + 1];
  char write_snapshot[HOSTCALL_STDOUT_PROBE_REGION_SIZE];
  struct hc_file_read_req_v0 read_request_snapshot;
  unsigned long ret1;
  unsigned long ret2;
  unsigned long ret3;
  unsigned long ret4;
  unsigned long ret5;
  unsigned long ret6;
  unsigned long ret7;
  hostcall_u64_t write_handle_token;
  hostcall_u64_t read_handle_token;
  int fd;
  ssize_t read_result;

  if (argc != 2) {
    fprintf(stderr, "usage: %s <smode-domain-path>\n", argv[0]);
    return 2;
  }

  if (capstone_init()) {
    fprintf(stderr,
            "hostcall-combined-file-object-probe: failed to initialize Capstone\n");
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

  print_nobuf("hostcall-combined-file-object-probe: created domain ID = %lu\n",
              dom_id);
  print_nobuf("hostcall-combined-file-object-probe: metadata region ID = %lu\n",
              metadata_region_id);
  print_nobuf("hostcall-combined-file-object-probe: payload region ID = %lu\n",
              payload_region_id);

  shared_region_annotated(dom_id, metadata_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED);
  shared_region_annotated(dom_id, payload_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_OUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf(
      "hostcall-combined-file-object-probe: metadata shared, payload borrowed-out for first open request\n");

  ret1 = call_dom(dom_id);
  snapshot_request(&request, metadata);
  print_nobuf("hostcall-combined-file-object-probe: first call retval = %lu\n",
              ret1);
  if (ret1 != HC_V0_RET_PENDING)
    return fail_cleanup("unexpected first call retval", ret1, &request);
  if (request.phase != HC_V0_PHASE_REQ || request.opcode != HC_V0_OP_FILE_OPEN)
    return fail_cleanup("unexpected first open request",
                        (unsigned long)request.opcode, &request);
  if (request.offset != HC_FILE_OPEN_REQ_V0_PATH_OFFSET || !request.length)
    return fail_cleanup("unexpected first open layout",
                        (unsigned long)request.offset, &request);
  if (!payload_range_valid(&request))
    return fail_cleanup("first open path exceeds payload region", 0, &request);

  memcpy(path_snapshot, payload + request.offset, (size_t)request.length);
  path_snapshot[request.length] = '\0';
  print_nobuf(
      "hostcall-combined-file-object-probe: servicing first HC_V0_OP_FILE_OPEN for %s\n",
      path_snapshot);
  fd = open(path_snapshot, (int)((struct hc_file_open_req_v0 *)payload)->flags,
            (mode_t)((struct hc_file_open_req_v0 *)payload)->mode);
  if (fd < 0) {
    metadata->result = -1;
    metadata->error = errno;
    metadata->phase = HC_V0_PHASE_ERROR;
    return fail_cleanup("first open request failed", (unsigned long)errno,
                        metadata);
  }
  write_handle_token = allocate_handle_token(fd);
  if (!write_handle_token) {
    int saved_errno = errno;
    close(fd);
    metadata->result = -1;
    metadata->error = saved_errno;
    metadata->phase = HC_V0_PHASE_ERROR;
    return fail_cleanup("first handle allocation failed",
                        (unsigned long)saved_errno, metadata);
  }
  metadata->result = (hostcall_s64_t)write_handle_token;
  metadata->error = 0;
  metadata->phase = HC_V0_PHASE_RESP;

  revoke_region(payload_region_id);
  shared_region_annotated(dom_id, payload_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_OUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf(
      "hostcall-combined-file-object-probe: payload revoked and re-shared for write request\n");

  ret2 = call_dom(dom_id);
  snapshot_request(&request, metadata);
  print_nobuf("hostcall-combined-file-object-probe: second call retval = %lu\n",
              ret2);
  if (ret2 != HC_V0_RET_PENDING)
    return fail_cleanup("unexpected second call retval", ret2, &request);
  if (request.phase != HC_V0_PHASE_REQ || request.opcode != HC_V0_OP_FILE_WRITE)
    return fail_cleanup("unexpected write request",
                        (unsigned long)request.opcode, &request);
  if (request.offset != HC_FILE_WRITE_REQ_V0_DATA_OFFSET)
    return fail_cleanup("unexpected write data offset",
                        (unsigned long)request.offset, &request);
  if (!payload_range_valid(&request))
    return fail_cleanup("write payload exceeds payload region", 0, &request);
  if (((struct hc_file_write_req_v0 *)payload)->handle != write_handle_token)
    return fail_cleanup("unexpected write handle token",
                        (unsigned long)((struct hc_file_write_req_v0 *)payload)
                            ->handle,
                        &request);

  memcpy(write_snapshot, payload + request.offset, (size_t)request.length);
  print_nobuf(
      "hostcall-combined-file-object-probe: servicing HC_V0_OP_FILE_WRITE for token %llu\n",
      write_handle_token);
  fd = lookup_handle_fd(write_handle_token);
  if (fd < 0) {
    metadata->result = -1;
    metadata->error = errno;
    metadata->phase = HC_V0_PHASE_ERROR;
    return fail_cleanup("lookup write handle failed", (unsigned long)errno,
                        metadata);
  }
  if (write_full_at_offset(fd, write_snapshot, (size_t)request.length,
                           (off_t)((struct hc_file_write_req_v0 *)payload)
                               ->file_offset) < 0) {
    metadata->result = -1;
    metadata->error = errno;
    metadata->phase = HC_V0_PHASE_ERROR;
    return fail_cleanup("write request failed", (unsigned long)errno, metadata);
  }
  metadata->result = (hostcall_s64_t)request.length;
  metadata->error = 0;
  metadata->phase = HC_V0_PHASE_RESP;

  revoke_region(payload_region_id);
  shared_region_annotated(dom_id, payload_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_OUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf(
      "hostcall-combined-file-object-probe: payload revoked and re-shared for first close request\n");

  ret3 = call_dom(dom_id);
  snapshot_request(&request, metadata);
  print_nobuf("hostcall-combined-file-object-probe: third call retval = %lu\n",
              ret3);
  if (ret3 != HC_V0_RET_PENDING)
    return fail_cleanup("unexpected third call retval", ret3, &request);
  if (request.phase != HC_V0_PHASE_REQ || request.opcode != HC_V0_OP_FILE_CLOSE)
    return fail_cleanup("unexpected first close request",
                        (unsigned long)request.opcode, &request);
  if (((struct hc_file_close_req_v0 *)payload)->handle != write_handle_token)
    return fail_cleanup("unexpected first close token",
                        (unsigned long)((struct hc_file_close_req_v0 *)payload)
                            ->handle,
                        &request);
  print_nobuf(
      "hostcall-combined-file-object-probe: servicing first HC_V0_OP_FILE_CLOSE for token %llu\n",
      write_handle_token);
  if (close_handle_token(write_handle_token) < 0) {
    metadata->result = -1;
    metadata->error = errno;
    metadata->phase = HC_V0_PHASE_ERROR;
    return fail_cleanup("first close request failed", (unsigned long)errno,
                        metadata);
  }
  metadata->result = 0;
  metadata->error = 0;
  metadata->phase = HC_V0_PHASE_RESP;

  revoke_region(payload_region_id);
  shared_region_annotated(dom_id, payload_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_OUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf(
      "hostcall-combined-file-object-probe: payload revoked and re-shared for second open request\n");

  ret4 = call_dom(dom_id);
  snapshot_request(&request, metadata);
  print_nobuf("hostcall-combined-file-object-probe: fourth call retval = %lu\n",
              ret4);
  if (ret4 != HC_V0_RET_PENDING)
    return fail_cleanup("unexpected fourth call retval", ret4, &request);
  if (request.phase != HC_V0_PHASE_REQ || request.opcode != HC_V0_OP_FILE_OPEN)
    return fail_cleanup("unexpected second open request",
                        (unsigned long)request.opcode, &request);
  if (request.offset != HC_FILE_OPEN_REQ_V0_PATH_OFFSET || !request.length)
    return fail_cleanup("unexpected second open layout",
                        (unsigned long)request.offset, &request);
  if (!payload_range_valid(&request))
    return fail_cleanup("second open path exceeds payload region", 0, &request);

  memcpy(path_snapshot, payload + request.offset, (size_t)request.length);
  path_snapshot[request.length] = '\0';
  print_nobuf(
      "hostcall-combined-file-object-probe: servicing second HC_V0_OP_FILE_OPEN for %s\n",
      path_snapshot);
  fd = open(path_snapshot, (int)((struct hc_file_open_req_v0 *)payload)->flags,
            (mode_t)((struct hc_file_open_req_v0 *)payload)->mode);
  if (fd < 0) {
    metadata->result = -1;
    metadata->error = errno;
    metadata->phase = HC_V0_PHASE_ERROR;
    return fail_cleanup("second open request failed", (unsigned long)errno,
                        metadata);
  }
  read_handle_token = allocate_handle_token(fd);
  if (!read_handle_token) {
    int saved_errno = errno;
    close(fd);
    metadata->result = -1;
    metadata->error = saved_errno;
    metadata->phase = HC_V0_PHASE_ERROR;
    return fail_cleanup("second handle allocation failed",
                        (unsigned long)saved_errno, metadata);
  }
  metadata->result = (hostcall_s64_t)read_handle_token;
  metadata->error = 0;
  metadata->phase = HC_V0_PHASE_RESP;

  revoke_region(payload_region_id);
  shared_region_annotated(dom_id, payload_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_OUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf(
      "hostcall-combined-file-object-probe: payload revoked and re-shared for read request\n");

  ret5 = call_dom(dom_id);
  snapshot_request(&request, metadata);
  print_nobuf("hostcall-combined-file-object-probe: fifth call retval = %lu\n",
              ret5);
  if (ret5 != HC_V0_RET_PENDING)
    return fail_cleanup("unexpected fifth call retval", ret5, &request);
  if (request.phase != HC_V0_PHASE_REQ || request.opcode != HC_V0_OP_FILE_READ)
    return fail_cleanup("unexpected read request",
                        (unsigned long)request.opcode, &request);
  if (request.offset != HC_FILE_READ_REQ_V0_DATA_OFFSET)
    return fail_cleanup("unexpected read data offset",
                        (unsigned long)request.offset, &request);
  if (!payload_range_valid(&request))
    return fail_cleanup("read request exceeds payload region", 0, &request);
  if (((struct hc_file_read_req_v0 *)payload)->handle != read_handle_token)
    return fail_cleanup("unexpected read handle token",
                        (unsigned long)((struct hc_file_read_req_v0 *)payload)
                            ->handle,
                        &request);

  memcpy(&read_request_snapshot, payload, sizeof(read_request_snapshot));
  print_nobuf(
      "hostcall-combined-file-object-probe: servicing HC_V0_OP_FILE_READ for token %llu\n",
      read_handle_token);
  fd = lookup_handle_fd(read_handle_token);
  if (fd < 0) {
    metadata->result = -1;
    metadata->error = errno;
    metadata->phase = HC_V0_PHASE_ERROR;
    return fail_cleanup("lookup read handle failed", (unsigned long)errno,
                        metadata);
  }

  revoke_region(payload_region_id);
  shared_region_annotated(dom_id, payload_region_id,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf(
      "hostcall-combined-file-object-probe: payload revoked and re-shared for read response plus final close request\n");

  read_result = read_into_buffer_at_offset(
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

  ret6 = call_dom(dom_id);
  snapshot_request(&request, metadata);
  print_nobuf("hostcall-combined-file-object-probe: sixth call retval = %lu\n",
              ret6);
  if (ret6 != HC_V0_RET_PENDING)
    return fail_cleanup("unexpected sixth call retval", ret6, &request);
  if (request.phase != HC_V0_PHASE_REQ || request.opcode != HC_V0_OP_FILE_CLOSE)
    return fail_cleanup("unexpected second close request",
                        (unsigned long)request.opcode, &request);
  if (((struct hc_file_close_req_v0 *)payload)->handle != read_handle_token)
    return fail_cleanup("unexpected second close token",
                        (unsigned long)((struct hc_file_close_req_v0 *)payload)
                            ->handle,
                        &request);
  print_nobuf(
      "hostcall-combined-file-object-probe: servicing second HC_V0_OP_FILE_CLOSE for token %llu\n",
      read_handle_token);
  if (close_handle_token(read_handle_token) < 0) {
    metadata->result = -1;
    metadata->error = errno;
    metadata->phase = HC_V0_PHASE_ERROR;
    return fail_cleanup("second close request failed", (unsigned long)errno,
                        metadata);
  }
  metadata->result = 0;
  metadata->error = 0;
  metadata->phase = HC_V0_PHASE_RESP;

  ret7 = call_dom(dom_id);
  print_nobuf("hostcall-combined-file-object-probe: seventh call retval = %lu\n",
              ret7);
  if (ret7 != HC_V0_RET_DONE)
    return fail_cleanup("unexpected seventh call retval", ret7, metadata);
  if (metadata->phase != HC_V0_PHASE_DONE)
    return fail_cleanup("unexpected final phase",
                        (unsigned long)metadata->phase, metadata);
  if (metadata->result != 0)
    return fail_cleanup("unexpected final result",
                        (unsigned long)metadata->result, metadata);
  if (metadata->error != 0)
    return fail_cleanup("unexpected final error",
                        (unsigned long)metadata->error, metadata);

  print_nobuf("hostcall-combined-file-object-probe: success\n");
  print_nobuf("__HOSTCALL_COMBINED_FILE_OBJECT_OK__\n");

  cleanup_open_handles();
  if (capstone_cleanup()) {
    fprintf(stderr,
            "hostcall-combined-file-object-probe: failed to clean up Capstone\n");
    return 1;
  }

  return 0;
}

