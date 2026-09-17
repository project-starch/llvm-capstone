/* Host side of the musl file probe: services the HostCall v0 file opcodes for a
 * pure-capability domain running musl.
 *
 * REUSES the handle table from hostcall-file-service-probe-common.h rather than
 * keeping its own, so the token semantics here are the same ones the existing
 * .smode file probes were validated against: tokens are slot index + 1, token 0
 * is reserved as invalid, and a token is never a raw descriptor.
 *
 * pread AND pwrite, NOT read AND write. The wire protocol carries an explicit
 * file_offset on every round, and the domain is what keeps the POSIX position.
 * Using the helper descriptor's own position instead would work for a strictly
 * sequential probe and diverge the moment anything seeks, which is exactly what
 * arm 1 does.
 *
 * ERROR SIGN. The wire spec (section 11) says metadata.error carries a NEGATIVE
 * errno. Some probes in this tree write a positive one. This host follows the
 * spec, and the domain forces the sign anyway, because a positive errno reaches
 * musl's __syscall_ret as a successful result.
 *
 * The round loop is bounded: a domain that never reaches DONE is a reported
 * failure rather than a hang the harness timeout cannot tell from a stall.
 */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "libcapstone.h"
#include "hostcall_stdout_probe.h"
#include "hostcall-file-service-probe-common.h"
#include "file_probe.h"

#define FILE_PROBE_MAX_ROUNDS 16
#define FILE_PROBE_REGION_SIZE HOSTCALL_STDOUT_PROBE_REGION_SIZE

static struct hostcall_file_service_handle_slot
    handle_slots[HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES];

static const char *status_name(long long s) {
  switch (s) {
  case FP_OK:                return "OK";
  case FP_OPEN_FAILED:       return "OPEN_FAILED";
  case FP_WRITE_FAILED:      return "WRITE_FAILED";
  case FP_SEEK_FAILED:       return "SEEK_FAILED";
  case FP_READ_FAILED:       return "READ_FAILED";
  case FP_CONTENT_MISMATCH:  return "CONTENT_MISMATCH (position not tracked?)";
  case FP_CLOSE_FAILED:      return "CLOSE_FAILED";
  case FP_MISSING_OPENED:    return "MISSING_OPENED (error did not cross)";
  case FP_MISSING_WRONG_ERR: return "MISSING_WRONG_ERR (errno lost or sign wrong)";
  case FP_CLOSED_FD_READ:    return "CLOSED_FD_READ (stale slot)";
  case FP_CLOSED_WRONG_ERR:  return "CLOSED_WRONG_ERR";
  default:                   return "UNKNOWN";
  }
}

/* Every failure answer takes the same shape, so it is written once. */
static void respond_error(struct hostcall_v0 *metadata, int err) {
  metadata->result = -1;
  metadata->error = -(hostcall_s64_t)err;
  metadata->phase = HC_V0_PHASE_RESP;
}

static void respond_ok(struct hostcall_v0 *metadata, long long value) {
  metadata->result = (hostcall_s64_t)value;
  metadata->error = 0;
  metadata->phase = HC_V0_PHASE_RESP;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <file_probe.dom>\n", argv[0]);
    return 2;
  }
  /* The probe's own preconditions, asserted rather than assumed: arm 1 must
     create the file, and arm 2 needs its path to be absent. A leftover from an
     earlier run would make arm 2 pass for the wrong reason. */
  unlink(MUSL_FILE_PROBE_PATH);
  unlink(MUSL_FILE_PROBE_MISSING_PATH);

  if (capstone_init()) {
    fprintf(stderr, "file-probe: capstone_init failed\n");
    return 1;
  }
  dom_id_t domain = create_dom(argv[1], NULL);
  if ((long)domain < 0) {
    fprintf(stderr, "file-probe: create_dom failed (%ld)\n", (long)domain);
    capstone_cleanup();
    return 1;
  }

  region_id_t metadata_region = create_region(FILE_PROBE_REGION_SIZE);
  region_id_t payload_region = create_region(FILE_PROBE_REGION_SIZE);
  struct hostcall_v0 *metadata =
      (struct hostcall_v0 *)map_region(metadata_region, FILE_PROBE_REGION_SIZE);
  char *payload = (char *)map_region(payload_region, FILE_PROBE_REGION_SIZE);
  if (!metadata || !payload) {
    fprintf(stderr, "file-probe: map_region failed\n");
    capstone_cleanup();
    return 1;
  }
  memset(metadata, 0, FILE_PROBE_REGION_SIZE);
  memset(payload, 0, FILE_PROBE_REGION_SIZE);

  shared_region_annotated(domain, metadata_region,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED);
  shared_region_annotated(domain, payload_region,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED);

  unsigned serviced = 0;
  char path_snapshot[FILE_PROBE_REGION_SIZE];

  for (unsigned round = 0; round < FILE_PROBE_MAX_ROUNDS; ++round) {
    (void)call_dom(domain);

    /* Snapshot before acting: metadata stays INOUT+SHARED and re-reading it
       mid-service is a TOCTOU (HostCall v0 design note). */
    struct hostcall_v0 request;
    hostcall_snapshot_request(&request, metadata);

    if (request.phase == HC_V0_PHASE_DONE) {
      printf("file-probe: DONE, serviced %u request(s), capstone_main = %lld (%s)\n",
             serviced, (long long)request.result,
             status_name((long long)request.result));
      fflush(stdout);
      if (serviced == MUSL_FILE_PROBE_EXPECTED_ROUNDS &&
          request.result == FP_OK) {
        printf("__CAPSTONE_MUSL_FILE_PROBE_PASSED__\n");
        fflush(stdout);
        hostcall_cleanup_open_handles(handle_slots,
                                      HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES);
        capstone_cleanup();
        return 0;
      }
      fprintf(stderr, "file-probe: FAILED (rounds=%u want %d, status=%lld)\n",
              serviced, MUSL_FILE_PROBE_EXPECTED_ROUNDS,
              (long long)request.result);
      break;
    }
    if (request.phase != HC_V0_PHASE_REQ) {
      fprintf(stderr, "file-probe: unexpected phase %llu\n",
              (unsigned long long)request.phase);
      break;
    }
    if (!hostcall_payload_range_valid(&request)) {
      fprintf(stderr, "file-probe: request out of bounds\n");
      break;
    }

    switch (request.opcode) {
    case HC_V0_OP_WRITE_STDOUT: {
      ssize_t n = write(STDOUT_FILENO, payload + request.offset,
                        (size_t)request.length);
      fflush(stdout);
      if (n < 0) respond_error(metadata, errno);
      else       respond_ok(metadata, n);
      break;
    }
    case HC_V0_OP_FILE_OPEN: {
      const struct hc_file_open_req_v0 *req =
          (const struct hc_file_open_req_v0 *)payload;
      unsigned long long flags = req->flags, mode = req->mode;
      memcpy(path_snapshot, payload + request.offset, (size_t)request.length);
      path_snapshot[request.length] = '\0';
      int fd = open(path_snapshot, (int)flags, (mode_t)mode);
      if (fd < 0) { respond_error(metadata, errno); break; }
      hostcall_u64_t opened = hostcall_allocate_handle_token(
          handle_slots, HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES, fd);
      if (!opened) { int e = errno; close(fd); respond_error(metadata, e); break; }
      printf("file-probe: opened %s as token %llu\n", path_snapshot,
             (unsigned long long)opened);
      fflush(stdout);
      respond_ok(metadata, (long long)opened);
      break;
    }
    case HC_V0_OP_FILE_WRITE: {
      const struct hc_file_write_req_v0 *req =
          (const struct hc_file_write_req_v0 *)payload;
      hostcall_u64_t handle = req->handle, off = req->file_offset;
      int fd = hostcall_lookup_handle_fd(
          handle_slots, HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES, handle);
      if (fd < 0) { respond_error(metadata, errno); break; }
      ssize_t n = pwrite(fd, payload + request.offset, (size_t)request.length,
                         (off_t)off);
      if (n < 0) respond_error(metadata, errno);
      else       respond_ok(metadata, n);
      break;
    }
    case HC_V0_OP_FILE_READ: {
      const struct hc_file_read_req_v0 *req =
          (const struct hc_file_read_req_v0 *)payload;
      hostcall_u64_t handle = req->handle, off = req->file_offset;
      int fd = hostcall_lookup_handle_fd(
          handle_slots, HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES, handle);
      if (fd < 0) { respond_error(metadata, errno); break; }
      ssize_t n = pread(fd, payload + request.offset, (size_t)request.length,
                        (off_t)off);
      if (n < 0) respond_error(metadata, errno);
      else       respond_ok(metadata, n);
      break;
    }
    case HC_V0_OP_FILE_CLOSE: {
      const struct hc_file_close_req_v0 *req =
          (const struct hc_file_close_req_v0 *)payload;
      if (hostcall_close_handle_token(
              handle_slots, HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES,
              req->handle) < 0)
        respond_error(metadata, errno);
      else
        respond_ok(metadata, 0);
      break;
    }
    default:
      fprintf(stderr, "file-probe: unexpected opcode %llu\n",
              (unsigned long long)request.opcode);
      respond_error(metadata, ENOSYS);
      break;
    }
    ++serviced;
  }

  fprintf(stderr, "file-probe: did not reach DONE\n");
  hostcall_cleanup_open_handles(handle_slots,
                                HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES);
  capstone_cleanup();
  return 1;
}
