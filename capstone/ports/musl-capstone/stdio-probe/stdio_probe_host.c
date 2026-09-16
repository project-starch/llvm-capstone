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
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "libcapstone.h"
#include "hostcall_stdout_probe.h"
#include "hostcall-file-service-probe-common.h"
#include "stdio_probe.h"

/* 64, not 16. The bound exists so a domain that never reaches DONE is a
   reported failure rather than a hang the harness timeout cannot tell from a
   stall, and it has to sit above what this probe legitimately needs: stdio
   flushes when it chooses, each iovec entry is its own round, and the file
   service arms add a round apiece. Measured at 16 as "did not reach DONE" with
   no fault anywhere, which is exactly the shape of a bound set too low. */
#define STDIO_PROBE_MAX_ROUNDS 64
#define STDIO_PROBE_REGION_SIZE HOSTCALL_STDOUT_PROBE_REGION_SIZE

static struct hostcall_file_service_handle_slot
    handle_slots[HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES];

static const char *status_name(long long s) {
  switch (s) {
  case SP_OK:               return "OK";
  case SP_FOPEN_W_FAILED:   return "FOPEN_W_FAILED";
  case SP_FPRINTF_FAILED:   return "FPRINTF_FAILED (writev missing?)";
  case SP_FCLOSE_W_FAILED:  return "FCLOSE_W_FAILED";
  case SP_FOPEN_R_FAILED:   return "FOPEN_R_FAILED";
  case SP_FGETS_FAILED:     return "FGETS_FAILED (readv missing?)";
  case SP_CONTENT_MISMATCH: return "CONTENT_MISMATCH";
  case SP_FCLOSE_R_FAILED:  return "FCLOSE_R_FAILED";
  case SP_PRINTF_FAILED:    return "PRINTF_FAILED";
  case SP_SEEK_END_WRONG:   return "SEEK_WRONG";
  default:                  return "UNKNOWN";
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
    fprintf(stderr, "usage: %s <stdio_probe.dom>\n", argv[0]);
    return 2;
  }
  /* The probe's own preconditions, asserted rather than assumed: arm 1 must
     create the file, and arm 2 needs its path to be absent. A leftover from an
     earlier run would make arm 2 pass for the wrong reason. */
  unlink(MUSL_STDIO_PROBE_PATH);

  if (capstone_init()) {
    fprintf(stderr, "stdio-probe: capstone_init failed\n");
    return 1;
  }
  dom_id_t domain = create_dom(argv[1], NULL);
  if ((long)domain < 0) {
    fprintf(stderr, "stdio-probe: create_dom failed (%ld)\n", (long)domain);
    capstone_cleanup();
    return 1;
  }

  region_id_t metadata_region = create_region(STDIO_PROBE_REGION_SIZE);
  region_id_t payload_region = create_region(STDIO_PROBE_REGION_SIZE);
  struct hostcall_v0 *metadata =
      (struct hostcall_v0 *)map_region(metadata_region, STDIO_PROBE_REGION_SIZE);
  char *payload = (char *)map_region(payload_region, STDIO_PROBE_REGION_SIZE);
  if (!metadata || !payload) {
    fprintf(stderr, "stdio-probe: map_region failed\n");
    capstone_cleanup();
    return 1;
  }
  memset(metadata, 0, STDIO_PROBE_REGION_SIZE);
  memset(payload, 0, STDIO_PROBE_REGION_SIZE);

  shared_region_annotated(domain, metadata_region,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED);
  shared_region_annotated(domain, payload_region,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED);

  unsigned serviced = 0;
  char path_snapshot[STDIO_PROBE_REGION_SIZE];

  for (unsigned round = 0; round < STDIO_PROBE_MAX_ROUNDS; ++round) {
    (void)call_dom(domain);

    /* Snapshot before acting: metadata stays INOUT+SHARED and re-reading it
       mid-service is a TOCTOU (HostCall v0 design note). */
    struct hostcall_v0 request;
    hostcall_snapshot_request(&request, metadata);

    if (request.phase == HC_V0_PHASE_DONE) {
      printf("stdio-probe: DONE, serviced %u request(s), capstone_main = %lld (%s)\n",
             serviced, (long long)request.result,
             status_name((long long)request.result));
      fflush(stdout);
      /* No round-count assertion, deliberately. stdio decides for itself when
         to flush, so the round count is a property of buffer sizes rather than
         of correctness. The oracle is content, checked inside the domain, plus
         the line the host printed from the WRITE_STDOUT path. serviced > 0 so
         that a probe which serviced nothing still fails. */
      if (serviced > 0 && request.result == SP_OK) {
        printf("__CAPSTONE_MUSL_STDIO_PROBE_PASSED__\n");
        fflush(stdout);
        hostcall_cleanup_open_handles(handle_slots,
                                      HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES);
        capstone_cleanup();
        return 0;
      }
      fprintf(stderr, "stdio-probe: FAILED (rounds=%u, status=%lld)\n",
              serviced, (long long)request.result);
      break;
    }
    if (request.phase != HC_V0_PHASE_REQ) {
      fprintf(stderr, "stdio-probe: unexpected phase %llu\n",
              (unsigned long long)request.phase);
      break;
    }
    if (!hostcall_payload_range_valid(&request)) {
      fprintf(stderr, "stdio-probe: request out of bounds\n");
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
      printf("stdio-probe: opened %s as token %llu\n", path_snapshot,
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
    case HC_V0_OP_FILE_STAT_BASIC: {
      const struct hc_file_stat_basic_req_v0 *req =
          (const struct hc_file_stat_basic_req_v0 *)payload;
      int fd = hostcall_lookup_handle_fd(
          handle_slots, HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES, req->handle);
      if (fd < 0) { respond_error(metadata, errno); break; }
      struct stat st;
      if (fstat(fd, &st) < 0) { respond_error(metadata, errno); break; }
      struct hc_file_stat_basic_resp_v0 *resp =
          (struct hc_file_stat_basic_resp_v0 *)payload;
      resp->file_size = (hostcall_u64_t)st.st_size;
      resp->mode = (hostcall_u64_t)st.st_mode;
      resp->reserved0 = 0;
      resp->reserved1 = 0;
      metadata->offset = 0;
      metadata->length = HC_FILE_STAT_BASIC_RESP_V0_SIZE;
      respond_ok(metadata, 0);
      break;
    }
    case HC_V0_OP_FILE_SYNC: {
      const struct hc_file_sync_req_v0 *req =
          (const struct hc_file_sync_req_v0 *)payload;
      int fd = hostcall_lookup_handle_fd(
          handle_slots, HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES, req->handle);
      if (fd < 0) { respond_error(metadata, errno); break; }
      if (fsync(fd) < 0) respond_error(metadata, errno);
      else               respond_ok(metadata, 0);
      break;
    }
    case HC_V0_OP_FILE_TRUNCATE: {
      const struct hc_file_truncate_req_v0 *req =
          (const struct hc_file_truncate_req_v0 *)payload;
      hostcall_u64_t want = req->size;
      int fd = hostcall_lookup_handle_fd(
          handle_slots, HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES, req->handle);
      if (fd < 0) { respond_error(metadata, errno); break; }
      if (ftruncate(fd, (off_t)want) < 0) respond_error(metadata, errno);
      else                                respond_ok(metadata, 0);
      break;
    }
    case HC_V0_OP_PATH_ACCESS:
    case HC_V0_OP_PATH_DELETE: {
      memcpy(path_snapshot, payload + request.offset, (size_t)request.length);
      path_snapshot[request.length] = '\0';
      int rc = request.opcode == HC_V0_OP_PATH_ACCESS
                   ? access(path_snapshot, F_OK)
                   : unlink(path_snapshot);
      if (rc < 0) respond_error(metadata, errno);
      else        respond_ok(metadata, 0);
      break;
    }
    default:
      fprintf(stderr, "stdio-probe: unexpected opcode %llu\n",
              (unsigned long long)request.opcode);
      respond_error(metadata, ENOSYS);
      break;
    }
    ++serviced;
  }

  fprintf(stderr, "stdio-probe: did not reach DONE\n");
  hostcall_cleanup_open_handles(handle_slots,
                                HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES);
  capstone_cleanup();
  return 1;
}
