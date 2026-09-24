/* The host side of HostCall v0, once, for every musl probe host.
 *
 * Three hosts had grown the same switch by copy: write-probe, file-probe and
 * stdio-probe. A fourth would have been the point at which a fix to one of
 * them stops reaching the others, so the switch lives here and a host is now
 * only its regions, its round loop and its oracle.
 *
 * Contract, from hostcall-file-service-v0-wire-spec.md: the caller has already
 * snapshotted the request, checked the payload range, and will set phase RESP
 * itself after this returns. This function services exactly one request and
 * writes result and error. Errors carry a NEGATIVE errno (spec section 11);
 * the domain forces the sign anyway, but this side is the one the spec binds.
 *
 * pread and pwrite, not read and write: the wire carries an explicit offset and
 * the domain keeps the POSIX position, so the helper descriptor's own position
 * must not be allowed to matter.
 */
#ifndef CAPSTONE_MUSL_HOST_SERVICE_H
#define CAPSTONE_MUSL_HOST_SERVICE_H

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "hostcall_stdout_probe.h"
#include "hostcall-file-service-probe-common.h"

struct hc_host {
  struct hostcall_file_service_handle_slot
      slots[HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES];
  char path[HOSTCALL_STDOUT_PROBE_REGION_SIZE + 1];
  int verbose; /* print each OPEN, for probes whose markers want it */
  const char *tag;
};

static inline void hc_host_error(struct hostcall_v0 *m, int err) {
  m->result = -1;
  m->error = -(hostcall_s64_t)err;
}
static inline void hc_host_ok(struct hostcall_v0 *m, long long v) {
  m->result = (hostcall_s64_t)v;
  m->error = 0;
}

/* Every byte the KERNEL moves to or from the payload region goes through this
 * buffer; the region is never handed to a syscall. Services that only memcpy
 * (paths, fixed-size request and response structs) need not use it.
 *
 * OBSERVED (2026-09-23, matched pair): the same domain image read 0 bytes and got
 * EFAULT from /mnt/host/input.mkv on the 9p share, and read the correct 4096 bytes
 * from a /tmp copy; reading through a bounce buffer, with no other change, made the
 * 9p read succeed. The stdout direction (WRITE_STDOUT with the host's stdout on a 9p
 * file) is measured by tests/runtime-qemu/large-io.
 * MECHANISM, INFERRED FROM THE GUEST KERNEL SOURCE (Linux 6.1), NOT MEASURED: the
 * region is mapped with remap_pfn_range (modcapstone module/capstone.c), i.e.
 * VM_IO|VM_PFNMAP, and get-user-pages refuses such a mapping with -EFAULT (mm/gup.c);
 * an uncached 9p read or write larger than 1024 bytes goes zero-copy
 * (net/9p/client.c, trans_virtio.c) and pins the user pages that way. tmpfs and the
 * console use copy_to/from_user, which work on the mapping -- why the musl probes
 * (all under /tmp, all printing to the console) never saw it.
 *
 * Returns the buffer, or NULL after answering EINVAL for a request longer than one
 * region (the domain chunks at the region size, so a longer one is malformed). */
static char hc_host_bounce_buf[HOSTCALL_STDOUT_PROBE_REGION_SIZE];
static inline char *hc_host_bounce(struct hostcall_v0 *metadata, size_t len) {
  if (len > sizeof hc_host_bounce_buf) { hc_host_error(metadata, EINVAL); return NULL; }
  return hc_host_bounce_buf;
}

/* Returns 0 if the opcode was recognised, -1 if not (caller decides). */
static inline int hc_host_service(struct hc_host *h, const struct hostcall_v0 *req,
                                  struct hostcall_v0 *metadata, char *payload) {
  const hostcall_u64_t max = HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES;
  switch (req->opcode) {
  case HC_V0_OP_WRITE_STDOUT: {
    size_t len = (size_t)req->length;
    char *b = hc_host_bounce(metadata, len);
    if (!b) return 0;
    memcpy(b, payload + req->offset, len);
    /* Flush the host's own buffered output FIRST. The domain's bytes go out with
       write() on the descriptor, past stdio's buffer; flushing afterwards, as this
       did, put whatever the host had printed (the loader's messages) in the middle
       of the domain's line. Invisible on a console, where stdio is line-buffered;
       with stdout redirected to a file it split a 5000-byte line at 4096. */
    fflush(stdout);
    ssize_t n = write(STDOUT_FILENO, b, len);
    if (n < 0) hc_host_error(metadata, errno); else hc_host_ok(metadata, n);
    return 0;
  }
  case HC_V0_OP_FILE_OPEN: {
    const struct hc_file_open_req_v0 *r = (const struct hc_file_open_req_v0 *)payload;
    unsigned long long flags = r->flags, mode = r->mode;
    memcpy(h->path, payload + req->offset, (size_t)req->length);
    h->path[req->length] = '\0';
    int fd = open(h->path, (int)flags, (mode_t)mode);
    if (fd < 0) { hc_host_error(metadata, errno); return 0; }
    hostcall_u64_t opened = hostcall_allocate_handle_token(h->slots, max, fd);
    if (!opened) { int e = errno; close(fd); hc_host_error(metadata, e); return 0; }
    if (h->verbose) { printf("%s: opened %s as token %llu\n", h->tag, h->path, (unsigned long long)opened); fflush(stdout); }
    hc_host_ok(metadata, (long long)opened);
    return 0;
  }
  case HC_V0_OP_FILE_WRITE: {
    const struct hc_file_write_req_v0 *r = (const struct hc_file_write_req_v0 *)payload;
    hostcall_u64_t handle = r->handle, off = r->file_offset;
    int fd = hostcall_lookup_handle_fd(h->slots, max, handle);
    if (fd < 0) { hc_host_error(metadata, errno); return 0; }
    size_t wlen = (size_t)req->length;
    char *b = hc_host_bounce(metadata, wlen);
    if (!b) return 0;
    memcpy(b, payload + req->offset, wlen);
    ssize_t n = pwrite(fd, b, wlen, (off_t)off);
    if (n < 0) hc_host_error(metadata, errno); else hc_host_ok(metadata, n);
    return 0;
  }
  case HC_V0_OP_FILE_READ: {
    const struct hc_file_read_req_v0 *r = (const struct hc_file_read_req_v0 *)payload;
    hostcall_u64_t handle = r->handle, off = r->file_offset;
    int fd = hostcall_lookup_handle_fd(h->slots, max, handle);
    if (fd < 0) { hc_host_error(metadata, errno); return 0; }
    size_t rlen = (size_t)req->length;
    char *b = hc_host_bounce(metadata, rlen);
    if (!b) return 0;
    ssize_t n = pread(fd, b, rlen, (off_t)off);
    if (n > 0) memcpy(payload + req->offset, b, (size_t)n);
    if (n < 0) hc_host_error(metadata, errno); else hc_host_ok(metadata, n);
    return 0;
  }
  case HC_V0_OP_FILE_CLOSE: {
    const struct hc_file_close_req_v0 *r = (const struct hc_file_close_req_v0 *)payload;
    if (hostcall_close_handle_token(h->slots, max, r->handle) < 0) hc_host_error(metadata, errno);
    else hc_host_ok(metadata, 0);
    return 0;
  }
  case HC_V0_OP_FILE_STAT_BASIC: {
    const struct hc_file_stat_basic_req_v0 *r = (const struct hc_file_stat_basic_req_v0 *)payload;
    int fd = hostcall_lookup_handle_fd(h->slots, max, r->handle);
    if (fd < 0) { hc_host_error(metadata, errno); return 0; }
    struct stat st;
    if (fstat(fd, &st) < 0) { hc_host_error(metadata, errno); return 0; }
    struct hc_file_stat_basic_resp_v0 *resp = (struct hc_file_stat_basic_resp_v0 *)payload;
    resp->file_size = (hostcall_u64_t)st.st_size;
    resp->mode = (hostcall_u64_t)st.st_mode;
    resp->reserved0 = resp->reserved1 = 0;
    metadata->offset = 0;
    metadata->length = HC_FILE_STAT_BASIC_RESP_V0_SIZE;
    hc_host_ok(metadata, 0);
    return 0;
  }
  case HC_V0_OP_FILE_SYNC: {
    const struct hc_file_sync_req_v0 *r = (const struct hc_file_sync_req_v0 *)payload;
    int fd = hostcall_lookup_handle_fd(h->slots, max, r->handle);
    if (fd < 0) { hc_host_error(metadata, errno); return 0; }
    if (fsync(fd) < 0) hc_host_error(metadata, errno); else hc_host_ok(metadata, 0);
    return 0;
  }
  case HC_V0_OP_FILE_TRUNCATE: {
    const struct hc_file_truncate_req_v0 *r = (const struct hc_file_truncate_req_v0 *)payload;
    hostcall_u64_t want = r->size;
    int fd = hostcall_lookup_handle_fd(h->slots, max, r->handle);
    if (fd < 0) { hc_host_error(metadata, errno); return 0; }
    if (ftruncate(fd, (off_t)want) < 0) hc_host_error(metadata, errno); else hc_host_ok(metadata, 0);
    return 0;
  }
  case HC_V0_OP_PATH_ACCESS:
  case HC_V0_OP_PATH_DELETE:
  case HC_V0_OP_PATH_MKDIR: {
    /* One layout for the three: the flags word at 0 (PATH_DELETE's directory
       flag, PATH_MKDIR's mode), the path at req->offset. */
    const struct hc_file_open_req_v0 *r = (const struct hc_file_open_req_v0 *)payload;
    unsigned long long flags = r->flags;
    memcpy(h->path, payload + req->offset, (size_t)req->length);
    h->path[req->length] = '\0';
    int rc;
    if (req->opcode == HC_V0_OP_PATH_ACCESS) rc = access(h->path, F_OK);
    else if (req->opcode == HC_V0_OP_PATH_MKDIR) rc = mkdir(h->path, (mode_t)flags);
    else if (flags & HC_PATH_DELETE_FLAG_DIRECTORY) rc = rmdir(h->path);
    else rc = unlink(h->path);
    if (rc < 0) hc_host_error(metadata, errno); else hc_host_ok(metadata, 0);
    return 0;
  }
  case HC_V0_OP_PATH_RENAME: {
    /* "old NUL new" in one range. The NUL has to be inside it, or there is no
       second path and the request is malformed rather than a rename to "". */
    memcpy(h->path, payload + req->offset, (size_t)req->length);
    h->path[req->length] = '\0';
    size_t lo = strnlen(h->path, (size_t)req->length);
    if (lo >= (size_t)req->length) { hc_host_error(metadata, EINVAL); return 0; }
    if (rename(h->path, h->path + lo + 1) < 0) hc_host_error(metadata, errno);
    else hc_host_ok(metadata, 0);
    return 0;
  }
  case HC_V0_OP_PATH_READLINK: {
    /* The target goes back at payload offset 0, over the request; the path is
       copied out first. The payload holds any target Linux can return. */
    memcpy(h->path, payload + req->offset, (size_t)req->length);
    h->path[req->length] = '\0';
    ssize_t n = readlink(h->path, payload, HOSTCALL_STDOUT_PROBE_REGION_SIZE);
    if (n < 0) { hc_host_error(metadata, errno); return 0; }
    metadata->offset = 0;
    metadata->length = (hostcall_u64_t)n;
    hc_host_ok(metadata, n);
    return 0;
  }
  case HC_V0_OP_DIR_READ: {
    /* lseek to the cookie, then getdents64 into the bounce buffer and on into
       the payload: the records are linux_dirent64 on both sides (the domain's
       musl uses the same 64-bit layout), so they travel unconverted. A cookie of
       0 restarts the listing, which is also what rewinddir sends. */
    const struct hc_dir_read_req_v0 *r = (const struct hc_dir_read_req_v0 *)payload;
    int fd = hostcall_lookup_handle_fd(h->slots, max, r->handle);
    if (fd < 0) { hc_host_error(metadata, errno); return 0; }
    if (lseek(fd, (off_t)r->cookie, SEEK_SET) < 0) { hc_host_error(metadata, errno); return 0; }
    size_t dlen = (size_t)req->length;
    char *b = hc_host_bounce(metadata, dlen);
    if (!b) return 0;
    long n = syscall(SYS_getdents64, fd, b, dlen);
    if (n < 0) { hc_host_error(metadata, errno); return 0; }
    memcpy(payload + req->offset, b, (size_t)n);
    metadata->length = (hostcall_u64_t)n;
    hc_host_ok(metadata, n);
    return 0;
  }
  case HC_V0_OP_CLOCK_GETTIME: {
    const struct hc_clock_gettime_req_v0 *r = (const struct hc_clock_gettime_req_v0 *)payload;
    struct timespec ts;
    if (clock_gettime((clockid_t)r->clock_id, &ts) < 0) { hc_host_error(metadata, errno); return 0; }
    struct hc_clock_gettime_resp_v0 *resp = (struct hc_clock_gettime_resp_v0 *)payload;
    resp->sec = (hostcall_s64_t)ts.tv_sec;
    resp->nsec = (hostcall_s64_t)ts.tv_nsec;
    metadata->offset = 0;
    metadata->length = HC_CLOCK_GETTIME_RESP_V0_SIZE;
    hc_host_ok(metadata, 0);
    return 0;
  }
  default:
    return -1;
  }
}

#endif
