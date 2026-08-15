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
/* Handle-token table, payload range checks and snapshotting, already written and
   already exercised by the hostcall-file-* probes. Reused rather than rewritten:
   the token semantics are what the domain side was written against. */
#include "../../tests/runtime-qemu/hostcall-file-service-probe-common.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>

#define HC_HOST_MAX_ROUNDS 256
#define HC_HOST_REGION_SIZE HOSTCALL_STDOUT_PROBE_REGION_SIZE

/* REV_TRANSFERRED is not in hostcall_stdout_probe.h, which stops at REV_SHARED
   (0x2). Defined here rather than added to that shared header, exactly as
   xlang_shim_host.c does, because only an arena grant needs it. */
#define HC_HOST_ANNOTATION_REV_TRANSFERRED 0x3UL

/* mruby carves 178,750 bytes over a full run and the revoking allocator NEVER
   returns arena space -- it only shrinks from the top -- so the arena has to
   hold every byte ever allocated, not the peak. 512 KiB is that with headroom.
   It is host memory, not part of the domain image, so it does not count against
   the ~4 MB image ceiling. */
#ifndef HC_HOST_ARENA_SIZE
#define HC_HOST_ARENA_SIZE (512UL * 1024)
#endif


static struct hostcall_file_service_handle_slot
    handles[HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES];

/* Returns 1 if the opcode was a file service and has been answered, 0 if this
   host does not implement it. Every answer sets result/error/phase, so a caller
   never has to guess whether the domain will see a reply. */
static int service_file_op(struct hostcall_v0 *metadata, char *payload,
                           hostcall_u64_t opcode, hostcall_u64_t offset,
                           hostcall_u64_t length) {
  const hostcall_u64_t *req = (const hostcall_u64_t *)payload;
  long result = -1;
  int err = 0;
  int fd;

  switch (opcode) {
  case HC_V0_OP_FILE_OPEN: {
    /* Snapshot the path before acting: the metadata region is INOUT+SHARED and
       the domain may rewrite it at any point. */
    char path[HC_HOST_REGION_SIZE];
    if (offset != HC_FILE_OPEN_REQ_V0_PATH_OFFSET ||
        length == 0 || length >= sizeof(path) ||
        offset > HC_HOST_REGION_SIZE || length > HC_HOST_REGION_SIZE - offset) {
      err = EINVAL;
      break;
    }
    memcpy(path, payload + offset, (size_t)length);
    path[length] = '\0';
    fd = open(path, (int)req[0], (mode_t)req[1]);
    if (fd < 0) {
      err = errno;
      break;
    }
    result = (long)hostcall_allocate_handle_token(
        handles, HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES, fd);
    if (!result) {
      err = EMFILE;
      close(fd);
      result = -1;
    }
    break;
  }

  case HC_V0_OP_FILE_READ:
  case HC_V0_OP_FILE_WRITE: {
    if (offset != HC_FILE_READ_REQ_V0_DATA_OFFSET ||
        offset > HC_HOST_REGION_SIZE || length > HC_HOST_REGION_SIZE - offset) {
      err = EINVAL;
      break;
    }
    fd = hostcall_lookup_handle_fd(handles,
                                   HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES, req[0]);
    if (fd < 0) {
      err = errno;
      break;
    }
    ssize_t moved = opcode == HC_V0_OP_FILE_READ
                        ? pread(fd, payload + offset, (size_t)length, (off_t)req[1])
                        : pwrite(fd, payload + offset, (size_t)length, (off_t)req[1]);
    if (moved < 0)
      err = errno;
    else
      result = (long)moved;
    break;
  }

  case HC_V0_OP_FILE_CLOSE:
    if (hostcall_close_handle_token(handles,
                                    HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES, req[0]))
      err = errno;
    else
      result = 0;
    break;

  case HC_V0_OP_FILE_SYNC:
    fd = hostcall_lookup_handle_fd(handles,
                                   HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES, req[0]);
    if (fd < 0)
      err = errno;
    else if (fsync(fd))
      err = errno;
    else
      result = 0;
    break;

  case HC_V0_OP_FILE_TRUNCATE:
    fd = hostcall_lookup_handle_fd(handles,
                                   HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES, req[0]);
    if (fd < 0)
      err = errno;
    else if (ftruncate(fd, (off_t)req[1]))
      err = errno;
    else
      result = 0;
    break;

  default:
    return 0;
  }

  metadata->result = (hostcall_s64_t)result;
  metadata->error = err;
  metadata->phase = HC_V0_PHASE_RESP;
  return 1;
}

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
#ifdef CAPSTONE_DOMAIN_ARENA
  /* A THIRD region: the linear arena the domain's allocator carves from.
   *
   * Shared REV_TRANSFERRED, not REV_SHARED, because the domain must be able to
   * MREV sub-capabilities of it -- that is the whole point of revoke-on-free,
   * and a borrowed or shared grant cannot do it. Same arrangement as
   * xlang/capstone/xlang_shim_host.c, which is where the pattern comes from.
   *
   * NOT memset here, and never read back: it is the domain's memory once
   * transferred. */
  region_id_t arena_region = create_region(HC_HOST_ARENA_SIZE);
#endif
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
#ifdef CAPSTONE_DOMAIN_ARENA
  /* Share order IS capture order: the domain counts the entries it is given, so
     the arena must be third and stay third. */
  shared_region_annotated(domain, arena_region,
                          HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
                          HC_HOST_ANNOTATION_REV_TRANSFERRED);
#endif

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
      hostcall_cleanup_open_handles(handles,
                                    HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES);
      capstone_cleanup();
      return result == 0 ? 0 : 1;
    }

    if (phase != HC_V0_PHASE_REQ) {
      fprintf(stderr, "hostcall-host: unexpected phase=%llu\n",
              (unsigned long long)phase);
      break;
    }
    if (opcode == HC_V0_OP_WRITE_STDOUT) {
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
      continue;
    }

    if (!service_file_op(metadata, payload, opcode, offset, length)) {
      /* 0xE0/0xE1 are the domain saying "I refused a syscall"; `offset` carries
         the syscall number or fd it refused. Printed on stdout, not stderr, so
         it lands in the serial log next to the domain's own output. */
      printf("hc-host: kind=0x%llx nr/val=%lld arg0=%lld\n",
             (unsigned long long)opcode, (long long)offset, (long long)length);
      fflush(stdout);
      metadata->error = 1;
      metadata->result = -1;
      metadata->phase = HC_V0_PHASE_RESP;
    }
    ++serviced;
  }

  fprintf(stderr, "hostcall-host: did not reach DONE after %u serviced\n",
          serviced);
  capstone_cleanup();
  return 1;
}
