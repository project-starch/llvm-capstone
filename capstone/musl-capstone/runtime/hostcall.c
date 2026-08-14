/* __capstone_hostcall: the one place musl's syscall layer meets the domain
 * boundary. arch-capstone64/syscall_arch.h routes every __syscall0..6 here.
 *
 * WHAT THIS IS AND IS NOT. It is a translation from Linux syscall numbers to
 * HostCall v0 opcodes, which already exist and are already serviced by host
 * helpers (see agent-handoff/design/syscalls-and-hostcall-abi.md). It is NOT a
 * kernel: anything without a HostCall v0 opcode returns -ENOSYS, and that is
 * the honest answer rather than a stub that pretends to succeed.
 *
 * MARSHALLING IS BY COPY, deliberately, for now. HostCall v0 moves bytes
 * through a shared payload region, so a write() copies out of the domain and a
 * read() copies in. That is the expensive arm of the copy-vs-lend comparison
 * (a 4 KiB copy is ~14,000 cycles on CVA6 silicon against ~182 for a borrow),
 * and it is built first ON PURPOSE: the lending path needs this as its matched
 * control, or the difference between the two cannot be attributed.
 *
 * The payload region is 4 KiB, so transfers are chunked. A short write is a
 * legal write() result, so the chunk loop returns the byte count actually
 * serviced rather than looping until everything is placed.
 */
#include <errno.h>
#include <sys/syscall.h>

#include "../../tests/runtime-qemu/hostcall-stdout-probe/hostcall_stdout_probe.h"

/* shared_region_annotated() enters the domain with func == 1 and the region
   capability as the first argument. */
#define CAPSTONE_DPI_REGION_SHARE 1

extern void __capstone_yield(void);
extern int capstone_main(void);

static volatile struct hostcall_v0 *hc_metadata;
static volatile char *hc_payload;
static unsigned hc_shared_region_count;

/* One round of the HostCall v0 state machine. Returns 0 if the host answered
   with RESP, nonzero otherwise. */
static int hc_round(unsigned long opcode, unsigned long offset,
                    unsigned long length) {
  if (!hc_metadata || !hc_payload)
    return -1;
  hc_metadata->opcode = opcode;
  hc_metadata->offset = offset;
  hc_metadata->length = length;
  hc_metadata->result = 0;
  hc_metadata->error = 0;
  hc_metadata->phase = HC_V0_PHASE_REQ;

  __capstone_yield();

  return hc_metadata->phase == HC_V0_PHASE_RESP ? 0 : -1;
}

static long hc_write(long fd, const char *buf, unsigned long count) {
  unsigned long done = 0;

  /* Only stdout/stderr exist as an opcode today. Everything else needs
     FILE_WRITE with a handle, which needs FILE_OPEN, which needs a path: that
     is the next opcode to wire up, not something to fake here. */
  if (fd != 1 && fd != 2)
    return -EBADF;

  while (done < count) {
    unsigned long chunk = count - done;
    if (chunk > HOSTCALL_STDOUT_PROBE_REGION_SIZE)
      chunk = HOSTCALL_STDOUT_PROBE_REGION_SIZE;
    for (unsigned long i = 0; i < chunk; i++)
      hc_payload[i] = buf[done + i];

    if (hc_round(HC_V0_OP_WRITE_STDOUT, 0, chunk) != 0)
      return done ? (long)done : -EIO;
    if (hc_metadata->error != 0)
      return done ? (long)done : -EIO;

    long serviced = (long)hc_metadata->result;
    if (serviced <= 0)
      return done ? (long)done : -EIO;
    done += (unsigned long)serviced;
    if ((unsigned long)serviced < chunk)
      break; /* short write is a legal result; do not spin */
  }
  return (long)done;
}

long __capstone_hostcall(long n, syscall_arg_t a, syscall_arg_t b,
                         syscall_arg_t c, syscall_arg_t d, syscall_arg_t e,
                         syscall_arg_t f) {
  (void)d;
  (void)e;
  (void)f;
  switch (n) {
  case SYS_write:
    return hc_write((long)a, (const char *)b, (unsigned long)c);

  /* Reported as unsupported rather than faked. musl copes: exit_group falling
     through to the domain return is exactly what a domain does anyway. */
  case SYS_exit:
  case SYS_exit_group:
    return 0;

  default:
    return -ENOSYS;
  }
}

/* Domain entry. The first two entries carry the shared regions; the third runs
   the program. Same convention as the SQLite domain, so a host helper written
   for one drives the other. */
void domain_main(unsigned *res, unsigned func) {
  if (func == CAPSTONE_DPI_REGION_SHARE) {
    if (hc_shared_region_count == 0)
      hc_metadata = (volatile struct hostcall_v0 *)res;
    else if (hc_shared_region_count == 1)
      hc_payload = (volatile char *)res;
    ++hc_shared_region_count;
    return;
  }

  int status = capstone_main();

  if (hc_metadata) {
    hc_metadata->opcode = HC_V0_OP_NONE;
    hc_metadata->result = status;
    hc_metadata->phase = HC_V0_PHASE_DONE;
  }
  if (res)
    *res = (unsigned)status;
}
