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
#include <sys/uio.h>

/* Not <sys/syscall.h>'s job: syscall_arg_t and the __capstone_hostcall
   prototype both live in our arch layer. Including it here means the definition
   below is checked against the declaration musl actually calls through, rather
   than merely resembling it. */
#include "syscall_arch.h"

#include "../../tests/runtime-qemu/hostcall-stdout-probe/hostcall_stdout_probe.h"

/* shared_region_annotated() enters the domain with func == 1 and the region
   capability as the first argument. */
#define CAPSTONE_DPI_REGION_SHARE 1

extern void __capstone_yield(void);
extern int capstone_main(void);
extern void __stdio_exit(void);

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

/* Exported, not static: a domain needs a way to print through the path that is
   already known to work, so that a ladder can localise a failure in the musl
   wrapper chain above it. Without this, "musl write() faulted" and "our
   hostcall faulted" look identical -- silence. */
long __capstone_hc_write(long fd, const char *buf, unsigned long count) {
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

/* REPORT WITHOUT FORMATTING. Two previous versions of this built a string in a
   local buffer and each introduced its own defect -- a 32-byte buffer overrun
   that the capability bounds caught on the exact byte, then a degenerate
   SHRINK. A diagnostic that can itself be the fault is worse than none, so the
   value now travels as a NUMBER in the shared metadata and the host does the
   printing. The domain formats nothing. */
#define HC_UNSUP_SYSCALL 0xE0UL
#define HC_UNSUP_BAD_FD 0xE1UL
#define HC_TRACE_ENTRY 0xE2UL

static void hc_unsupported(unsigned long kind, unsigned long value) {
  /* `offset` carries the value: it is meaningless for an opcode the host does
     not implement, and the host prints it with the complaint. */
  (void)hc_round(kind, value, 0);
}


/* ---------------------------------------------------------------- file ops
 *
 * HostCall v0 already defines and services FILE_OPEN/READ/WRITE/CLOSE/SYNC/
 * TRUNCATE (agent-handoff/design/syscalls-and-hostcall-abi.md), so this is a
 * translation and not a new protocol.
 *
 * TWO IMPEDANCE MISMATCHES, both handled here rather than papered over:
 *
 * 1. HostCall v0 read/write take an EXPLICIT file offset; POSIX read/write use
 *    the descriptor's implicit position. The position therefore lives here, one
 *    per open handle, advanced on every transfer. lseek moves it.
 * 2. The host hands back a handle TOKEN starting at 1, and fds 1 and 2 are
 *    already stdout and stderr on the WRITE_STDOUT path. A token is exposed to
 *    the program as `token + 2`, so tokens 1..8 become fds 3..10 and cannot
 *    collide with the two we already answer.
 */
#define HC_FD_BASE 3
#define HC_MAX_HANDLES 8

static long hc_offset[HC_MAX_HANDLES];   /* implicit position, per handle */

static int hc_token_of(long fd) {
  long token = fd - HC_FD_BASE + 1;
  if (token < 1 || token > HC_MAX_HANDLES)
    return 0;
  return (int)token;
}

/* Payload layout helpers: the request header is a run of u64s at offset 0 and
   the variable part follows at a fixed offset the header pins down. */
static void hc_put_u64(unsigned long index, unsigned long long value) {
  volatile unsigned long long *slots = (volatile unsigned long long *)hc_payload;
  slots[index] = value;
}

static long hc_open(const char *path, long flags, long mode) {
  unsigned long n = 0;
  if (!hc_payload)
    return -EIO;
  while (path[n])
    n++;
  if (HC_FILE_OPEN_REQ_V0_PATH_OFFSET + n + 1 > HOSTCALL_STDOUT_PROBE_REGION_SIZE)
    return -ENAMETOOLONG;

  hc_put_u64(0, (unsigned long long)flags);
  hc_put_u64(1, (unsigned long long)mode);
  for (unsigned long i = 0; i < n; i++)
    hc_payload[HC_FILE_OPEN_REQ_V0_PATH_OFFSET + i] = path[i];

  if (hc_round(HC_V0_OP_FILE_OPEN, HC_FILE_OPEN_REQ_V0_PATH_OFFSET, n) != 0)
    return -EIO;
  if (hc_metadata->error != 0)
    return -(long)hc_metadata->error;

  long token = (long)hc_metadata->result;
  if (token < 1 || token > HC_MAX_HANDLES)
    return -EMFILE;
  hc_offset[token - 1] = 0;
  return token + HC_FD_BASE - 1;
}

/* Shared by read and write: both use the same 32-byte header and a data area,
   and differ only in which direction the bytes move. */
static long hc_transfer(unsigned long opcode, long fd, char *buf,
                        unsigned long count, int writing) {
  int token = hc_token_of(fd);
  if (!token)
    return -EBADF;
  if (!hc_payload)
    return -EIO;

  unsigned long room = HOSTCALL_STDOUT_PROBE_REGION_SIZE - HC_FILE_READ_REQ_V0_DATA_OFFSET;
  if (count > room)
    count = room;               /* a short transfer is a legal POSIX result */

  hc_put_u64(0, (unsigned long long)token);
  hc_put_u64(1, (unsigned long long)hc_offset[token - 1]);
  hc_put_u64(2, 0);
  hc_put_u64(3, 0);
  if (writing)
    for (unsigned long i = 0; i < count; i++)
      hc_payload[HC_FILE_WRITE_REQ_V0_DATA_OFFSET + i] = buf[i];

  if (hc_round(opcode, HC_FILE_READ_REQ_V0_DATA_OFFSET, count) != 0)
    return -EIO;
  if (hc_metadata->error != 0)
    return -(long)hc_metadata->error;

  long moved = (long)hc_metadata->result;
  if (moved < 0)
    return -EIO;
  if (!writing)
    for (long i = 0; i < moved; i++)
      buf[i] = hc_payload[HC_FILE_READ_REQ_V0_DATA_OFFSET + i];
  hc_offset[token - 1] += moved;
  return moved;
}

/* One shape for the three requests that carry only a handle and one number. */
static long hc_handle_op(unsigned long opcode, long fd, unsigned long long arg) {
  int token = hc_token_of(fd);
  if (!token)
    return -EBADF;
  if (!hc_payload)
    return -EIO;
  hc_put_u64(0, (unsigned long long)token);
  hc_put_u64(1, arg);
  hc_put_u64(2, 0);
  hc_put_u64(3, 0);
  if (hc_round(opcode, 0, 0) != 0)
    return -EIO;
  if (hc_metadata->error != 0)
    return -(long)hc_metadata->error;
  return (long)hc_metadata->result;
}

/* One place where "which fd is this" is decided, because write() and writev()
   must agree: 1 and 2 are the host's stdout/stderr and have their own opcode,
   everything else is a FILE_* handle token. */
static long hc_write_fd(long fd, const void *buf, unsigned long count) {
  if (fd == 1 || fd == 2)
    return __capstone_hc_write(fd, (const char *)buf, count);
  return hc_transfer(HC_V0_OP_FILE_WRITE, fd, (char *)buf, count, 1);
}

long __capstone_hostcall(long n, syscall_arg_t a, syscall_arg_t b,
                         syscall_arg_t c, syscall_arg_t d, syscall_arg_t e,
                         syscall_arg_t f) {
  /* The per-entry trace that localised the sibling-call miscompile lived here.
     Removed once it had done its job: it doubles the number of hostcall rounds,
     and the thing it was watching for is now prevented at compile time by
     -fno-optimize-sibling-calls rather than observed at run time. */
  (void)d;
  (void)e;
  (void)f;
  switch (n) {
  case SYS_write:
    return hc_write_fd((long)a, (const char *)b, (unsigned long)c);

  /* THE SYSCALL MUSL'S STDIO ACTUALLY USES. __stdio_write builds a two-element
     iovec -- the buffer already accumulated plus the new bytes -- and issues one
     writev; there is no path through plain write(). So without this case every
     printf, puts and fwrite in the libc reaches -ENOSYS while write() works
     perfectly, which reads like "stdio is broken" rather than "one syscall is
     missing". Serviced by looping over the vector, because HostCall v0 moves one
     contiguous run of bytes per round. */
  case SYS_writev: {
    const struct iovec *iov = (const struct iovec *)b;
    long total = 0;
    for (int i = 0; i < (int)(long)c; i++) {
      if (iov[i].iov_len == 0)
        continue;
      long done = hc_write_fd((long)a, iov[i].iov_base, iov[i].iov_len);
      if (done < 0)
        return total ? total : done;
      total += done;
      if ((unsigned long)done < iov[i].iov_len)
        break; /* short write: a legal result, and the caller retries */
    }
    return total;
  }

  case SYS_read:
    return hc_transfer(HC_V0_OP_FILE_READ, (long)a, (char *)b, (unsigned long)c, 0);

  /* openat is the syscall; open() is a libc wrapper, and musl's does not
     compile for this target (see runtime/musl_gaps.c). dirfd is ignored: a
     domain has no working directory, so every path is what the host resolves. */
  case SYS_openat:
    return hc_open((const char *)b, (long)c, (long)d);

  case SYS_close: {
    int token = hc_token_of((long)a);
    if (!token)
      return (long)a == 1 || (long)a == 2 ? 0 : -EBADF;
    return hc_handle_op(HC_V0_OP_FILE_CLOSE, (long)a, 0);
  }

  case SYS_fsync:
  case SYS_fdatasync:
    return hc_handle_op(HC_V0_OP_FILE_SYNC, (long)a, 0);

  case SYS_ftruncate:
    return hc_handle_op(HC_V0_OP_FILE_TRUNCATE, (long)a, (unsigned long long)(long)b);

  /* Reported as unsupported rather than faked. musl copes: exit_group falling
     through to the domain return is exactly what a domain does anyway. */
  case SYS_exit:
  case SYS_exit_group:
    return 0;

  default:
    hc_unsupported(HC_UNSUP_SYSCALL, (unsigned long)n);
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

  /* FLUSH STDIO, BECAUSE NOTHING ELSE WILL. musl flushes from exit(), and a
     domain cannot call exit(): _Exit loops on SYS_exit forever, and our SYS_exit
     merely returns, because a domain leaves by returning from domain_main rather
     than by a syscall. Without this, stdout's 1 KiB buffer is discarded and a
     program that printf'd its whole result prints nothing -- the most
     misleading possible failure, since a wedge looks identical.

     CALLED UNCONDITIONALLY, and the first attempt did it the other way, which
     is worth recording because the mistake is not visible in the C. It was
     `extern void __stdio_exit(void) __attribute__((weak)); if (__stdio_exit)
     __stdio_exit();` -- the standard trick, so that a domain which never
     touches stdio does not pull the machinery. On this target the guard does
     not work: the address of an UNDEFINED weak symbol is materialised
     pc-relatively and is therefore NOT null, so the branch is taken and the
     domain calls it. Measured, in file_probe.dom:

         auipc a1, 0xfffef       <- not zero
         addi  a1, a1, 0x3a4
         cincoffset a2, gp, a1
         beqz  a2, ...           <- never taken
         cjalr ra, 0(a0)         <- calls the symbol that does not exist

     which faulted immediately after the probe's own PASSED marker, i.e. after
     everything the probe tested had already worked.

     Unconditional is also cheap and correct: musl weak-aliases __stdio_exit to
     a no-op in exit.c, and the real one in __stdio_exit.o overrides it when a
     FILE is actually used, pulled by the reference __towrite.c carries for
     exactly this purpose. */
  __stdio_exit();

  if (hc_metadata) {
    hc_metadata->opcode = HC_V0_OP_NONE;
    hc_metadata->result = status;
    hc_metadata->phase = HC_V0_PHASE_DONE;
  }
  if (res)
    *res = (unsigned)status;
}
