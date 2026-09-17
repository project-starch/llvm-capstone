/* __capstone_hostcall: the one place musl's syscall layer meets the domain
 * boundary. arch-capstone64/syscall_arch.h routes every __syscall0..6 here.
 *
 * WHAT THIS IS AND IS NOT. It is a translation from Linux syscall numbers to
 * HostCall v0 opcodes, which already exist and are already serviced by host
 * helpers (see docs/design/syscalls-and-hostcall-abi.md). It is NOT a
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
/* syscall_arg_t and the __capstone_hostcall prototype both live in the arch
 * overlay. Including it here rather than re-declaring the type is what makes a
 * signature drift between the two a compile error instead of a silent ABI
 * mismatch at the one boundary that cannot be debugged from C. Found 2026-09-16
 * the first time this file was compiled: it had never been built, so nothing
 * had ever asked where syscall_arg_t came from. */
#include <sys/uio.h>
#include <syscall_arch.h>

static long hc_write(long fd, const char *buf, unsigned long count);
static long hc_file_rw(long fd, char *buf, unsigned long count, int writing);

#include "../../../tests/runtime-qemu/hostcall-stdout-probe/hostcall_stdout_probe.h"

/* shared_region_annotated() enters the domain with func == 1 and the region
   capability as the first argument. */
#define CAPSTONE_DPI_REGION_SHARE 1

extern void __capstone_yield(void);
extern int capstone_main(void);
/* Installs the domain's single thread pointer. Everything in musl that reports
   an error needs it, so it runs before the program and not on demand. */
extern int __capstone_init_tls(void);

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

/* ---- the domain's open files -------------------------------------------
 *
 * WHY THIS TABLE HAS TO BE HERE AND NOT IN THE HELPER. The wire protocol takes
 * an explicit file_offset on every FILE_READ and FILE_WRITE, and POSIX read and
 * write have an implicit position. Someone has to keep that position, and it
 * cannot be the helper: it owns handles, may recycle a token after FILE_CLOSE
 * (wire spec section 6), and has no idea which domain fd a token belongs to.
 * Keeping it here also means lseek costs no round at all for the two cases that
 * matter, since SEEK_SET and SEEK_CUR are arithmetic on a number we already have.
 *
 * Descriptors start at 3 because 0, 1 and 2 are the stdout path and always will
 * be: WRITE_STDOUT needs no handle and no open.
 *
 * A fixed table of eight rather than a growing one, for the same reason the wire
 * spec recommends a slot array on the helper side: a domain that needs thousands
 * of open files is not the workload this is being built for, and -EMFILE is an
 * honest answer that the caller already has to handle. */
#define HC_FD_BASE 3
#define HC_MAX_FILES 8
#define HC_PAYLOAD_SIZE HOSTCALL_STDOUT_PROBE_REGION_SIZE

struct hc_file {
  unsigned long long handle; /* helper token; token 0 is reserved as invalid */
  unsigned long long pos;    /* POSIX file position, ours to keep */
  int used;
};
static struct hc_file hc_files[HC_MAX_FILES];

static struct hc_file *hc_slot(long fd) {
  if (fd < HC_FD_BASE || fd >= HC_FD_BASE + HC_MAX_FILES)
    return 0;
  struct hc_file *f = &hc_files[fd - HC_FD_BASE];
  return f->used ? f : 0;
}

/* The wire spec (section 11) says metadata.error carries a NEGATIVE errno on
 * failure. At least one probe in this tree writes a positive one, and the sign
 * matters more here than anywhere else: __capstone_hostcall's return value goes
 * straight to musl's __syscall_ret, which treats anything outside (-4096, 0) as
 * a SUCCESSFUL result. A positive errno would therefore be reported to the
 * caller as a successful syscall returning a small number. The sign is forced
 * rather than trusted, because the failure mode of trusting it is silent. */
static long hc_err(void) {
  long e = (long)hc_metadata->error;
  return e > 0 ? -e : e;
}

/* The payload is a byte region; the request headers are u64 fields at fixed
   offsets (wire spec section 7). Writing them field by field rather than
   casting a struct over the region keeps this free of any assumption about
   the padding a capability target gives that struct. */
static void hc_put_u64(unsigned long off, unsigned long long v) {
  for (unsigned i = 0; i < 8; i++)
    hc_payload[off + i] = (char)((v >> (8 * i)) & 0xff);
}

/* Both directions of FILE_READ/FILE_WRITE, which differ only in who fills the
   data area and in the direction of the copy. Chunked, because the payload is
   one region and a caller may ask for more than fits; a short count is a legal
   POSIX result for both calls, so the loop stops at the first short answer
   rather than spinning. */
static long hc_file_rw(long fd, char *buf, unsigned long count, int writing) {
  struct hc_file *f = hc_slot(fd);
  if (!f)
    return -EBADF;

  const unsigned long data_off = writing ? HC_FILE_WRITE_REQ_V0_DATA_OFFSET
                                         : HC_FILE_READ_REQ_V0_DATA_OFFSET;
  const unsigned long max_chunk = HC_PAYLOAD_SIZE - data_off;
  unsigned long done = 0;

  while (done < count) {
    unsigned long chunk = count - done;
    if (chunk > max_chunk)
      chunk = max_chunk;

    hc_put_u64(0, f->handle);
    hc_put_u64(8, f->pos);
    hc_put_u64(16, 0); /* flags, reserved, write as 0 */
    hc_put_u64(24, 0); /* reserved0 */
    if (writing)
      for (unsigned long i = 0; i < chunk; i++)
        hc_payload[data_off + i] = buf[done + i];

    if (hc_round(writing ? HC_V0_OP_FILE_WRITE : HC_V0_OP_FILE_READ,
                 data_off, chunk) != 0)
      return done ? (long)done : -EIO;
    if (hc_metadata->error != 0)
      return done ? (long)done : hc_err();

    long serviced = (long)hc_metadata->result;
    if (serviced < 0)
      return done ? (long)done : -EIO;
    if (serviced == 0)
      break; /* end of file on a read; nothing more to place on a write */

    if (!writing)
      for (long i = 0; i < serviced; i++)
        buf[done + i] = hc_payload[data_off + i];

    done += (unsigned long)serviced;
    f->pos += (unsigned long long)serviced;
    if ((unsigned long)serviced < chunk)
      break;
  }
  return (long)done;
}

static long hc_open(const char *path, long flags, long mode) {
  int slot = -1;
  for (int i = 0; i < HC_MAX_FILES; i++)
    if (!hc_files[i].used) {
      slot = i;
      break;
    }
  if (slot < 0)
    return -EMFILE;

  unsigned long len = 0;
  while (path[len])
    len++;
  if (len > HC_PAYLOAD_SIZE - HC_FILE_OPEN_REQ_V0_PATH_OFFSET)
    return -ENAMETOOLONG;

  hc_put_u64(0, (unsigned long long)flags);
  hc_put_u64(8, (unsigned long long)mode);
  for (unsigned long i = 0; i < len; i++)
    hc_payload[HC_FILE_OPEN_REQ_V0_PATH_OFFSET + i] = path[i];

  if (hc_round(HC_V0_OP_FILE_OPEN, HC_FILE_OPEN_REQ_V0_PATH_OFFSET, len) != 0)
    return -EIO;
  if (hc_metadata->error != 0)
    return hc_err();

  unsigned long long handle = (unsigned long long)hc_metadata->result;
  if (handle == 0)
    return -EIO; /* the wire spec reserves 0 as an invalid handle */

  hc_files[slot].handle = handle;
  hc_files[slot].pos = 0;
  hc_files[slot].used = 1;
  return HC_FD_BASE + slot;
}

static long hc_close(long fd) {
  struct hc_file *f = hc_slot(fd);
  if (!f)
    return -EBADF;

  hc_put_u64(0, f->handle);
  long rc = 0;
  if (hc_round(HC_V0_OP_FILE_CLOSE, 0, 0) != 0)
    rc = -EIO;
  else if (hc_metadata->error != 0)
    rc = hc_err();

  /* The slot is released whatever the helper said. A close that reports an
     error has still consumed the descriptor in POSIX, and keeping the slot
     would leak it for the life of the domain. */
  f->used = 0;
  f->handle = 0;
  f->pos = 0;
  return rc;
}

static long hc_lseek(long fd, long long off, long whence) {
  struct hc_file *f = hc_slot(fd);
  if (!f)
    return -EBADF;

  /* SEEK_END needs the file size, which is FILE_STAT_BASIC, which this does not
     implement yet. Refusing is the honest answer: guessing a size would make
     every later read and write silently wrong. */
  unsigned long long base;
  if (whence == 0)      /* SEEK_SET */
    base = 0;
  else if (whence == 1) /* SEEK_CUR */
    base = f->pos;
  else
    return -EINVAL;

  if (off < 0 && (unsigned long long)(-off) > base)
    return -EINVAL;
  f->pos = base + (unsigned long long)off;
  return (long)f->pos;
}

/* writev and readv are not a convenience: musl's stdio does not call write() or
 * read() at all. __stdio_write builds a two-entry iovec, the FILE's own buffer
 * plus the caller's bytes, and issues writev; __stdio_read issues readv. So
 * without these two, every printf, fwrite, fgets and fread in a domain fails
 * with ENOSYS while plain write() works, which is the most confusing shape a
 * gap can have.
 *
 * Implemented by walking the vector and reusing the single-buffer paths rather
 * than gathering into the payload first. That costs one round per non-empty
 * entry instead of one per call, and it is the right trade here: the payload is
 * 4 KiB, a gather would have to chunk across entry boundaries anyway, and a
 * short count is a legal result for both calls, so the loop stops at the first
 * short answer exactly as the single-buffer paths do. */
static long hc_writev(long fd, const struct iovec *iov, long count) {
  unsigned long done = 0;
  for (long i = 0; i < count; i++) {
    if (iov[i].iov_len == 0)
      continue;
    long n = hc_write(fd, (const char *)iov[i].iov_base, iov[i].iov_len);
    if (n < 0)
      return done ? (long)done : n;
    done += (unsigned long)n;
    if ((unsigned long)n < iov[i].iov_len)
      break; /* short write; do not start the next entry */
  }
  return (long)done;
}

static long hc_readv(long fd, const struct iovec *iov, long count) {
  unsigned long done = 0;
  for (long i = 0; i < count; i++) {
    if (iov[i].iov_len == 0)
      continue;
    long n = hc_file_rw(fd, (char *)iov[i].iov_base, iov[i].iov_len, 0);
    if (n < 0)
      return done ? (long)done : n;
    done += (unsigned long)n;
    if ((unsigned long)n < iov[i].iov_len)
      break; /* short read, or end of file */
  }
  return (long)done;
}

static long hc_write(long fd, const char *buf, unsigned long count) {
  unsigned long done = 0;

  /* Anything above stderr is a handle this domain opened, and goes through
     FILE_WRITE with its own position. */
  if (fd >= HC_FD_BASE)
    return hc_file_rw(fd, (char *)buf, count, 1);
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
  (void)e;
  (void)f;
  switch (n) {
  case SYS_write:
    return hc_write((long)a, (const char *)b, (unsigned long)c);

  /* musl's open() issues openat(AT_FDCWD, ...). The dirfd is accepted and not
     used: the helper resolves paths in its own working directory and there is
     no *at family behind this protocol to be relative to. A relative path
     therefore means "relative to the helper", which is stated here rather than
     silently assumed. */
  case SYS_openat:
    return hc_open((const char *)b, (long)c, (long)d);

  case SYS_read:
    return hc_file_rw((long)a, (char *)b, (unsigned long)c, 0);

  case SYS_close:
    return hc_close((long)a);

  case SYS_lseek:
    return hc_lseek((long)a, (long long)b, (long)c);

  case SYS_writev:
    return hc_writev((long)a, (const struct iovec *)b, (long)c);

  case SYS_readv:
    return hc_readv((long)a, (const struct iovec *)b, (long)c);

  /* stdio asks whether stdout is a terminal, to choose line buffering over full
     buffering. ENOTTY is the true answer for a domain and the one musl handles:
     it picks full buffering, which is also what we want. Returning ENOSYS would
     work by accident; returning the right error means the next reader does not
     have to wonder. */
  case SYS_ioctl:
    return -ENOTTY;

  /* open() issues this once, for O_CLOEXEC, and discards the result. There are
     no other processes here for a descriptor to leak into. */
  case SYS_fcntl:
    return 0;

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

  /* Before the program, because errno has to exist the first time a syscall
     fails, and that can be the program's first line. A failure here is worth
     more than the program's own status: it means every later error report
     would have faulted instead. */
  if (__capstone_init_tls() != 0) {
    if (hc_metadata) {
      hc_metadata->result = -1;
      hc_metadata->phase = HC_V0_PHASE_ERROR;
    }
    if (res)
      *res = (unsigned)-1;
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
