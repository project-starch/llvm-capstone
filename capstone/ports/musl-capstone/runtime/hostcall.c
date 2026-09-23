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
#include <setjmp.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <time.h>
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

/* Optional: what a domain wants done when the program exits from inside a call
   rather than by returning from main. Weak, so a domain that does not define it
   links unchanged and the exit path simply skips it. */
__attribute__((__weak__)) int __capstone_at_exit(int status);

/* Where exit() lands. Armed once domain_main is ready to receive it. */
static jmp_buf hc_exit_jb;
static volatile int hc_exit_armed;
static volatile int hc_exit_status;

#define HC_UNSERVED_MAX 16
static long hc_unserved[HC_UNSERVED_MAX];
static unsigned long hc_unserved_n;

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

static unsigned long long hc_get_u64(unsigned long off) {
  unsigned long long v = 0;
  for (unsigned i = 0; i < 8; i++)
    v |= (unsigned long long)(unsigned char)hc_payload[off + i] << (8 * i);
  return v;
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

/* FILE_STAT_BASIC, the one request whose answer arrives in the payload rather
 * than in metadata.result. Two callers need it: fstat, and lseek's SEEK_END. */
static long hc_stat_basic(struct hc_file *f, unsigned long long *size,
                          unsigned long long *mode) {
  hc_put_u64(0, f->handle);
  hc_put_u64(8, 0); /* flags 0: the conservative fstat slice */
  if (hc_round(HC_V0_OP_FILE_STAT_BASIC, 0, 0) != 0)
    return -EIO;
  if (hc_metadata->error != 0)
    return hc_err();
  if (hc_metadata->length < HC_FILE_STAT_BASIC_RESP_V0_SIZE)
    return -EIO;
  if (size) *size = hc_get_u64(0);
  if (mode) *mode = hc_get_u64(8);
  return 0;
}

/* PATH_ACCESS and PATH_DELETE share FILE_OPEN's layout: flags at 0, the path at
 * offset 8, length in bytes. Written once because only the opcode differs. */
static long hc_path_op(unsigned long long opcode, const char *path,
                       unsigned long long flags) {
  unsigned long len = 0;
  while (path[len]) len++;
  if (len > HC_PAYLOAD_SIZE - HC_PATH_ACCESS_REQ_V0_PATH_OFFSET)
    return -ENAMETOOLONG;
  hc_put_u64(0, flags);
  for (unsigned long i = 0; i < len; i++)
    hc_payload[HC_PATH_ACCESS_REQ_V0_PATH_OFFSET + i] = path[i];
  if (hc_round(opcode, HC_PATH_ACCESS_REQ_V0_PATH_OFFSET, len) != 0)
    return -EIO;
  return hc_metadata->error != 0 ? hc_err() : 0;
}

/* FILE_SYNC and FILE_TRUNCATE: handle-only requests with an empty payload area. */
static long hc_handle_op(long fd, unsigned long long opcode,
                         unsigned long long arg) {
  struct hc_file *f = hc_slot(fd);
  if (!f) return -EBADF;
  hc_put_u64(0, f->handle);
  hc_put_u64(8, arg);
  hc_put_u64(16, 0);
  hc_put_u64(24, 0);
  if (hc_round(opcode, 0, 0) != 0)
    return -EIO;
  return hc_metadata->error != 0 ? hc_err() : 0;
}

/* Only the fields the stat helper actually knows. The rest are zeroed rather
   than invented: a plausible st_dev or st_ino would be a lie that some caller
   eventually compares against another lie. */
static void hc_fill_stat(struct stat *st, unsigned long long size,
                         unsigned long long mode) {
  for (unsigned long i = 0; i < sizeof *st; i++)
    ((char *)st)[i] = 0;
  st->st_size = (off_t)size;
  st->st_mode = (mode_t)mode;
  st->st_nlink = 1;
  st->st_blksize = 4096;
  st->st_blocks = (blkcnt_t)((size + 511) / 512);
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

  /* SEEK_END is the only one that costs a round: the size lives with the helper
     and guessing it would make every later access quietly wrong. */
  unsigned long long base;
  if (whence == 0) {        /* SEEK_SET */
    base = 0;
  } else if (whence == 1) { /* SEEK_CUR */
    base = f->pos;
  } else if (whence == 2) { /* SEEK_END */
    unsigned long long size;
    long rc = hc_stat_basic(f, &size, 0);
    if (rc < 0) return rc;
    base = size;
  } else {
    return -EINVAL;
  }

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

  case SYS_fsync:
  case SYS_fdatasync:
    return hc_handle_op((long)a, HC_V0_OP_FILE_SYNC, 0);

  case SYS_ftruncate:
    return hc_handle_op((long)a, HC_V0_OP_FILE_TRUNCATE, (unsigned long long)b);

  /* musl's unlink() and access() both go through the *at forms. The dirfd is
     accepted and unused for the same reason it is in openat: the helper
     resolves paths in its own working directory and there is no *at family
     behind this protocol for a relative path to be relative to. */
  case SYS_unlinkat:
    return hc_path_op(HC_V0_OP_PATH_DELETE, (const char *)b,
                      HC_PATH_DELETE_FLAG_NONE);

  case SYS_faccessat:
    return hc_path_op(HC_V0_OP_PATH_ACCESS, (const char *)b,
                      HC_PATH_ACCESS_FLAG_EXISTS);

  case SYS_fstat: {
    /* stdout and stderr exist in every domain -- the service writes them --
       so fstat describes them: a character device, and not a terminal (the
       tty ioctl answers ENOTTY). Answering EBADF made programs that check a
       descriptor before using it conclude they have no output: CPython sets
       sys.stdout and sys.stderr to None that way, and print() then writes
       nothing, silently. stdin has no service here and stays EBADF. */
    if ((long)a == 1 || (long)a == 2) {
      hc_fill_stat((struct stat *)b, 0, S_IFCHR | 0620);
      return 0;
    }
    struct hc_file *f = hc_slot((long)a);
    if (!f)
      return -EBADF;
    unsigned long long size, mode;
    long rc = hc_stat_basic(f, &size, &mode);
    if (rc < 0)
      return rc;
    hc_fill_stat((struct stat *)b, size, mode);
    return 0;
  }

  /* stat(path) is fstatat(AT_FDCWD, path, st, 0). There is no path-stat
     opcode and none is needed: open, stat, close is three round trips for a
     call made a handful of times, through the same service fstat is already
     measured against. The open is read-only, which a directory accepts too. */
  case SYS_newfstatat: {
    long fd = hc_open((const char *)b, 0, 0);
    if (fd < 0)
      return fd;
    unsigned long long size, mode;
    long rc = hc_stat_basic(hc_slot(fd), &size, &mode);
    hc_close(fd);
    if (rc < 0)
      return rc;
    hc_fill_stat((struct stat *)c, size, mode);
    return 0;
  }

  /* The identity a domain has is the host service's, and that runs as the
     guest's root: uid and gid 0. It is the same number the zeroed stat fields
     carry, so st_uid == geteuid() holds by construction, not by coincidence. */
  case SYS_getuid:
  case SYS_geteuid:
  case SYS_getgid:
  case SYS_getegid:
    return 0;

  /* open() issues this once, for O_CLOEXEC, and discards the result. There are
     no other processes here for a descriptor to leak into. It does answer
     whether a descriptor EXISTS, though: musl's fstat asks F_GETFD after an
     EBADF, and a 0 for a descriptor nobody opened sent it on to fstatat(fd, "")
     and ENOENT, so fstat(0) reported the wrong error. */
  case SYS_fcntl:
    if ((long)a == 1 || (long)a == 2 || hc_slot((long)a))
      return 0;
    return -EBADF;

  /* A domain has exactly one thread, and 1 is its identifier. This is not an
     invented value in the way a fabricated st_dev would be: nothing outside the
     domain consumes a tid, musl only stores what comes back, and the number is
     the domain's own to choose. Serving it rather than refusing it is what
     empties the unserved list, which is the property that makes the list worth
     reading: a list with one permanent entry teaches everyone to ignore it. */
  case SYS_set_tid_address:
    return 1;

  /* exit_group and exit: see domain_main. The status is the program's and the
     jump lands where capstone_main() would have returned, so everything after
     it runs exactly once either way. This case used to return 0 on the
     reasoning that a refused exit is harmless; it is not. libc-test's mntent
     calls exit() on a failed assertion, and returning from the syscall put
     musl's _Exit in its `for (;;) __syscall(SYS_exit, ec)` loop, which took
     the boot with it. A syscall a caller cannot survive the return of has to
     be served or the domain has to end. */
  case SYS_exit_group:
  case SYS_exit:
    if (hc_exit_armed) {
      hc_exit_status = (int)(long)a;
      /* Whatever the program would have done after main returned still has to
         happen, and only the program knows what that is. A domain that has
         something to report defines this; one that has not pays nothing. */
      if (__capstone_at_exit)
        hc_exit_status = __capstone_at_exit(hc_exit_status);
      longjmp(hc_exit_jb, 1);
    }
    return -ENOSYS;

  /* Time comes from the helper. rdtime would give a counter with a frequency
     the domain has no way to learn, and a timespec built on a guessed timebase
     is the kind of number that looks right until something computes with it.
     One round per call; musl's clock_gettime has no vDSO here to short-cut. */
  case SYS_clock_gettime: {
    hc_put_u64(0, (unsigned long long)a);
    if (hc_round(HC_V0_OP_CLOCK_GETTIME, 0, 0) != 0)
      return -EIO;
    if (hc_metadata->error != 0)
      return hc_err();
    if (hc_metadata->length < HC_CLOCK_GETTIME_RESP_V0_SIZE)
      return -EIO;
    struct timespec *ts = (struct timespec *)b;
    ts->tv_sec = (time_t)hc_get_u64(0);
    ts->tv_nsec = (long)hc_get_u64(8);
    return 0;
  }

  default:
    /* RECORDED, NOT JUST REFUSED. A libc is never finished in the sense that
       every syscall is served; it is finished in the sense that what is not
       served is visible. ENOSYS alone is not visible: musl turns most of them
       into a plausible-looking failure and the caller carries on, so a missing
       opcode surfaces later as wrong behaviour somewhere unrelated. Recording
       the number costs nothing when nothing is missing, and a probe can print
       the list after its work is done, which is safe because printing from here
       would re-enter through stdio. */
    if (hc_unserved_n < HC_UNSERVED_MAX)
      hc_unserved[hc_unserved_n] = n;
    hc_unserved_n++;
    return -ENOSYS;
  }
}

/* Read by probes after the program has finished, never during. Returns the
   total seen, which may exceed what was kept. */
/* Said out loud at exit, because recording is not reporting.
 *
 * A refused syscall comes back as -ENOSYS, musl turns that into an ordinary
 * failed call, and a program that does not check carries on with a wrong
 * answer. The ring below has always held what was refused, but only a caller
 * that asked ever saw it, so a domain that never asked was silently wrong.
 * Every domain now says so on its way out.
 *
 * write(2) and not printf: this runs after the program is finished, and on the
 * exit() path musl's stdio has already been torn down. write goes straight
 * through the hostcall and needs nothing that may have been taken apart.
 *
 * Numbers, not names: a table of three hundred names is not worth the bytes in
 * every domain image when both readers already translate. The suite's runner
 * prints them by name, and check-domain-support.py says which CALL needs each
 * one, which is the question a person actually has.
 */
static void hc_report_unserved(void) {
  if (hc_unserved_n == 0)
    return;
  static const char head[] = "capstone-domain: UNSERVED syscalls:";
  char buf[256];
  unsigned long p = 0;
  for (unsigned long i = 0; i < sizeof head - 1; i++)
    buf[p++] = head[i];
  unsigned long shown = hc_unserved_n < HC_UNSERVED_MAX ? hc_unserved_n : HC_UNSERVED_MAX;
  for (unsigned long i = 0; i < shown && p + 32 < sizeof buf; i++) {
    unsigned long seen = 0, times = 0;
    for (unsigned long j = 0; j < shown; j++) {
      if (hc_unserved[j] != hc_unserved[i])
        continue;
      if (j < i)
        seen = 1;
      times++;
    }
    if (seen)   /* one line per distinct number, with how often it was asked */
      continue;
    buf[p++] = ' ';
    long v = hc_unserved[i];
    char d[20];
    unsigned long k = 0;
    do { d[k++] = (char)('0' + v % 10); v /= 10; } while (v);
    while (k)
      buf[p++] = d[--k];
    if (times > 1) {
      buf[p++] = 'x';
      k = 0;
      do { d[k++] = (char)('0' + times % 10); times /= 10; } while (times);
      while (k)
        buf[p++] = d[--k];
    }
  }
  if (hc_unserved_n > shown && p + 8 < sizeof buf) {
    static const char more[] = " ...";
    for (unsigned long i = 0; i < sizeof more - 1; i++)
      buf[p++] = more[i];
  }
  buf[p++] = '\n';
  write(1, buf, p);
}

unsigned long __capstone_unserved_count(void) { return hc_unserved_n; }
long __capstone_unserved_at(unsigned long i) {
  return i < HC_UNSERVED_MAX && i < hc_unserved_n ? hc_unserved[i] : -1;
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

  /* exit() has to be able to end the program from anywhere. musl's _Exit is
     `__syscall(SYS_exit_group, ec); for (;;) __syscall(SYS_exit, ec);`, so a
     refused exit is not an error the program sees, it is an unbreakable loop,
     and a spinning domain holds the only hart. The way out is the one the C
     library itself uses to leave a call frame: the exit syscall longjmps here.
     Volatile because a local that setjmp returns to may not live in a
     register. */
  volatile int status;
  int jumped = setjmp(hc_exit_jb);
  hc_exit_armed = 1;
  if (jumped)
    status = hc_exit_status;
  else
    status = capstone_main();
  hc_exit_armed = 0;
  hc_report_unserved();

  if (hc_metadata) {
    hc_metadata->opcode = HC_V0_OP_NONE;
    hc_metadata->result = status;
    hc_metadata->phase = HC_V0_PHASE_DONE;
  }
  if (res)
    *res = (unsigned)status;
}
