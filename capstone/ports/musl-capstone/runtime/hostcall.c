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
#include <fcntl.h>
#include <poll.h>
#include <setjmp.h>
#include <stdlib.h>
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

#ifdef CAPSTONE_PROGRAM_REGIONS
/* Regions the host shares AFTER the two HostCall v0 regions belong to the program: parked here
   as they arrive and handed over once each by __capstone_region. The first use is a heap the
   host transfers LINEAR (REV_TRANSFERRED), which the Sublet heap (sublet_heap.c) carves and
   revokes; without this they would be counted and dropped. Opt-in, so every domain built
   without it is byte-identical to before. A linear capability moves when it is loaded on
   hardware that enforces linearity, so the slot is read once and cleared. */
#define HC_PROGRAM_REGIONS 2
static void *hc_program_region[HC_PROGRAM_REGIONS];

void *__capstone_region(unsigned index) {
  if (index >= HC_PROGRAM_REGIONS)
    return 0;
  void *r = hc_program_region[index];
  hc_program_region[index] = 0;
  return r;
}
#endif

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
   rather than by returning from main. A domain overrides it with a strong
   definition; this weak one is what every other domain gets.

   DEFINED, not merely declared weak. An undefined weak symbol's address is not
   NULL in a domain (C-56): it is formed pc-relative against gp, the linker
   resolves the symbol to 0 at the link address, and the image runs at another
   base without relocation, so the old `if (__capstone_at_exit)` was always true
   and exit() called the image base. */
__attribute__((__weak__)) int __capstone_at_exit(int status) { return status; }

/* Where exit() lands. Armed once domain_main is ready to receive it. */
static jmp_buf hc_exit_jb;
static volatile int hc_exit_armed;
static volatile int hc_exit_status;

#define HC_UNSERVED_MAX 16
static long hc_unserved[HC_UNSERVED_MAX];
static unsigned long hc_unserved_n;

/* Syscalls answered here with a success that has no service behind it -- a
   timer that will never fire -- recorded like the unserved ones and printed at
   exit beside them, so that "served" never quietly comes to mean "pretended". */
#define HC_NOOP_MAX 16
static long hc_noop[HC_NOOP_MAX];
static unsigned long hc_noop_n;
static void hc_note_noop(long n) {
  if (hc_noop_n < HC_NOOP_MAX)
    hc_noop[hc_noop_n] = n;
  hc_noop_n++;
}

/* The mask a program sees when it asks; the helper's own applies to the files
   it creates. 022, the usual default. */
static long hc_umask = 022;

/* stdout and stderr are served by WRITE_STDOUT, so they exist from the start;
   a domain may close them, and a closed one then answers like any descriptor
   nobody opened. stdin has no service here and never exists. Every syscall that
   takes a descriptor asks this and hc_slot, so all of them agree on which
   descriptors exist: fstat, fcntl, ioctl, lseek, write and close. */
static int hc_stdio_closed[3];
static int hc_is_stdio(long fd) { return (fd == 1 || fd == 2) && !hc_stdio_closed[fd]; }

static struct hc_file *hc_slot(long fd) {
  if (fd < HC_FD_BASE || fd >= HC_FD_BASE + HC_MAX_FILES)
    return 0;
  struct hc_file *f = &hc_files[fd - HC_FD_BASE];
  return f->used ? f : 0;
}

/* PIPES, kept in the domain. A pipe is a byte queue between two descriptors,
 * and with one thread on one hart nothing can ever fill or drain it while the
 * program waits on it, so a pipe here NEVER BLOCKS: an empty read answers
 * EAGAIN (or 0 once the write end is closed), a full write EAGAIN, whatever the
 * O_NONBLOCK flag says. That is the only useful answer -- the blocking one is
 * a deadlock -- and it is what the one consumer asks for anyway: PostgreSQL's
 * self-pipe latch (WAIT_USE_SELF_PIPE) sets both ends non-blocking, writes a
 * byte to raise the latch and reads until EAGAIN to lower it. The descriptors
 * sit above the file table, so hc_slot and this never disagree about a number.
 * poll() on them (hc_poll) reports what is queued; its timeout is where the
 * same fact shows again, see there. */
#define HC_PIPE_FD_BASE (HC_FD_BASE + HC_MAX_FILES)
#define HC_MAX_PIPES 2
#define HC_PIPE_BYTES 4096

struct hc_pipe {
  char buf[HC_PIPE_BYTES];
  unsigned head, count; /* ring: the next byte to read, and how many are queued */
  int rd_open, wr_open;
};
static struct hc_pipe hc_pipes[HC_MAX_PIPES];

/* The pipe behind fd, and which end it is (0 read, 1 write), if that end is open. */
static struct hc_pipe *hc_pipe_end(long fd, int *writing) {
  if (fd < HC_PIPE_FD_BASE || fd >= HC_PIPE_FD_BASE + 2 * HC_MAX_PIPES)
    return 0;
  struct hc_pipe *p = &hc_pipes[(fd - HC_PIPE_FD_BASE) / 2];
  *writing = (int)((fd - HC_PIPE_FD_BASE) % 2);
  if (*writing ? !p->wr_open : !p->rd_open)
    return 0;
  return p;
}

static long hc_pipe2(int *fds, long flags) {
  if (flags & ~(O_CLOEXEC | O_NONBLOCK))
    return -EINVAL;
  int i;
  for (i = 0; i < HC_MAX_PIPES && (hc_pipes[i].rd_open || hc_pipes[i].wr_open); i++)
    ;
  if (i == HC_MAX_PIPES)
    return -EMFILE;
  hc_pipes[i].head = hc_pipes[i].count = 0;
  hc_pipes[i].rd_open = hc_pipes[i].wr_open = 1;
  fds[0] = (int)(HC_PIPE_FD_BASE + 2 * i);
  fds[1] = (int)(HC_PIPE_FD_BASE + 2 * i + 1);
  return 0;
}

static long hc_pipe_read(struct hc_pipe *p, char *buf, unsigned long count) {
  if (p->count == 0)
    return p->wr_open ? -EAGAIN : 0; /* nothing queued: try again, or end of file */
  unsigned long n = count < p->count ? count : p->count;
  for (unsigned long i = 0; i < n; i++)
    buf[i] = p->buf[(p->head + i) % HC_PIPE_BYTES];
  p->head = (p->head + (unsigned)n) % HC_PIPE_BYTES;
  p->count -= (unsigned)n;
  return (long)n;
}

static long hc_pipe_write(struct hc_pipe *p, const char *buf, unsigned long count) {
  if (!p->rd_open)
    return -EPIPE; /* nobody will ever read it; no SIGPIPE here to say so */
  unsigned long room = HC_PIPE_BYTES - p->count;
  if (room == 0)
    return count ? -EAGAIN : 0;
  unsigned long n = count < room ? count : room; /* a short write is a legal write */
  for (unsigned long i = 0; i < n; i++)
    p->buf[(p->head + p->count + (unsigned)i) % HC_PIPE_BYTES] = buf[i];
  p->count += (unsigned)n;
  return (long)n;
}

static long hc_pipe_close(long fd) {
  int writing;
  struct hc_pipe *p = hc_pipe_end(fd, &writing);
  if (!p)
    return -EBADF;
  if (writing)
    p->wr_open = 0;
  else
    p->rd_open = 0;
  return 0;
}

/* Does fd name an open pipe end. */
static int hc_pipe_exists(long fd) {
  int writing;
  return hc_pipe_end(fd, &writing) != 0;
}

/* poll(), which musl issues as ppoll. Readiness is a fact this side knows for
 * every descriptor it has: a pipe end has bytes queued or room for them, a
 * file is always ready, stdout and stderr always take a write, anything else
 * is POLLNVAL. The TIMEOUT is not a fact this side can act on: with one thread
 * on one hart nothing can make a descriptor ready while the program waits, so
 * a wait with nothing ready returns 0 at once, as a timeout would, whatever
 * the timeout was -- and that call is recorded under NO-OP in the exit report,
 * because a program that expected to sleep did not. A zero timeout asks only
 * for the facts and is served in full. */
static long hc_poll(struct pollfd *fds, unsigned long n, const long *ts) {
  long ready = 0;
  for (unsigned long i = 0; i < n; i++) {
    long fd = fds[i].fd;
    short want = fds[i].events, got = 0;
    int writing;
    struct hc_pipe *p;
    if (fd < 0) {
      /* skipped by request */
    } else if ((p = hc_pipe_end(fd, &writing)) != 0) {
      if (!writing) {
        if (p->count)
          got |= POLLIN;
        if (!p->wr_open)
          got |= POLLHUP;
      } else {
        if (p->count < HC_PIPE_BYTES)
          got |= POLLOUT;
        if (!p->rd_open)
          got |= POLLERR;
      }
    } else if (hc_is_stdio(fd)) {
      got = POLLOUT;
    } else if (hc_slot(fd)) {
      got = POLLIN | POLLOUT;
    } else {
      got = POLLNVAL;
    }
    got &= (short)(want | POLLHUP | POLLERR | POLLNVAL);
    fds[i].revents = got;
    if (got)
      ready++;
  }
  if (ready == 0 && (ts == 0 || ts[0] != 0 || ts[1] != 0))
    hc_note_noop(SYS_ppoll); /* it was asked to wait, and there is nothing to wait for */
  return ready;
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

/* Both directions of FILE_READ/FILE_WRITE at an explicit offset, which differ
   only in who fills the data area and in the direction of the copy. Chunked,
   because the payload is one region and a caller may ask for more than fits; a
   short count is a legal POSIX result for both calls, so the loop stops at the
   first short answer rather than spinning. The position is not touched here:
   the wire has carried an offset on every round from the start, so pread and
   pwrite are this function as it is, and read and write are this function
   plus a position update (hc_file_rw). */
static long hc_file_rw_at(struct hc_file *f, char *buf, unsigned long count,
                          int writing, unsigned long long off) {
  const unsigned long data_off = writing ? HC_FILE_WRITE_REQ_V0_DATA_OFFSET
                                         : HC_FILE_READ_REQ_V0_DATA_OFFSET;
  const unsigned long max_chunk = HC_PAYLOAD_SIZE - data_off;
  unsigned long done = 0;

  while (done < count) {
    unsigned long chunk = count - done;
    if (chunk > max_chunk)
      chunk = max_chunk;

    hc_put_u64(0, f->handle);
    hc_put_u64(8, off + done);
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
    if ((unsigned long)serviced < chunk)
      break;
  }
  return (long)done;
}

static long hc_file_rw(long fd, char *buf, unsigned long count, int writing) {
  struct hc_file *f = hc_slot(fd);
  if (!f)
    return -EBADF;
  long n = hc_file_rw_at(f, buf, count, writing, f->pos);
  if (n > 0)
    f->pos += (unsigned long long)n;
  return n;
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

/* THE DOMAIN'S WORKING DIRECTORY. The protocol has no notion of one: the helper
 * resolves every path in its own working directory, and until a domain calls
 * chdir() that is what a relative path means here, unchanged. chdir() gives the
 * domain a directory of its own, kept on this side as a string: from then on
 * every relative path that goes over the wire is written as "<cwd>/<path>"
 * first, so the helper sees a path that no longer depends on where it runs.
 * Nothing is normalised -- "a/../b" goes out as written and the helper's kernel
 * resolves it, which is also what a chdir("..") leaves in the string -- and a
 * first chdir() to a RELATIVE path keeps a relative cwd (still relative to the
 * helper), which musl's getcwd() then refuses with ENOENT because it does not
 * start with "/". getcwd() answers ENOENT before the first chdir() too: there is
 * no directory to name, and inventing one would be a lie some path would later
 * be relative to. chdir() is checked the way opendir() checks a directory: the
 * helper opens the path with O_DIRECTORY and the name is kept only if that
 * succeeds, so a missing or non-directory target answers ENOENT/ENOTDIR and the
 * cwd stays where it was. First consumer is PostgreSQL, whose backend does
 * chdir(DataDir) and names every file relative to it from then on. */
static char hc_cwd[HC_PAYLOAD_SIZE];
static unsigned long hc_cwd_len; /* 0: none set */

/* "<cwd>/<path>" for a relative path under a cwd, path as written otherwise,
   into dst without a terminator; the length, or -ENAMETOOLONG past room. */
static long hc_join(char *dst, unsigned long room, const char *path) {
  unsigned long n = 0;
  if (hc_cwd_len && path[0] != '/') {
    for (unsigned long i = 0; i < hc_cwd_len; i++) {
      if (n >= room) return -ENAMETOOLONG;
      dst[n++] = hc_cwd[i];
    }
    if (hc_cwd[hc_cwd_len - 1] != '/') {
      if (n >= room) return -ENAMETOOLONG;
      dst[n++] = '/';
    }
  }
  for (unsigned long i = 0; path[i]; i++) {
    if (n >= room) return -ENAMETOOLONG;
    dst[n++] = path[i];
  }
  return (long)n;
}

/* Every path that goes over the wire goes through here: joined with the cwd,
   written into the payload at `at`; the length, or -ENAMETOOLONG. */
static long hc_put_path(unsigned long at, const char *path) {
  static char joined[HC_PAYLOAD_SIZE];
  long n = hc_join(joined, HC_PAYLOAD_SIZE - at, path);
  if (n < 0)
    return n;
  for (long i = 0; i < n; i++)
    hc_payload[at + i] = joined[i];
  return n;
}

/* PATH_ACCESS and PATH_DELETE share FILE_OPEN's layout: flags at 0, the path at
 * offset 8, length in bytes. Written once because only the opcode differs. */
static long hc_path_op(unsigned long long opcode, const char *path,
                       unsigned long long flags) {
  hc_put_u64(0, flags);
  long len = hc_put_path(HC_PATH_ACCESS_REQ_V0_PATH_OFFSET, path);
  if (len < 0)
    return len;
  if (hc_round(opcode, HC_PATH_ACCESS_REQ_V0_PATH_OFFSET, (unsigned long)len) != 0)
    return -EIO;
  return hc_metadata->error != 0 ? hc_err() : 0;
}

/* PATH_RENAME: PATH_ACCESS's layout with two paths behind the flags word, as
   "old NUL new", the length covering both and the NUL. musl's rename() arrives as
   renameat2 with flags 0; RENAME_NOREPLACE and friends have no service behind
   them and are refused rather than quietly dropped. */
static long hc_path_rename(const char *old, const char *new, long flags) {
  if (flags != 0)
    return -EINVAL;
  hc_put_u64(0, 0);
  const unsigned long at = HC_PATH_RENAME_REQ_V0_PATH_OFFSET;
  long lo = hc_put_path(at, old);
  if (lo < 0)
    return lo;
  if (at + (unsigned long)lo + 1 >= HC_PAYLOAD_SIZE)
    return -ENAMETOOLONG;
  hc_payload[at + lo] = '\0';
  long ln = hc_put_path(at + (unsigned long)lo + 1, new);
  if (ln < 0)
    return ln;
  if (hc_round(HC_V0_OP_PATH_RENAME, at, (unsigned long)(lo + 1 + ln)) != 0)
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

/* FILE_OPEN. `joined` says whether the path is resolved against the cwd (every
   caller's path is) or sent as written (chdir's, which has been joined already
   and must not be joined twice when the cwd itself is relative). */
static long hc_open_wire(const char *path, long flags, long mode, int joined) {
  int slot = -1;
  for (int i = 0; i < HC_MAX_FILES; i++)
    if (!hc_files[i].used) {
      slot = i;
      break;
    }
  if (slot < 0)
    return -EMFILE;

  hc_put_u64(0, (unsigned long long)flags);
  hc_put_u64(8, (unsigned long long)mode);
  long len;
  if (joined) {
    len = 0;
    while (path[len]) len++;
    if ((unsigned long)len > HC_PAYLOAD_SIZE - HC_FILE_OPEN_REQ_V0_PATH_OFFSET)
      return -ENAMETOOLONG;
    for (long i = 0; i < len; i++)
      hc_payload[HC_FILE_OPEN_REQ_V0_PATH_OFFSET + i] = path[i];
  } else {
    len = hc_put_path(HC_FILE_OPEN_REQ_V0_PATH_OFFSET, path);
    if (len < 0)
      return len;
  }

  if (hc_round(HC_V0_OP_FILE_OPEN, HC_FILE_OPEN_REQ_V0_PATH_OFFSET, (unsigned long)len) != 0)
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

static long hc_open(const char *path, long flags, long mode) {
  return hc_open_wire(path, flags, mode, 0);
}

static long hc_close(long fd);

static long hc_chdir(const char *path) {
  static char next[HC_PAYLOAD_SIZE];
  long n = hc_join(next, HC_PAYLOAD_SIZE - 1, path);
  if (n < 0)
    return n;
  if (n == 0)
    return -ENOENT;
  next[n] = '\0';
  /* The wire has no "is this a directory" question. Opening it as one asks
     the helper's kernel, whose ENOENT or ENOTDIR is the answer chdir owes. */
  long fd = hc_open_wire(next, O_RDONLY | O_DIRECTORY, 0, 1);
  if (fd < 0)
    return fd;
  hc_close(fd);
  for (long i = 0; i < n; i++)
    hc_cwd[i] = next[i];
  hc_cwd_len = (unsigned long)n;
  return 0;
}

/* The kernel's getcwd returns the length INCLUDING the terminator; musl's
   getcwd() then checks for a leading "/" and says ENOENT without one. */
static long hc_getcwd(char *buf, unsigned long size) {
  if (!hc_cwd_len)
    return -ENOENT;
  if (hc_cwd_len + 1 > size)
    return -ERANGE;
  for (unsigned long i = 0; i < hc_cwd_len; i++)
    buf[i] = hc_cwd[i];
  buf[hc_cwd_len] = '\0';
  return (long)hc_cwd_len + 1;
}

static long hc_close(long fd) {
  if (hc_is_stdio(fd)) {
    hc_stdio_closed[fd] = 1;
    return 0;
  }
  if (hc_pipe_exists(fd))
    return hc_pipe_close(fd);
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

/* getdents64 through DIR_READ. The handle is an ordinary FILE_OPEN of the
 * directory (musl's opendir passes O_DIRECTORY, which the helper hands to
 * open), and f->pos is the directory cookie rather than a byte offset: the d_off
 * of the last record returned, 0 at the start. seekdir and rewinddir reach it
 * through lseek(SEEK_SET), which sets exactly that. The records arrive in the
 * layout musl's struct dirent already has (linux_dirent64) and are copied as
 * they are; only whole records are ever returned, so a short buffer gets fewer
 * of them and the cookie picks up at the next. */
static long hc_getdents(long fd, char *buf, unsigned long count) {
  struct hc_file *f = hc_slot(fd);
  if (!f)
    return -EBADF;
  const unsigned long data_off = HC_DIR_READ_REQ_V0_DATA_OFFSET;
  if (count > HC_PAYLOAD_SIZE - data_off)
    count = HC_PAYLOAD_SIZE - data_off;

  hc_put_u64(0, f->handle);
  hc_put_u64(8, f->pos);
  if (hc_round(HC_V0_OP_DIR_READ, data_off, count) != 0)
    return -EIO;
  if (hc_metadata->error != 0)
    return hc_err();
  long n = (long)hc_metadata->result;
  if (n < 0 || (unsigned long)n > count)
    return -EIO;

  unsigned long long last_off = f->pos;
  for (long off = 0; off < n;) {
    const unsigned char *rec = (const unsigned char *)&hc_payload[data_off + off];
    unsigned reclen = rec[16] | (unsigned)rec[17] << 8;
    if (reclen < 19 || off + (long)reclen > n)
      return -EIO; /* a record the helper cannot have produced */
    unsigned long long d_off = 0;
    for (int i = 7; i >= 0; i--)
      d_off = d_off << 8 | rec[8 + i];
    last_off = d_off;
    off += reclen;
  }
  for (long i = 0; i < n; i++)
    buf[i] = hc_payload[data_off + i];
  f->pos = last_off;
  return n;
}

static long hc_lseek(long fd, long long off, long whence) {
  if (hc_is_stdio(fd))
    return -ESPIPE; /* a stream, as on a pipe or a terminal */
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

/* read(): a pipe end answers from its queue, everything else is a file. */
static long hc_read(long fd, char *buf, unsigned long count) {
  int writing;
  struct hc_pipe *p = hc_pipe_end(fd, &writing);
  if (p)
    return writing ? -EBADF : hc_pipe_read(p, buf, count);
  return hc_file_rw(fd, buf, count, 0);
}

static long hc_readv(long fd, const struct iovec *iov, long count) {
  unsigned long done = 0;
  for (long i = 0; i < count; i++) {
    if (iov[i].iov_len == 0)
      continue;
    long n = hc_read(fd, (char *)iov[i].iov_base, iov[i].iov_len);
    if (n < 0)
      return done ? (long)done : n;
    done += (unsigned long)n;
    if ((unsigned long)n < iov[i].iov_len)
      break; /* short read, or end of file */
  }
  return (long)done;
}

/* pread, pwrite and their vector forms: the offset comes from the caller and
 * the position stays where it was. This is most of what a database does with a
 * file (PostgreSQL reads and writes every page with pread64/pwrite64 and its
 * WAL with pwritev), and it is the wire's own shape: every FILE_READ/FILE_WRITE
 * round has carried an explicit offset from the start, so nothing crosses the
 * boundary differently, only the position bookkeeping on this side differs.
 * On stdout and stderr the answer is ESPIPE, as on a pipe.
 *
 * musl hands the vector forms' offset over as two words (src/unistd/preadv.c:
 * (long)ofs and (long)(ofs >> 32)), because the kernel ABI they were written
 * for takes pos_l and pos_h; the kernel joins them as (hi << 32) | (u32)lo, and
 * so does hc_join_offset. */
static long hc_prw(long fd, char *buf, unsigned long count, int writing,
                   unsigned long long off) {
  if (hc_is_stdio(fd))
    return -ESPIPE;
  struct hc_file *f = hc_slot(fd);
  if (!f)
    return -EBADF;
  return hc_file_rw_at(f, buf, count, writing, off);
}

static long hc_prwv(long fd, const struct iovec *iov, long count, int writing,
                    unsigned long long off) {
  if (hc_is_stdio(fd))
    return -ESPIPE;
  struct hc_file *f = hc_slot(fd);
  if (!f)
    return -EBADF;
  unsigned long done = 0;
  for (long i = 0; i < count; i++) {
    if (iov[i].iov_len == 0)
      continue;
    long n = hc_file_rw_at(f, (char *)iov[i].iov_base, iov[i].iov_len, writing,
                           off + done);
    if (n < 0)
      return done ? (long)done : n;
    done += (unsigned long)n;
    if ((unsigned long)n < iov[i].iov_len)
      break; /* short count, or end of file: do not start the next entry */
  }
  return (long)done;
}

static unsigned long long hc_join_offset(long lo, long hi) {
  return (unsigned long long)(unsigned int)lo | ((unsigned long long)hi << 32);
}

/* The WRITE_STDOUT rounds themselves, with no descriptor check: the program's
   own writes reach this through hc_write, and the runtime's own report at exit
   calls it directly (hc_report_unserved). */
static long hc_stdout_bytes(const char *buf, unsigned long count) {
  unsigned long done = 0;
  if (!hc_payload)
    return -EIO; /* no region shared yet: there is nowhere to put the bytes */
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

static long hc_write(long fd, const char *buf, unsigned long count) {
  int writing;
  struct hc_pipe *p = hc_pipe_end(fd, &writing);
  if (p)
    return writing ? hc_pipe_write(p, buf, count) : -EBADF;
  /* Anything above stderr is a handle this domain opened, and goes through
     FILE_WRITE with its own position. */
  if (fd >= HC_FD_BASE)
    return hc_file_rw(fd, (char *)buf, count, 1);
  if (!hc_is_stdio(fd))
    return -EBADF;
  return hc_stdout_bytes(buf, count);
}

long __capstone_hostcall(long n, syscall_arg_t a, syscall_arg_t b,
                         syscall_arg_t c, syscall_arg_t d, syscall_arg_t e,
                         syscall_arg_t f) {
  (void)f;
  switch (n) {
  case SYS_write:
    return hc_write((long)a, (const char *)b, (unsigned long)c);

  /* musl's open() issues openat(AT_FDCWD, ...). The dirfd is accepted and not
     used: there is no *at family behind this protocol to be relative to. A
     relative path is relative to the domain's own working directory once
     chdir() has set one (hc_put_path joins it), and to the helper's before
     that, which is stated here rather than silently assumed. */
  case SYS_openat:
    return hc_open((const char *)b, (long)c, (long)d);

  case SYS_chdir:
    return hc_chdir((const char *)a);

  case SYS_getcwd:
    return hc_getcwd((char *)a, (unsigned long)b);

  case SYS_read:
    return hc_read((long)a, (char *)b, (unsigned long)c);

  case SYS_close:
    return hc_close((long)a);

  case SYS_lseek:
    if (hc_pipe_exists((long)a))
      return -ESPIPE;
    return hc_lseek((long)a, (long long)b, (long)c);

  /* musl's pipe() is pipe2(fds, 0) here. */
  case SYS_pipe2:
    return hc_pipe2((int *)a, (long)b);

  /* musl's poll() is ppoll(fds, n, timeout ? &ts : 0, 0, _NSIG/8). */
  case SYS_ppoll:
    return hc_poll((struct pollfd *)a, (unsigned long)b, (const long *)c);

  case SYS_writev:
    return hc_writev((long)a, (const struct iovec *)b, (long)c);

  case SYS_readv:
    return hc_readv((long)a, (const struct iovec *)b, (long)c);

  case SYS_pread64:
    return hc_prw((long)a, (char *)b, (unsigned long)c, 0, (unsigned long long)(long)d);

  case SYS_pwrite64:
    return hc_prw((long)a, (char *)b, (unsigned long)c, 1, (unsigned long long)(long)d);

  case SYS_preadv:
    return hc_prwv((long)a, (const struct iovec *)b, (long)c, 0, hc_join_offset((long)d, (long)e));

  case SYS_pwritev:
    return hc_prwv((long)a, (const struct iovec *)b, (long)c, 1, hc_join_offset((long)d, (long)e));

  case SYS_getdents64:
    return hc_getdents((long)a, (char *)b, (unsigned long)c);

  /* stdio asks whether stdout is a terminal, to choose line buffering over full
     buffering. ENOTTY is the true answer for a domain and the one musl handles:
     it picks full buffering. Returning ENOSYS would work by accident; returning
     the right error means the next reader does not have to wonder.
     Full buffering is only safe if someone flushes: musl switches stdout to full
     buffering on its FIRST flush (__stdout_write.c). exit() flushes, and since
     domain_main ends a returning program with exit() too, so does returning. */
  case SYS_ioctl:
    if (!hc_is_stdio((long)a) && !hc_slot((long)a) && !hc_pipe_exists((long)a))
      return -EBADF;
    return -ENOTTY;

  case SYS_fsync:
  case SYS_fdatasync:
    return hc_handle_op((long)a, HC_V0_OP_FILE_SYNC, 0);

  case SYS_ftruncate:
    return hc_handle_op((long)a, HC_V0_OP_FILE_TRUNCATE, (unsigned long long)b);

  /* musl's unlink() and access() both go through the *at forms. The dirfd is
     accepted and unused for the same reason it is in openat; the path is
     joined with the domain's cwd the same way. */
  case SYS_unlinkat:
    return hc_path_op(HC_V0_OP_PATH_DELETE, (const char *)b,
                      HC_PATH_DELETE_FLAG_NONE);

  case SYS_faccessat:
    return hc_path_op(HC_V0_OP_PATH_ACCESS, (const char *)b,
                      HC_PATH_ACCESS_FLAG_EXISTS);

  /* musl's rename() is renameat2(AT_FDCWD, old, AT_FDCWD, new, 0) here: this
     target has neither rename nor renameat. The directory descriptors are
     accepted and not used, as openat's is. */
  case SYS_renameat2:
    return hc_path_rename((const char *)b, (const char *)d, (long)e);

  case SYS_fstat: {
    /* stdout and stderr exist in every domain -- the service writes them --
       so fstat describes them: a character device, and not a terminal (the
       tty ioctl answers ENOTTY). Answering EBADF made programs that check a
       descriptor before using it conclude they have no output: CPython sets
       sys.stdout and sys.stderr to None that way, and print() then writes
       nothing, silently. stdin has no service here and stays EBADF. */
    if (hc_is_stdio((long)a)) {
      hc_fill_stat((struct stat *)b, 0, S_IFCHR | 0620);
      return 0;
    }
    {
      int writing;
      struct hc_pipe *p = hc_pipe_end((long)a, &writing);
      if (p) { /* a fifo, with what is queued as its size */
        hc_fill_stat((struct stat *)b, p->count, S_IFIFO | 0600);
        return 0;
      }
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
    if (hc_pipe_exists((long)a)) {
      /* A pipe here never blocks (see hc_pipes), so F_GETFL says O_NONBLOCK
         whatever was set; the other commands are accepted as for a file. */
      if ((long)b == F_GETFL)
        return O_NONBLOCK;
      return 0;
    }
    if (hc_is_stdio((long)a) || hc_slot((long)a))
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

  /* A domain is one process: 1 is its pid, as 1 is its tid above, and 0 its
     parent, as init's is. Not invented the way a st_dev would be: nothing
     outside the domain consumes the number, callers write it into lock files
     and log lines and compare it with their own. PostgreSQL puts getpid() in
     postmaster.pid and probes a stale one with kill(pid, 0), which is unserved
     and so says "no such process", which is the right answer. */
  case SYS_getpid:
    return 1;

  case SYS_getppid:
    return 0;

  case SYS_umask: {
    long old = hc_umask;
    hc_umask = (long)a & 0777;
    return old;
  }

  /* setitimer and getitimer: ACCEPTED, AND THE TIMER NEVER FIRES. A domain gets
     no signals (rt_sigaction is unserved), so a running timer could not deliver
     SIGALRM anyway; refusing setitimer is fatal to PostgreSQL (timeout.c: "could
     not enable SIGALRM timer"), whose timeouts never matter in single-user mode.
     Serving it as a no-op is the honest middle: the call succeeds, the timer
     reads as unarmed, and the exit report lists it under NO-OP so a success is
     never mistaken for a timer. A program that waits for the alarm waits for
     ever; the report is what says why. The struct is two timevals, four longs
     on this target, as musl's wrappers pass it. */
  case SYS_setitimer:
    if ((long)a < 0 || (long)a > 2)
      return -EINVAL;
    if (c) {
      long *old = (long *)c;
      old[0] = old[1] = old[2] = old[3] = 0;
    }
    hc_note_noop(n);
    return 0;

  case SYS_getitimer:
    if ((long)a < 0 || (long)a > 2)
      return -EINVAL;
    if (b) {
      long *cur = (long *)b;
      cur[0] = cur[1] = cur[2] = cur[3] = 0;
    }
    hc_note_noop(n);
    return 0;

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
 * Straight through the hostcall, not printf and not write(2): this runs after
 * the program is finished, and on the exit() path musl's stdio has already been
 * torn down. Not write(2) either, because that goes through the program's
 * descriptors: a program that closed fd 1 (a full tshark run does) had the report
 * refused with EBADF, and a missing line reads as "nothing unserved" (ISSUES I-11).
 *
 * Numbers, not names: a table of three hundred names is not worth the bytes in
 * every domain image when both readers already translate. The suite's runner
 * prints them by name, and check-domain-support.py says which CALL needs each
 * one, which is the question a person actually has.
 */
static void hc_report_list(const char *head, const long *list,
                           unsigned long total, unsigned long max) {
  if (total == 0)
    return;
  char buf[256];
  unsigned long p = 0;
  for (unsigned long i = 0; head[i]; i++)
    buf[p++] = head[i];
  unsigned long shown = total < max ? total : max;
  for (unsigned long i = 0; i < shown && p + 32 < sizeof buf; i++) {
    unsigned long seen = 0, times = 0;
    for (unsigned long j = 0; j < shown; j++) {
      if (list[j] != list[i])
        continue;
      if (j < i)
        seen = 1;
      times++;
    }
    if (seen)   /* one entry per distinct number, with how often it was asked */
      continue;
    buf[p++] = ' ';
    long v = list[i];
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
  if (total > shown && p + 8 < sizeof buf) {
    static const char more[] = " ...";
    for (unsigned long i = 0; i < sizeof more - 1; i++)
      buf[p++] = more[i];
  }
  buf[p++] = '\n';
  hc_stdout_bytes(buf, p);
}

static void hc_report_unserved(void) {
  hc_report_list("capstone-domain: UNSERVED syscalls:", hc_unserved, hc_unserved_n,
                 HC_UNSERVED_MAX);
  hc_report_list("capstone-domain: NO-OP syscalls:", hc_noop, hc_noop_n, HC_NOOP_MAX);
}

unsigned long __capstone_unserved_count(void) { return hc_unserved_n; }
long __capstone_unserved_at(unsigned long i) {
  return i < HC_UNSERVED_MAX && i < hc_unserved_n ? hc_unserved[i] : -1;
}

/* Constructors and destructors: .init_array and .fini_array (ISSUES C-64).
 *
 * Nothing else in a domain runs them. start-musl.S runs only .capstone_cap_init,
 * and musl's __libc_start_main, which would, is not used. musl's exit() walks
 * .fini_array through uintptr_t,
 *
 *     uintptr_t a = (uintptr_t)&__fini_array_end;
 *     for (; a > (uintptr_t)&__fini_array_start; a -= sizeof(void(*)()))
 *         (*(void (**)())(a - sizeof(void(*)())))();
 *
 * and loads each slot through an integer address: cause 24 on the first one.
 * Both stayed invisible while no domain had either array. The tshark port was
 * the first, with GLib's and libgpg-error's constructors and libxml2's
 * destructor.
 *
 * THE SLOTS ARE NOT CAPABILITIES. A static domain image carries no relocations,
 * and the capability initialisers do not cover these arrays, so each 16-byte slot
 * holds the function's LINK address as a plain integer in its low 8 bytes, and
 * the domain runs at another base. The callable capability is derived from the
 * code capability of a function in this file, the anchor, moved by the distance
 * between the two link addresses. The anchor's own link address is written into
 * .rodata by the assembler (`.quad`), which the static link resolves exactly as
 * it resolves the slots. A slot that does hold a tagged capability is called as
 * it is.
 *
 * The array markers are defined by my_first_domain/link.ld, the script every musl
 * domain links with, so their addresses are real ones (an undefined weak symbol's
 * would not be: C-56). __libc_exit_fini replaces musl's weak alias of the same
 * name (src/exit/exit.c); musl's version also calls _fini(), which in a domain is
 * its empty default. */
extern const unsigned char __init_array_start[], __init_array_end[];
extern const unsigned char __fini_array_start[], __fini_array_end[];

void __capstone_init_fini_anchor(void);
void __capstone_init_fini_anchor(void) {}

extern const unsigned long __capstone_init_fini_anchor_link;
__asm__(".section .rodata\n"
        ".p2align 3\n"
        ".globl __capstone_init_fini_anchor_link\n"
        "__capstone_init_fini_anchor_link:\n"
        ".quad __capstone_init_fini_anchor\n"
        ".previous\n");

typedef void (*hc_array_fn)(void);

static void hc_call_array_slot(const unsigned char *slot) {
  hc_array_fn f = *(const hc_array_fn *)slot;
  if (!__builtin_capstone_cap_get_tag(f)) {
    unsigned long link = *(const unsigned long *)slot;
    f = (hc_array_fn)((const char *)__capstone_init_fini_anchor +
                      (long)(link - __capstone_init_fini_anchor_link));
  }
  f();
}

/* The environment the program starts with. In C, environ exists before any
 * constructor runs, and a constructor may read it: GLib's reads G_DEBUG and
 * G_MESSAGES_PREFIXED. A domain has no environment of its own, so its entry says
 * what it is, and domain_main sets it before the constructors run. DEFINED weak,
 * not declared (C-56): the default is an empty environment, and an entry whose
 * program has one defines this. An entry may still set __environ itself in
 * capstone_main, as every entry did before this existed. */
extern char **__environ;
__attribute__((__weak__)) char **__capstone_domain_environ(void) {
  static char *none[] = { 0 };
  return none;
}

static void hc_run_init_array(void) {
  for (const unsigned char *p = __init_array_start; p < __init_array_end;
       p += sizeof(hc_array_fn))
    hc_call_array_slot(p);
}

void __libc_exit_fini(void) {
  for (const unsigned char *p = __fini_array_end; p > __fini_array_start;
       p -= sizeof(hc_array_fn))
    hc_call_array_slot(p - sizeof(hc_array_fn));
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
#ifdef CAPSTONE_PROGRAM_REGIONS
    else if (hc_shared_region_count - 2 < HC_PROGRAM_REGIONS)
      hc_program_region[hc_shared_region_count - 2] = res;
#endif
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
  if (jumped) {
    status = hc_exit_status;
  } else {
    /* The environment, then the constructors, then main, as in C; after the
       exit jump is armed, since a constructor may call exit() (C-64). */
    __environ = __capstone_domain_environ();
    hc_run_init_array();
    /* A program that returns ends as returning from main ends in C: through
       exit(), so its atexit handlers run and stdio is flushed, and the exit
       syscall brings the status back here through the jump above. Returning
       straight to the host skipped both: musl buffers stdout fully after its
       first flush, so exactly the first line of a returning program reached the
       host and the rest was lost (FFmpeg and CPython each worked around it). */
    exit(capstone_main());
  }
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
