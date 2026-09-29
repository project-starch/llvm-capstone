/* The delegated syscall stub: musl's syscall entry under CAPSTONE_DELEGATE_RUNTIME.
 *
 * Every Linux call becomes one entry in the entry region and one yield. Pointer
 * arguments are copied into the exchange region and cross as offsets; nothing
 * that names domain memory ever leaves the domain. The launcher runs the real
 * syscall and writes the result back. The three exception groups never cross:
 * memory is the domain allocator's, processes are not this branch's, signals
 * are a table here and a mask there. See docs/plans/delegation-abi.md.
 */
#include "capstone/delegate.h"
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/uio.h>

typedef void *syscall_arg_t;
/* Integers travel in pointer-typed argument slots, as musl's syscall_arch.h
   states; the casts are the port's convention, not an accident. */
#pragma clang diagnostic ignored "-Wint-to-void-pointer-cast"
#pragma clang diagnostic ignored "-Wvoid-pointer-to-int-cast"
extern void __capstone_yield(void);
extern int __capstone_at_exit(int status);
void __capstone_hc_note_unserved(long n);
void __capstone_hc_note_noop(long n);
void __capstone_hc_report_unserved(void);

static volatile struct capstone_delegate_entry *dl_entry;
static char *dl_exchange;
static size_t dl_capacity, dl_used;

/* Argument marshalling record for one call. */
struct dl_slot {
  void *domain;      /* the caller's buffer, or NULL */
  uint64_t offset;   /* where it went in the exchange region */
  size_t bytes;
  unsigned char copy_back;
};

static size_t cap_bytes(void *cap) {
  return (size_t)(__builtin_capstone_cap_get_end(cap) -
                  __builtin_capstone_cap_get_cursor(cap));
}

void __capstone_delegate_regions(void *entry, void *exchange) {
  dl_entry = entry;
  dl_exchange = exchange;
  dl_capacity = exchange ? cap_bytes(exchange) : 0;
  if (dl_capacity > 0x40000000u)
    dl_capacity = 0x40000000u;
}

int __capstone_delegate_ready(void) {
  return dl_entry && dl_exchange && cap_bytes((void *)dl_entry) >= sizeof *dl_entry;
}

static int dl_alloc(size_t bytes, uint64_t *offset) {
  size_t aligned = (dl_used + 15) & ~(size_t)15;
  if (aligned > dl_capacity || bytes > dl_capacity - aligned)
    return -1;
  *offset = aligned;
  dl_used = aligned + bytes;
  return 0;
}

/* One round trip. args are already offsets where the shape says so. */
static long dl_round(uint64_t nr, const uint64_t args[CAPSTONE_DELEGATE_ARGS]) {
  struct capstone_delegate_entry e;
  if (capstone_delegate_pack(&e, nr, args))
    return -EINVAL;
  memcpy((void *)dl_entry, &e, sizeof e);
  __capstone_yield();
  return (long)dl_entry->result;
}

static long dl_string(const char *s, uint64_t *offset) {
  size_t n;
  if (!s)
    return -EFAULT;
  n = strlen(s) + 1;
  if (dl_alloc(n, offset))
    return -ENAMETOOLONG;
  memcpy(dl_exchange + *offset, s, n);
  return 0;
}

/* Buffers whose length is another argument are clamped to what fits: a short
 * read, write or listing is a result every caller already handles. */
static long dl_call(uint64_t nr, syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS]) {
  const struct capstone_delegate_shape *s = capstone_delegate_shape(nr);
  uint64_t args[CAPSTONE_DELEGATE_ARGS];
  struct dl_slot slots[CAPSTONE_DELEGATE_ARGS] = {{0}};
  long result;
  if (!s)
    return -ENOSYS;
  dl_used = 0;
  for (unsigned i = 0; i < CAPSTONE_DELEGATE_ARGS; ++i)
    args[i] = (uint64_t)(unsigned long)raw[i];
  for (unsigned i = 0; i < s->argc; ++i) {
    const struct capstone_delegate_arg *a = &s->args[i];
    size_t bytes;
    int optional = a->kind == CAPSTONE_ARG_OPT_IN || a->kind == CAPSTONE_ARG_OPT_OUT ||
                   a->kind == CAPSTONE_ARG_OPT_INOUT;
    if (a->kind == CAPSTONE_ARG_INT)
      continue;
    if (a->kind == CAPSTONE_ARG_STR) {
      long rc = dl_string((const char *)raw[i], &args[i]);
      if (rc)
        return rc;
      continue;
    }
    if (!raw[i]) {
      if (optional) {
        args[i] = 0;
        continue;
      }
      return -EFAULT;
    }
    if (a->length == CAPSTONE_LEN_ARG && a->size < CAPSTONE_DELEGATE_ARGS &&
        args[a->size] > dl_capacity - ((dl_used + 15) & ~(size_t)15))
      args[a->size] = dl_capacity - ((dl_used + 15) & ~(size_t)15);
    {
      struct capstone_delegate_entry probe = {0};
      memcpy(probe.args, args, sizeof args);
      bytes = capstone_delegate_arg_bytes(s, &probe, i);
    }
    if (dl_alloc(bytes, &args[i]))
      return -ENOMEM;
    if (a->kind != CAPSTONE_ARG_OUT && a->kind != CAPSTONE_ARG_OPT_OUT)
      memcpy(dl_exchange + args[i], raw[i], bytes);
    slots[i].domain = raw[i];
    slots[i].offset = args[i];
    slots[i].bytes = bytes;
    slots[i].copy_back = a->kind != CAPSTONE_ARG_IN && a->kind != CAPSTONE_ARG_OPT_IN;
  }
  result = dl_round(nr, args);
  for (unsigned i = 0; i < CAPSTONE_DELEGATE_ARGS; ++i)
    if (slots[i].domain && slots[i].copy_back)
      memcpy(slots[i].domain, dl_exchange + slots[i].offset, slots[i].bytes);
  return result;
}

/* readv and writev, and their positioned forms, go element by element: the
 * exchange region is one buffer and iovec atomicity is not promised here. */
static long dl_vector(long fd, const struct iovec *iov, long count, int writing,
                      int positioned, long long offset) {
  long total = 0;
  for (long i = 0; i < count; ++i) {
    syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS] = {0};
    long n;
    if (!iov[i].iov_len)
      continue;
    raw[0] = (syscall_arg_t)fd;
    raw[1] = iov[i].iov_base;
    raw[2] = (syscall_arg_t)iov[i].iov_len;
    if (positioned)
      raw[3] = (syscall_arg_t)(unsigned long)(offset + total);
    n = dl_call(positioned ? (writing ? CAPSTONE_SYS_pwrite64 : CAPSTONE_SYS_pread64)
                           : (writing ? CAPSTONE_SYS_write : CAPSTONE_SYS_read), raw);
    if (n < 0)
      return total ? total : n;
    total += n;
    if ((size_t)n < iov[i].iov_len)
      break;
  }
  return total;
}

/* ioctl and fcntl carry a request-specific third argument. The ones a libc
 * uses are listed; anything else passes 0 and the kernel says EFAULT or
 * EINVAL, which is visible rather than a silent domain address. */
static long dl_ioctl(long fd, unsigned long request, void *argp) {
  syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS] = {(syscall_arg_t)fd, (syscall_arg_t)request, 0, 0, 0, 0};
  size_t bytes = 0;
  switch (request) {
  case TIOCGWINSZ: case TIOCSWINSZ: bytes = 8; break;
  case TCGETS: case TCSETS: case TCSETSW: case TCSETSF: bytes = 60; break;
  case FIONREAD: case FIONBIO: case FIOCLEX: case FIONCLEX: bytes = request == FIOCLEX || request == FIONCLEX ? 0 : 4; break;
  default: bytes = 0;
  }
  if (bytes && argp) {
    unsigned char buffer[64];
    long rc;
    memcpy(buffer, argp, bytes);
    raw[2] = buffer;
    /* the shape's fixed 64 bytes are what the wire reserves */
    dl_used = 0;
    {
      uint64_t args[CAPSTONE_DELEGATE_ARGS] = {(uint64_t)fd, request, 0, 0, 0, 0};
      if (dl_alloc(64, &args[2]))
        return -ENOMEM;
      memcpy(dl_exchange + args[2], buffer, bytes);
      rc = dl_round(CAPSTONE_SYS_ioctl, args);
      if (rc >= 0)
        memcpy(argp, dl_exchange + args[2], bytes);
      return rc;
    }
  }
  return dl_call(CAPSTONE_SYS_ioctl, raw);
}

static long dl_fcntl(long fd, long cmd, void *arg) {
  if (cmd == F_GETLK || cmd == F_SETLK || cmd == F_SETLKW) {
    uint64_t args[CAPSTONE_DELEGATE_ARGS] = {(uint64_t)fd, (uint64_t)cmd, 0, 0, 0, 0};
    long rc;
    if (!arg)
      return -EFAULT;
    dl_used = 0;
    if (dl_alloc(32, &args[2]))
      return -ENOMEM;
    memcpy(dl_exchange + args[2], arg, 32);
    rc = dl_round(CAPSTONE_SYS_fcntl, args);
    if (rc >= 0 && cmd == F_GETLK)
      memcpy(arg, dl_exchange + args[2], 32);
    return rc;
  }
  {
    /* the integer commands carry an integer in a pointer slot; it is passed
       as the number it is, and the wire shape leaves it unflagged */
    syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS] = {(syscall_arg_t)fd, (syscall_arg_t)cmd, arg, 0, 0, 0};
    return dl_call(CAPSTONE_SYS_fcntl, raw);
  }
}

long __capstone_delegate_hello(unsigned long entry_address, unsigned long code_base,
                               unsigned long code_end) {
  uint64_t args[CAPSTONE_DELEGATE_ARGS] = {entry_address, code_base, code_end, 0, 0, 0};
  return dl_round(CAPSTONE_NR_HELLO, args);
}

long __capstone_delegate_call(long n, syscall_arg_t a, syscall_arg_t b,
                              syscall_arg_t c, syscall_arg_t d,
                              syscall_arg_t e, syscall_arg_t f) {
  syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS] = {a, b, c, d, e, f};
  if (!__capstone_delegate_ready())
    return -EIO;
  switch (n) {
  case SYS_readv:
    return dl_vector((long)a, (const struct iovec *)b, (long)c, 0, 0, 0);
  case SYS_writev:
    return dl_vector((long)a, (const struct iovec *)b, (long)c, 1, 0, 0);
  case SYS_preadv:
    return dl_vector((long)a, (const struct iovec *)b, (long)c, 0, 1,
                     (long long)((unsigned long)d | ((unsigned long)e << 32)));
  case SYS_pwritev:
    return dl_vector((long)a, (const struct iovec *)b, (long)c, 1, 1,
                     (long long)((unsigned long)d | ((unsigned long)e << 32)));
  case SYS_ioctl:
    return dl_ioctl((long)a, (unsigned long)b, c);
  case SYS_fcntl:
    return dl_fcntl((long)a, (long)b, c);
  case SYS_exit:
  case SYS_exit_group: {
    /* The program's last words before the task ends it: the at-exit hook,
       then the unserved report, then the real exit_group. Nothing resumes. */
    long status = __capstone_at_exit((int)(long)a);
    __capstone_hc_report_unserved();
    raw[0] = (syscall_arg_t)status;
    return dl_call(CAPSTONE_SYS_exit_group, raw);
  }
  default:
    break;
  }
  switch (capstone_delegate_group_of((uint64_t)n)) {
  case CAPSTONE_GROUP_DELEGATED:
    return dl_call((uint64_t)n, raw);
  case CAPSTONE_GROUP_SIGNAL:
    /* Accepted and not delivered until the signals branch: recorded as a
       no-op so "served" never quietly means "pretended". */
    if (n == SYS_rt_sigaction || n == SYS_rt_sigprocmask) {
      __capstone_hc_note_noop(n);
      return 0;
    }
    __capstone_hc_note_unserved(n);
    return -ENOSYS;
  case CAPSTONE_GROUP_MEMORY:
  case CAPSTONE_GROUP_PROCESS:
  case CAPSTONE_GROUP_RUNTIME:
  case CAPSTONE_GROUP_UNKNOWN:
  default:
    __capstone_hc_note_unserved(n);
    return -ENOSYS;
  }
}

/* The unserved report's writer: two is the task's stderr. */
void __capstone_delegate_write2(const char *buf, unsigned long n) {
  syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS] = {(syscall_arg_t)2, (syscall_arg_t)buf,
                                               (syscall_arg_t)n, 0, 0, 0};
  while (n) {
    long k = dl_call(CAPSTONE_SYS_write, raw);
    if (k <= 0)
      return;
    buf += k;
    n -= (unsigned long)k;
    raw[1] = (syscall_arg_t)buf;
    raw[2] = (syscall_arg_t)n;
  }
}
