/* The delegated syscall stub: musl's syscall entry in every application.
 *
 * Every Linux call becomes one entry in the entry region and one yield. Pointer
 * arguments are copied into the exchange region and cross as offsets; nothing
 * that names domain memory ever leaves the domain. The launcher runs the real
 * syscall and writes the result back. The three exception groups never cross:
 * memory is the domain allocator's, processes are not this branch's, signals
 * are a table here and a mask there. See docs/plans/delegation-abi.md.
 */
#include "capstone/delegate.h"
#include "capstone/spawn.h"
#include "capstone/msghdr.h"
#include <sys/epoll.h>
#include <sys/socket.h>
#include "capstone/launch.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <signal.h>
#include <time.h>

typedef void *syscall_arg_t;
/* Integers travel in pointer-typed argument slots, as musl's syscall_arch.h
   states; the casts are the port's convention, not an accident. */
#pragma clang diagnostic ignored "-Wint-to-void-pointer-cast"
#pragma clang diagnostic ignored "-Wvoid-pointer-to-int-cast"
#pragma clang diagnostic ignored "-Wcapstone-pointer-roundtrip"
extern void __capstone_yield(void);
extern int __capstone_at_exit(int status);
void __capstone_hc_note_unserved(long n);
void __capstone_hc_note_noop(long n);
void __capstone_hc_report_unserved(void);
long __capstone_delegate_spawn(const void *block, unsigned long bytes);
const struct capstone_launch_task *__capstone_launch_task(void);
/* signals.c */
struct k_sigaction;
void __capstone_signals_regions(void *meta, unsigned long bytes);
int __capstone_signals_hint(void);
void __capstone_signals_take(void);
void __capstone_signals_deliver(void);
void __capstone_sigmask_note(int how, uint64_t m);
long __capstone_sigaction(int sig, const struct k_sigaction *new, struct k_sigaction *old);
long __capstone_sigprocmask(int how, const sigset_t *set, sigset_t *old, unsigned long size);
long __capstone_sigaltstack(const stack_t *ss, stack_t *old);
void __capstone_siginfo_translate(const unsigned char *raw, int sig, siginfo_t *si);
static int dl_buffer_ok(const void *buffer, size_t bytes, unsigned rights);

/* Answered without a round, from the launch record (see launch.h): the task's
 * identity, which cannot change under a domain, and the two clocks as rdtime
 * since the launcher read them, the way a vDSO answers. Neither invents state:
 * the values are Linux's, read once by the task itself. Every other clock, and
 * everything when the record is missing, is a round. */
static int dl_clock(long clock, struct timespec *ts) {
  const struct capstone_launch_task *t = __capstone_launch_task();
  uint64_t base, now, elapsed, ns;
  if (!t || !t->ticks_per_second || !ts)
    return 0;
  if (!dl_buffer_ok(ts, sizeof *ts, 2))
    return 0;  /* the shaped syscall path returns EFAULT */
  switch (clock) {
  case CLOCK_REALTIME: case CLOCK_REALTIME_COARSE:
    base = t->realtime_ns;
    break;
  case CLOCK_MONOTONIC: case CLOCK_MONOTONIC_COARSE: case CLOCK_MONOTONIC_RAW:
  case CLOCK_BOOTTIME:
    base = t->monotonic_ns;
    break;
  default:
    return 0;
  }
  __asm__ volatile("rdtime %0" : "=r"(now));
  elapsed = now - t->ticks;
  ns = base + elapsed / t->ticks_per_second * 1000000000u +
       elapsed % t->ticks_per_second * 1000000000u / t->ticks_per_second;
  ts->tv_sec = (time_t)(ns / 1000000000u);
  ts->tv_nsec = (long)(ns % 1000000000u);
  return 1;
}

static long dl_identity(long n, long *answer) {
  const struct capstone_launch_task *t = __capstone_launch_task();
  if (!t || !t->pid)
    return 0;
  switch (n) {
  case SYS_getpid: case SYS_gettid: *answer = t->pid; return 1;
  case SYS_getppid: *answer = t->ppid; return 1;
  case SYS_getuid: *answer = t->uid; return 1;
  case SYS_geteuid: *answer = t->euid; return 1;
  case SYS_getgid: *answer = t->gid; return 1;
  case SYS_getegid: *answer = t->egid; return 1;
  default: return 0;
  }
}

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

/* Keep the checked pointer and its copy-back coordinates together. Allocate
 * only as many records as the validated vector needs, including for messages;
 * a one-element call must not reserve 1024 capability slots on the stack. */
struct dl_iov_slot {
  void *base;
  uint64_t offset;
  size_t bytes;
};

static size_t cap_bytes(void *cap) {
  return (size_t)(__builtin_capstone_cap_get_end(cap) -
                  __builtin_capstone_cap_get_cursor(cap));
}

/* LCC's type selector is total even on an untagged value. The tag builtin
 * collapses all capability types to one bit, so it cannot reject a REV or a
 * sealed handle before the permission and bounds selectors are used. READ is
 * bit 2, WRITE bit 1 in the Capstone permission field. This checks the
 * requested span; the actual copy still uses the original capability and
 * checks liveness at each access. */
static int dl_buffer_ok(const void *buffer, size_t bytes, unsigned rights) {
  unsigned long type, base, cursor, end, perms;
  if (!bytes)
    return 1;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x04, %0, %1, x1"
                   : "=r"(type) : "r"(buffer));
  if (type != 0 && type != 1)  /* LINEAR or NONLINEAR data */
    return 0;
  base = __builtin_capstone_cap_get_base((void *)buffer);
  end = __builtin_capstone_cap_get_end((void *)buffer);
  cursor = __builtin_capstone_cap_get_cursor((void *)buffer);
  perms = __builtin_capstone_cap_get_perm((void *)buffer);
  return (perms & rights) == rights && base <= cursor && cursor <= end &&
         bytes <= end - cursor;
}

static unsigned dl_rights(unsigned kind) {
  switch (kind) {
  case CAPSTONE_ARG_IN: case CAPSTONE_ARG_OPT_IN: return 4;
  case CAPSTONE_ARG_OUT: case CAPSTONE_ARG_OPT_OUT: return 2;
  default: return 6;  /* INOUT */
  }
}

static uint64_t dl_status;  /* of the last round: DONE or RETRY */

/* The virtual adapter keeps one META/exchange pair for the process. Linux
 * threads share it, so serialize a complete wire round, including signal
 * delivery. Signal handlers can make a nested delegated call, so this is a
 * small recursive lock. The stack capability is the worker identity: virtual
 * threads have distinct stacks, while nested calls on one stack retain the
 * same bounded base. The QEMU supervisor still keeps register state per
 * thread; this lock protects only the process-wide syscall transport. */
static volatile uintptr_t dl_wire_owner;
static unsigned dl_wire_depth;
static uintptr_t dl_wire_token(void) {
  void *tp;
  volatile char marker;
  __asm__ volatile("movc %0, tp" : "=r"(tp));
  /* The thread pointer survives an alternate signal stack. Child contexts
   * that deliberately omit TLS fall back to their ordinary stack identity. */
  if (tp)
    return __builtin_capstone_cap_get_base(tp);
  return __builtin_capstone_cap_get_base((void *)&marker);
}
static void dl_wire_acquire(void) {
  uintptr_t token = dl_wire_token();
  if (__atomic_load_n(&dl_wire_owner, __ATOMIC_ACQUIRE) == token) {
    ++dl_wire_depth;
    return;
  }
  uintptr_t vacant = 0;
  while (!__atomic_compare_exchange_n(&dl_wire_owner, &vacant, token, 0,
                                      __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
    vacant = 0;
    __asm__ volatile("" ::: "memory");
  }
  dl_wire_depth = 1;
}
static void dl_wire_release(void) {
  if (--dl_wire_depth == 0)
    __atomic_store_n(&dl_wire_owner, 0, __ATOMIC_RELEASE);
}

void __capstone_delegate_regions(void *entry, void *exchange) {
  dl_entry = entry;
  dl_exchange = exchange;
  __capstone_signals_regions(entry, entry ? cap_bytes(entry) : 0);
  dl_capacity = exchange ? cap_bytes(exchange) : 0;
  if (dl_capacity > 0x40000000u)
    dl_capacity = 0x40000000u;
}

int __capstone_delegate_ready(void) {
  return dl_entry && dl_exchange && cap_bytes((void *)dl_entry) >= sizeof *dl_entry;
}

/* Copies between the caller's memory and the exchange region are DATA copies,
 * never the libc's memcpy: that one moves aligned 16-byte slots as
 * capabilities, tag included, and a buffer that held a pointer before a call
 * (an uninitialized struct on a stack that held one, pymalloc's free list)
 * would carry its tag into the region; the launcher's stores change the
 * bytes but capstone-qemu keeps the granule's tag, and the copy back would
 * load the old pointer instead of what the kernel wrote (CPython's recv_fds
 * and inet-dgram, 2026-09-30). Eight-byte integer moves neither carry nor
 * keep a tag. The runtime is built with -fno-builtin, so this stays a loop. */
static void dl_bytes(void *dst, const void *src, size_t n) {
  unsigned char *d = dst;
  const unsigned char *s = src;
  while (n && ((__builtin_capstone_cap_get_cursor(d) | __builtin_capstone_cap_get_cursor((void *)s)) & 7)) { *d++ = *s++; --n; }
  while (n >= 8) {
    uint64_t word;
    __builtin_memcpy(&word, s, 8);   /* an 8-byte load and store, never a capability */
    *(uint64_t *)d = word;
    d += 8; s += 8; n -= 8;
  }
  while (n) { *d++ = *s++; --n; }
}

/* Offset 0 means NULL for an optional buffer, so no buffer ever lives there:
   the first 16 bytes of the exchange region stay unused. */
#define DL_FIRST 16
static void dl_reset(void) { dl_used = DL_FIRST; }

static int dl_alloc(size_t bytes, uint64_t *offset) {
  size_t aligned = (dl_used + 15) & ~(size_t)15;
  if (aligned > dl_capacity || bytes > dl_capacity - aligned)
    return -1;
  *offset = aligned;
  dl_used = aligned + bytes;
  return 0;
}

/* One round trip. args are already offsets where the shape says so. The
 * events the launcher published are taken here; they run in dl_settle, once
 * the caller has its output data back and the exchange region is free. */
static long dl_round(uint64_t nr, const uint64_t args[CAPSTONE_DELEGATE_ARGS]) {
  struct capstone_delegate_entry e;
  if (capstone_delegate_pack(&e, nr, args)) {
    dl_status = CAPSTONE_ROUND_DONE;
    return -EINVAL;
  }
  memcpy((void *)dl_entry, &e, sizeof e);
  __capstone_yield();
  dl_status = dl_entry->status;
  __capstone_signals_take();
  return (long)dl_entry->result;
}

/* After a round's data is safe: deliver, and say whether the call must be
 * issued again because it has no result yet. */
static int dl_settle(void) {
  int retry = dl_status == CAPSTONE_ROUND_RETRY;
  __capstone_signals_deliver();
  return retry;
}

/* Runtime requests with integer arguments only. */
long __capstone_delegate_ints(uint64_t nr, uint64_t a, uint64_t b, uint64_t c) {
  uint64_t args[CAPSTONE_DELEGATE_ARGS] = {a, b, c, 0, 0, 0};
  long r;
  if (!__capstone_delegate_ready())
    return -EIO;
  dl_wire_acquire();
  do {
    dl_reset();
    r = dl_round(nr, args);
  } while (dl_settle());
  dl_wire_release();
  return r;
}

static long dl_string(const char *s, uint64_t *offset) {
  size_t n, available;
  if (!s)
    return -EFAULT;
  if (!dl_buffer_ok(s, 1, 4))
    return -EFAULT;
  available = cap_bytes((void *)s);
  n = strnlen(s, available);
  if (n == available)
    return -EFAULT;
  ++n;
  if (dl_alloc(n, offset))
    return -ENAMETOOLONG;
  dl_bytes(dl_exchange + *offset, s, n);
  return 0;
}

/* Buffers whose length is another argument are clamped to what fits: a short
 * read, write or listing is a result every caller already handles. */
static long dl_call_once(const struct capstone_delegate_shape *s, uint64_t nr,
                         syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS]) {
  uint64_t args[CAPSTONE_DELEGATE_ARGS];
  struct dl_slot slots[CAPSTONE_DELEGATE_ARGS] = {{0}};
  long result;
  dl_status = CAPSTONE_ROUND_DONE;
  dl_reset();
  dl_entry->nr = nr; /* named in a fault record if the copy below faults */
  for (unsigned i = 0; i < CAPSTONE_DELEGATE_ARGS; ++i)
    args[i] = (uint64_t)(unsigned long)raw[i];
  /* Two passes: a buffer whose length is a word in the region needs the word
     placed first. */
  for (unsigned pass = 0; pass < 2; ++pass)
   for (unsigned i = 0; i < s->argc; ++i) {
    const struct capstone_delegate_arg *a = &s->args[i];
    size_t bytes;
    int optional = a->kind == CAPSTONE_ARG_OPT_IN || a->kind == CAPSTONE_ARG_OPT_OUT ||
                   a->kind == CAPSTONE_ARG_OPT_INOUT;
    int word = a->length == CAPSTONE_LEN_WORD;
    if (a->kind == CAPSTONE_ARG_INT || word != (pass == 1))
      continue;
    if (a->kind == CAPSTONE_ARG_STR || a->kind == CAPSTONE_ARG_OPT_STR) {
      if (a->kind == CAPSTONE_ARG_OPT_STR && !raw[i]) { args[i] = 0; continue; }
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
      if (!((a->length == CAPSTONE_LEN_ARG || a->length == CAPSTONE_LEN_ARG_SCALED) &&
            args[a->size] == 0))
        return -EFAULT;
    }
    /* The caller's whole requested span must fit, even when the exchange
       region will service only a short prefix of this call. */
    if (a->length == CAPSTONE_LEN_ARG && a->size < CAPSTONE_DELEGATE_ARGS &&
        args[a->size] && !dl_buffer_ok(raw[i], args[a->size], dl_rights(a->kind)))
      return -EFAULT;
    if (a->length == CAPSTONE_LEN_ARG_SCALED && a->size < CAPSTONE_DELEGATE_ARGS &&
        a->scale && args[a->size]) {
      if (args[a->size] > SIZE_MAX / a->scale ||
          !dl_buffer_ok(raw[i], (size_t)args[a->size] * a->scale, dl_rights(a->kind)))
        return -EFAULT;
    }
    size_t aligned = (dl_used + 15) & ~(size_t)15;
    if (aligned >= dl_capacity)
      return -ENOMEM;
    size_t room = dl_capacity - aligned;
    if (a->length == CAPSTONE_LEN_ARG && a->size < CAPSTONE_DELEGATE_ARGS && args[a->size] > room)
      args[a->size] = room;
    /* an output list is clamped to what fits, as a read is: fewer entries */
    if (a->length == CAPSTONE_LEN_ARG_SCALED && a->size < CAPSTONE_DELEGATE_ARGS && a->scale &&
        (a->kind == CAPSTONE_ARG_OUT || a->kind == CAPSTONE_ARG_OPT_OUT) &&
        args[a->size] > room / a->scale)
      args[a->size] = room / a->scale;
    if (word && a->size < CAPSTONE_DELEGATE_ARGS && args[a->size]) {
      /* the length the caller passed behind a pointer, clamped to the room;
         the kernel reports the object's true length in the word regardless */
      uint32_t w;
      memcpy(&w, dl_exchange + args[a->size], sizeof w);
      if (w && !dl_buffer_ok(raw[i], w, dl_rights(a->kind)))
        return -EFAULT;
      if (w > room) {
        w = (uint32_t)room;
        memcpy(dl_exchange + args[a->size], &w, sizeof w);
      }
    }
    {
      struct capstone_delegate_entry probe = {0};
      memcpy(probe.args, args, sizeof args);
      bytes = capstone_delegate_arg_bytes(s, &probe, dl_exchange, dl_capacity, i);
    }
    if (bytes && !dl_buffer_ok(raw[i], bytes, dl_rights(a->kind)))
      return -EFAULT;
    if (dl_alloc(bytes, &args[i]))
      return -ENOMEM;
    if (bytes && a->kind != CAPSTONE_ARG_OUT && a->kind != CAPSTONE_ARG_OPT_OUT)
      dl_bytes(dl_exchange + args[i], raw[i], bytes);
    slots[i].domain = raw[i];
    slots[i].offset = args[i];
    slots[i].bytes = bytes;
    slots[i].copy_back = a->kind != CAPSTONE_ARG_IN && a->kind != CAPSTONE_ARG_OPT_IN;
   }
  result = dl_round(nr, args);
  if (dl_status == CAPSTONE_ROUND_RETRY)
    return 0;   /* no result, no output: the caller issues the call again */
  for (unsigned i = 0; i < CAPSTONE_DELEGATE_ARGS; ++i)
    if (slots[i].domain && slots[i].copy_back) {
      size_t bytes = capstone_delegate_result_bytes(nr, i, slots[i].bytes, result);
      if (bytes) dl_bytes(slots[i].domain, dl_exchange + slots[i].offset, bytes);
    }
  return result;
}

static long dl_call(uint64_t nr, syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS]) {
  const struct capstone_delegate_shape *s = capstone_delegate_shape(nr);
  long result;
  if (!s)
    return -ENOSYS;
  dl_wire_acquire();
  do result = dl_call_once(s, nr, raw);
  while (dl_settle());
  dl_wire_release();
  return result;
}

/* rt_sigtimedwait returns the kernel's 128-byte siginfo. Translate it into
 * musl's capability layout before a handler from this round can inspect the
 * caller's output buffer. */
static long dl_sigtimedwait(syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS]) {
  const struct capstone_delegate_shape *shape = capstone_delegate_shape(CAPSTONE_SYS_rt_sigtimedwait);
  siginfo_t *out = (siginfo_t *)raw[1];
  unsigned char wire[128];
  long result;
  if (out && !dl_buffer_ok(out, sizeof *out, 2))
    return -EFAULT;
  raw[1] = out ? (syscall_arg_t)wire : 0;
  dl_wire_acquire();
  do {
    result = dl_call_once(shape, CAPSTONE_SYS_rt_sigtimedwait, raw);
    if (result > 0 && dl_status != CAPSTONE_ROUND_RETRY && out)
      __capstone_siginfo_translate(wire, (int)result, out);
  } while (dl_settle());
  dl_wire_release();
  return result;
}

/* rt_sigprocmask for signals.c: two eight-byte buffers, and the mirror. */
long __capstone_delegate_procmask(int how, const uint64_t *set, uint64_t *old) {
  uint64_t in = set ? *set : 0, out = 0;
  syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS] = {(syscall_arg_t)(long)how, set ? (syscall_arg_t)&in : 0,
                                               old ? (syscall_arg_t)&out : 0, (syscall_arg_t)8, 0, 0};
  long r = dl_call(CAPSTONE_SYS_rt_sigprocmask, raw);
  if (r) return r;
  if (set) __capstone_sigmask_note(how, in);
  if (old) *old = out;
  return 0;
}

/* One kernel vector operation preserves pipe atomicity and file offsets.
 * Nested pointers are wire offsets as well. A large vector may return a short
 * prefix; a small vector (including PIPE_BUF writes) fits in one round. */
static long dl_vector_once(long fd, const struct iovec *iov, long count, int writing,
                           int positioned, long long offset) {
  uint64_t args[6] = {(uint64_t)fd, 0, 0, (uint64_t)offset & UINT32_MAX,
                      (uint64_t)offset >> 32, 0};
  size_t total = 0;
  dl_status = CAPSTONE_ROUND_DONE;
  if (count < 0 || count > 1024)
    return -EINVAL;
  if (count && !iov)
    return -EFAULT;
  if (count && !dl_buffer_ok(iov, (size_t)count * sizeof *iov, 4))
    return -EFAULT;
  struct dl_iov_slot slots[count ? (size_t)count : 1];
  for (long i = 0; i < count; ++i) {
    struct iovec current = iov[i];
    if (current.iov_len > (size_t)LONG_MAX - total)
      return -EINVAL;
    if (current.iov_len &&
        !dl_buffer_ok(current.iov_base, current.iov_len, writing ? 4 : 2))
      return -EFAULT;
    total += current.iov_len;
  }
  dl_reset();
  /* Reserve the entire descriptor array; oversized metadata is an error. */
  if (dl_alloc((size_t)count * 16, &args[1]))
    return -EMSGSIZE;
  for (long i = 0; i < count; ++i) {
    struct iovec current = iov[i];
    /* A concurrent edit of the descriptor cannot swap the checked pointer
       for an unchecked one. Use this same snapshot for copy-back. */
    if (current.iov_len &&
        !dl_buffer_ok(current.iov_base, current.iov_len, writing ? 4 : 2))
      return -EFAULT;
    size_t aligned = (dl_used + 15) & ~(size_t)15;
    size_t room = aligned < dl_capacity ? dl_capacity - aligned : 0;
    size_t bytes = current.iov_len < room ? current.iov_len : room;
    if (!bytes && current.iov_len)
      break;
    if (dl_alloc(bytes, &slots[i].offset))
      break;
    slots[i].base = current.iov_base;
    slots[i].bytes = bytes;
    uint64_t wire[2] = {slots[i].offset, bytes};
    memcpy(dl_exchange + args[1] + (size_t)i * 16, wire, sizeof wire);
    if (writing && bytes)
      dl_bytes(dl_exchange + slots[i].offset, slots[i].base, bytes);
    ++args[2];
    if (bytes < current.iov_len)
      break;
  }
  if (count && !args[2])
    return -EMSGSIZE;
  uint64_t nr = positioned ? (writing ? CAPSTONE_SYS_pwritev : CAPSTONE_SYS_preadv)
                           : (writing ? CAPSTONE_SYS_writev : CAPSTONE_SYS_readv);
  long result = dl_round(nr, args);
  if (dl_status == CAPSTONE_ROUND_RETRY)
    return 0;
  if (!writing && result > 0) {
    size_t left = (size_t)result;
    for (size_t i = 0; i < args[2] && left; ++i) {
      size_t n = slots[i].bytes < left ? slots[i].bytes : left;
      dl_bytes(slots[i].base, dl_exchange + slots[i].offset, n);
      left -= n;
    }
  }
  return result;
}

static long dl_vector(long fd, const struct iovec *iov, long count, int writing,
                      int positioned, long long offset) {
  long result;
  dl_wire_acquire();
  do {
    result = dl_vector_once(fd, iov, count, writing, positioned, offset);
    if (result < 0) { dl_wire_release(); return result; }
  }
  while (dl_settle());
  dl_wire_release();
  return result;
}

/* ioctl and fcntl carry a request-specific third argument. The ones a libc
 * uses are listed; anything else passes 0 and the kernel says EFAULT or
 * EINVAL, which is visible rather than a silent domain address. */
static long dl_ioctl(long fd, unsigned long request, void *argp) {
  /* musl passes the request as an int, so a request with bit 31 set (every
     _IOR one, TIOCGPTN among them) arrives sign-extended; the kernel reads
     an unsigned int, and so do the tables here and in the launcher */
  request &= 0xffffffffu;
  syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS] = {(syscall_arg_t)fd, (syscall_arg_t)request, 0, 0, 0, 0};
  size_t bytes = 0;
  switch (request) {
  case TIOCGWINSZ: case TIOCSWINSZ: bytes = 8; break;
  case TCGETS: case TCSETS: case TCSETSW: case TCSETSF: bytes = 60; break;
  case FIONREAD: case FIONBIO: case FIOCLEX: case FIONCLEX: bytes = request == FIOCLEX || request == FIONCLEX ? 0 : 4; break;
  /* the pseudo-terminal pair (unlockpt, ptsname) and the foreground process
     group (tcgetpgrp, tcsetpgrp): one int each, no pointer inside */
  case TIOCSPTLCK: case TIOCGPTN: case TIOCGPGRP: case TIOCSPGRP: bytes = 4; break;
  default: bytes = 0;
  }
  if (bytes && argp) {
    unsigned char buffer[64] = {0};
    long rc;
    if (!dl_buffer_ok(argp, bytes, 6))
      return -EFAULT;
    memcpy(buffer, argp, bytes);
    raw[2] = buffer;
    dl_wire_acquire();
    dl_reset();
    {
      uint64_t args[CAPSTONE_DELEGATE_ARGS] = {(uint64_t)fd, request, 0, 0, 0, 0};
      do {
        dl_reset();
        if (dl_alloc(64, &args[2]))
          { dl_wire_release(); return -ENOMEM; }
        dl_bytes(dl_exchange + args[2], buffer, sizeof buffer);
        rc = dl_round(CAPSTONE_NR_IOCTL_BUF, args);
        if (rc >= 0 && dl_status != CAPSTONE_ROUND_RETRY)
          dl_bytes(argp, dl_exchange + args[2], bytes);
      } while (dl_settle());
      dl_wire_release();
      return rc;
    }
  }
  if (bytes)
    return -EFAULT;
  if (request != FIOCLEX && request != FIONCLEX)
    return -ENOSYS;
  raw[2] = 0;
  return dl_call(CAPSTONE_SYS_ioctl, raw);
}

/* Sockets. A datagram is one unit: a message the exchange region cannot hold
 * is EMSGSIZE, the kernel's own answer for a datagram too long for its
 * protocol, never a short send. A stream send may be short, as a write may,
 * so the type is asked of the kernel only when a message does not fit. */
static long dl_socket_type(long fd) {
  int type = 0;
  uint32_t length = sizeof type;
  syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS] = {(syscall_arg_t)fd, (syscall_arg_t)SOL_SOCKET,
                                               (syscall_arg_t)SO_TYPE, (syscall_arg_t)&type,
                                               (syscall_arg_t)&length, 0};
  long r = dl_call(CAPSTONE_SYS_getsockopt, raw);
  return r < 0 ? r : type;
}

static size_t dl_room_after(size_t reserved) {
  size_t used = DL_FIRST + reserved;
  return used < dl_capacity ? dl_capacity - used : 0;
}

static long dl_sendto(syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS]) {
  size_t length = (size_t)raw[2], addrlen = raw[4] ? (size_t)raw[5] : 0;
  if (length > dl_room_after(64 + ((addrlen + 15) & ~(size_t)15))) {
    long type = dl_socket_type((long)raw[0]);
    if (type < 0) return type;
    if (type != SOCK_STREAM) return -EMSGSIZE;
  }
  return dl_call(CAPSTONE_SYS_sendto, raw);
}

/* sendmsg and recvmsg: the msghdr flattened into the block of capstone/msghdr.h,
 * offsets in place of pointers, the way readv's iovec array crosses. */
static long dl_msg_once(uint64_t nr, long fd, struct msghdr *msg, long flags) {
  int sending = nr == CAPSTONE_SYS_sendmsg;
  struct capstone_msghdr_block b = {0};
  uint64_t args[CAPSTONE_DELEGATE_ARGS] = {(uint64_t)fd, 0, (uint64_t)flags, 0, 0, 0};
  size_t count = (size_t)msg->msg_iovlen, fitted = 0;
  dl_status = CAPSTONE_ROUND_DONE;
  if (msg->msg_iovlen < 0 || count > CAPSTONE_MSGHDR_IOVS)
    return -EMSGSIZE;
  if (count && !msg->msg_iov)
    return -EFAULT;
  struct dl_iov_slot slots[count ? count : 1];
  dl_reset();
  if (dl_alloc(CAPSTONE_MSGHDR_BYTES, &args[1]))
    return -ENOMEM;
  if (count) {
    if (dl_alloc(count * 16, &b.iov))
      return -EMSGSIZE;
    b.iovlen = count;
  }
  if (msg->msg_name) {
    if (dl_alloc(msg->msg_namelen, &b.name))
      return -ENOMEM;
    dl_bytes(dl_exchange + b.name, msg->msg_name, msg->msg_namelen);
    b.namelen = msg->msg_namelen;
  }
  if (msg->msg_control) {
    if (dl_alloc(msg->msg_controllen, &b.control))
      return -ENOMEM;
    dl_bytes(dl_exchange + b.control, msg->msg_control, msg->msg_controllen);
    b.controllen = msg->msg_controllen;
  }
  for (size_t i = 0; i < count; ++i) {
    struct iovec current = msg->msg_iov[i];
    /* The iovec array may change after the outer preflight. Validate this
       snapshot and retain its capability through the later copy-back. */
    if (current.iov_len &&
        !dl_buffer_ok(current.iov_base, current.iov_len, sending ? 4 : 2))
      return -EFAULT;
    size_t aligned = (dl_used + 15) & ~(size_t)15;
    size_t room = aligned < dl_capacity ? dl_capacity - aligned : 0;
    size_t bytes = current.iov_len < room ? current.iov_len : room;
    if (!bytes && current.iov_len)
      break;
    if (dl_alloc(bytes, &slots[i].offset))
      break;
    slots[i].base = current.iov_base;
    slots[i].bytes = bytes;
    uint64_t pair[2] = {slots[i].offset, bytes};
    memcpy(dl_exchange + b.iov + 16 * i, pair, sizeof pair);
    if (sending && bytes)
      dl_bytes(dl_exchange + slots[i].offset, slots[i].base, bytes);
    ++fitted;
    if (bytes < current.iov_len)
      break;
  }
  if (count && !fitted)
    return -EMSGSIZE;
  b.iovlen = fitted;
  b.flags = (uint32_t)msg->msg_flags;
  memcpy(dl_exchange + args[1], &b, sizeof b);
  long result = dl_round(nr, args);
  if (dl_status == CAPSTONE_ROUND_RETRY)
    return 0;
  if (!sending && result >= 0) {
    struct capstone_msghdr_block back;
    size_t left = (size_t)result;
    dl_bytes(&back, dl_exchange + args[1], sizeof back);
    for (size_t i = 0; i < fitted && left; ++i) {
      size_t n = slots[i].bytes < left ? slots[i].bytes : left;
      dl_bytes(slots[i].base, dl_exchange + slots[i].offset, n);
      left -= n;
    }
    if (msg->msg_name) {
      size_t n = back.namelen < msg->msg_namelen ? (size_t)back.namelen : msg->msg_namelen;
      dl_bytes(msg->msg_name, dl_exchange + b.name, n);
    }
    if (msg->msg_control) {
      size_t n = back.controllen < msg->msg_controllen ? (size_t)back.controllen : msg->msg_controllen;
      dl_bytes(msg->msg_control, dl_exchange + b.control, n);
      msg->msg_controllen = (socklen_t)back.controllen;
    }
    msg->msg_namelen = (socklen_t)back.namelen;
    msg->msg_flags = (int)back.flags;
  }
  return result;
}

static long dl_msg(uint64_t nr, long fd, struct msghdr *msg, long flags) {
  long result;
  struct msghdr current;
  if (!msg)
    return -EFAULT;
  if (!dl_buffer_ok(msg, sizeof *msg, nr == CAPSTONE_SYS_sendmsg ? 4 : 6))
    return -EFAULT;
  current = *msg;
  if (current.msg_iovlen < 0 || current.msg_iovlen > CAPSTONE_MSGHDR_IOVS)
    return -EMSGSIZE;
  if (current.msg_iovlen &&
      !dl_buffer_ok(current.msg_iov, (size_t)current.msg_iovlen * sizeof *current.msg_iov, 4))
    return -EFAULT;
  if (current.msg_name && current.msg_namelen &&
      !dl_buffer_ok(current.msg_name, current.msg_namelen, nr == CAPSTONE_SYS_sendmsg ? 4 : 6))
    return -EFAULT;
  if (current.msg_control && current.msg_controllen &&
      !dl_buffer_ok(current.msg_control, current.msg_controllen, nr == CAPSTONE_SYS_sendmsg ? 4 : 6))
    return -EFAULT;
  for (size_t i = 0; i < (size_t)current.msg_iovlen; ++i) {
    struct iovec part = current.msg_iov[i];
    if (part.iov_len &&
        !dl_buffer_ok(part.iov_base, part.iov_len, nr == CAPSTONE_SYS_sendmsg ? 4 : 2))
      return -EFAULT;
  }
  if (nr == CAPSTONE_SYS_sendmsg) {
    /* the whole message must fit, or the socket must be a stream */
    size_t total = 0, reserved = CAPSTONE_MSGHDR_BYTES + 16;
    for (int i = 0; i < current.msg_iovlen; ++i)
      total += current.msg_iov[i].iov_len;
    reserved += (size_t)current.msg_iovlen * 16 + 16;
    if (current.msg_name) reserved += ((size_t)current.msg_namelen + 15 & ~(size_t)15) + 16;
    if (current.msg_control) reserved += ((size_t)current.msg_controllen + 15 & ~(size_t)15) + 16;
    reserved += (size_t)current.msg_iovlen * 16;   /* one alignment gap per buffer */
    if (total > dl_room_after(reserved)) {
      long type = dl_socket_type(fd);
      if (type < 0) return type;
      if (type != SOCK_STREAM) return -EMSGSIZE;
    }
  }
  dl_wire_acquire();
  do result = dl_msg_once(nr, fd, &current, flags);
  while (dl_settle());
  dl_wire_release();
  if (nr == CAPSTONE_SYS_recvmsg && result >= 0) {
    msg->msg_namelen = current.msg_namelen;
    msg->msg_controllen = current.msg_controllen;
    msg->msg_flags = current.msg_flags;
  }
  return result;
}

/* musl's epoll_event carries a pointer in its data union, 32 bytes here; the
 * kernel's is 16: the events, then the 64-bit data word. Converted both ways;
 * a pointer stored in the union crosses as its 64 bits. */
struct dl_epoll_wire { uint32_t events, pad; uint64_t data; };

static long dl_epoll_ctl(long epfd, long op, long fd, const struct epoll_event *ev) {
  struct dl_epoll_wire wire = {0, 0, 0};
  syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS] = {(syscall_arg_t)epfd, (syscall_arg_t)op, (syscall_arg_t)fd, 0, 0, 0};
  if (ev) {
    if (!dl_buffer_ok(ev, sizeof *ev, 4))
      return -EFAULT;
    wire.events = ev->events;
    wire.data = ev->data.u64;
    raw[3] = (syscall_arg_t)&wire;
  }
  return dl_call(CAPSTONE_SYS_epoll_ctl, raw);
}

static long dl_epoll_pwait(long epfd, struct epoll_event *ev, long count, long timeout,
                           const sigset_t *mask, long size) {
  static struct dl_epoll_wire wire[1024];   /* a shorter list is a legal answer */
  syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS];
  long r;
  if (count > 1024)
    count = 1024;
  if (ev && count > 0 && !dl_buffer_ok(ev, (size_t)count * sizeof *ev, 2))
    return -EFAULT;
  raw[0] = (syscall_arg_t)epfd;
  raw[1] = ev && count > 0 ? (syscall_arg_t)wire : (syscall_arg_t)ev;
  raw[2] = (syscall_arg_t)count;
  raw[3] = (syscall_arg_t)timeout;
  raw[4] = (syscall_arg_t)mask;
  raw[5] = (syscall_arg_t)size;
  r = dl_call(CAPSTONE_SYS_epoll_pwait, raw);
  for (long i = 0; ev && i < r && i < count; ++i) {
    ev[i].events = wire[i].events;
    ev[i].data.u64 = wire[i].data;
  }
  return r;
}

static long dl_fcntl(long fd, long cmd, void *arg) {
  if (cmd == F_GETLK || cmd == F_SETLK || cmd == F_SETLKW) {
    uint64_t args[CAPSTONE_DELEGATE_ARGS] = {(uint64_t)fd, (uint64_t)cmd, 0, 0, 0, 0};
    long rc;
    if (!arg)
      return -EFAULT;
    if (!dl_buffer_ok(arg, 32, cmd == F_GETLK ? 6 : 4))
      return -EFAULT;
    dl_wire_acquire();
    do {
      dl_reset();
      if (dl_alloc(32, &args[2]))
        { dl_wire_release(); return -ENOMEM; }
      dl_bytes(dl_exchange + args[2], arg, 32);
      rc = dl_round(CAPSTONE_NR_FCNTL_LOCK, args);
      if (rc >= 0 && cmd == F_GETLK && dl_status != CAPSTONE_ROUND_RETRY)
        dl_bytes(arg, dl_exchange + args[2], 32);
    } while (dl_settle());
    dl_wire_release();
    return rc;
  }
  {
    /* the integer commands: the value travels as the number it is */
    syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS] = {(syscall_arg_t)fd, (syscall_arg_t)cmd,
                                                 (syscall_arg_t)(unsigned long)arg, 0, 0, 0};
    return dl_call(CAPSTONE_SYS_fcntl, raw);
  }
}

long __capstone_delegate_hello(unsigned long entry_address, unsigned long code_base,
                               unsigned long code_end) {
  uint64_t args[CAPSTONE_DELEGATE_ARGS] = {entry_address, code_base, code_end, 0, 0, 0};
  long r;
  dl_wire_acquire();
  do r = dl_round(CAPSTONE_NR_HELLO, args);
  while (dl_settle());
  dl_wire_release();
  return r;
}

long __capstone_delegate_call(long n, syscall_arg_t a, syscall_arg_t b,
                              syscall_arg_t c, syscall_arg_t d,
                              syscall_arg_t e, syscall_arg_t f) {
  syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS] = {a, b, c, d, e, f};
  if (!__capstone_delegate_ready())
    return -EIO;
  /* The hint: the launcher's trampoline accepted a signal since the last
     round. Fetch it before this call, local answers included, so the handler
     runs at the next entry into this dispatcher and not at the next yield. */
  if (__capstone_signals_hint())
    __capstone_delegate_ints(CAPSTONE_NR_SIGPOLL, 0, 0, 0);
  /* Signals: handlers are domain addresses and stay here; the class, the
     flags, the mask and the alternate stack are what crosses or what the
     domain keeps. */
  switch (n) {
  case SYS_rt_sigaction:
    if ((unsigned long)d != 8) return -EINVAL;
    return __capstone_sigaction((int)(long)a, (const struct k_sigaction *)b, (struct k_sigaction *)c);
  case SYS_rt_sigprocmask:
    return __capstone_sigprocmask((int)(long)a, (const sigset_t *)b, (sigset_t *)c, (unsigned long)d);
  case SYS_sigaltstack:
    return __capstone_sigaltstack((const stack_t *)a, (stack_t *)b);
  default:
    break;
  }
  switch (n) {
  case SYS_rt_sigtimedwait:
    return dl_sigtimedwait(raw);
  case SYS_pselect6: {
    /* the sixth argument is {const sigset_t *, size_t}; the wire carries the
       mask itself, and the launcher rebuilds the pair for the kernel */
    const struct { const sigset_t *ss; size_t len; } *sig = (const void *)f;
    if (sig && sig->ss && sig->len != 8) return -EINVAL;
    raw[5] = sig && sig->ss ? (syscall_arg_t)sig->ss : 0;
    return dl_call(CAPSTONE_SYS_pselect6, raw);
  }
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
  case SYS_epoll_ctl:
    return dl_epoll_ctl((long)a, (long)b, (long)c, (const struct epoll_event *)d);
  case SYS_epoll_pwait:
    return dl_epoll_pwait((long)a, (struct epoll_event *)b, (long)c, (long)d, (const sigset_t *)e, (long)f);
  case SYS_sendto:
    return dl_sendto(raw);
  case SYS_sendmsg:
    return dl_msg(CAPSTONE_SYS_sendmsg, (long)a, (struct msghdr *)b, (long)c);
  case SYS_recvmsg:
    return dl_msg(CAPSTONE_SYS_recvmsg, (long)a, (struct msghdr *)b, (long)c);
  case SYS_ioctl:
    return dl_ioctl((long)a, (unsigned long)b, c);
  case SYS_fcntl:
    return dl_fcntl((long)a, (long)b, c);
  case SYS_execve: {
    /* exec in place: the task replaces itself with the named image */
    static char block[CAPSTONE_SPAWN_BYTES];
    size_t bytes;
    int error = capstone_spawn_pack(block, sizeof block, CAPSTONE_SPAWN_EXEC, 0, (const char *)a,
                                    (char *const *)b, (char *const *)c, NULL, 0, NULL, &bytes);
    if (error)
      return -error;
    {
      long rc = __capstone_delegate_spawn(block, bytes);
      if (rc == -ENOSYS)
        __capstone_hc_note_unserved(n);
      return rc;
    }
  }
  case SYS_getpid: case SYS_gettid: case SYS_getppid:
  case SYS_getuid: case SYS_geteuid: case SYS_getgid: case SYS_getegid: {
    long answer;
    if (dl_identity(n, &answer))
      return answer;
    break;
  }
  case SYS_clock_gettime:
    if (dl_clock((long)a, (struct timespec *)b))
      return 0;
    break;
  /* musl's thread setup, for the one thread a domain has: the tid it stores is
     the pid, and there is no robust list to register. Neither reaches Linux. */
  case SYS_set_tid_address: {
    const struct capstone_launch_task *t = __capstone_launch_task();
    return t && t->pid ? (long)t->pid : 1;
  }
  case SYS_set_robust_list:
    return 0;
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
  case CAPSTONE_GROUP_DELEGATED: {
    long rc = dl_call((uint64_t)n, raw);
    if (rc == -ENOSYS)
      __capstone_hc_note_unserved(n);
    return rc;
  }
  case CAPSTONE_GROUP_SIGNAL:
    /* rt_sigreturn: no kernel frame exists here; handlers return to the
       dispatcher that invoked them */
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

/* posix_spawn's request: the packed block crosses as an IN buffer. */
long __capstone_delegate_spawn(const void *block, unsigned long bytes) {
  syscall_arg_t raw[CAPSTONE_DELEGATE_ARGS] = {(syscall_arg_t)block, (syscall_arg_t)bytes, 0, 0, 0, 0};
  if (!__capstone_delegate_ready())
    return -EIO;
  if (bytes > dl_capacity - 16)
    return -E2BIG;
  return dl_call(CAPSTONE_NR_SPAWN, raw);
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
