/* Delegated syscall ABI v2: the wire block between a domain and its Linux task.
 *
 * One request per block for now. This is a runtime-specific layout, not an
 * io_uring SQE; a future transport must translate it to kernel submissions.
 * Pointer arguments cross as byte offsets into the exchange region, never as domain addresses and never as capabilities.
 * The semantics of every delegated call are Linux's; this header carries only
 * the transport, the per-syscall argument shapes and the closed exception list.
 * See docs/plans/delegation-abi.md.
 */
#ifndef CAPSTONE_DELEGATE_H
#define CAPSTONE_DELEGATE_H

#include <stddef.h>
#include <stdint.h>

#define CAPSTONE_DELEGATE_VERSION 2u
#define CAPSTONE_DELEGATE_ARGS 6u

/* Fixed-width, little-endian, 96 bytes. `flags` bit i says args[i] is an
 * offset into the exchange region; the shape table says how many bytes that
 * offset must cover. An optional buffer argument of 0 is NULL, so the libc
 * never places a buffer at offset 0. The launcher writes `result`, `pending`
 * (bit n-1 set: a signal n event was published this round) and `status` on
 * every return. */
struct capstone_delegate_entry {
  uint32_t version;
  uint32_t count;
  uint64_t nr;
  uint64_t args[CAPSTONE_DELEGATE_ARGS];
  uint64_t flags;
  int64_t result;
  uint64_t pending;
  uint64_t status;
};

/* Round status. RETRY: the call has no result for the application yet, either
 * because signals were accepted before it entered the kernel or because Linux
 * prepared a restart; the libc delivers the published events and issues the
 * call again from its caller's buffers. Only DONE carries output data. */
#define CAPSTONE_ROUND_DONE 0u
#define CAPSTONE_ROUND_RETRY 1u

/* One transport per context: a META block of CAPSTONE_DELEGATE_META_BYTES
 * (the entry at offset 0, the signal handover block at CAPSTONE_SIGNAL_OFFSET)
 * and an exchange region of the descriptor's exchange_bytes. The launcher
 * grants 1 + contexts of each as two regions, transport i at i times the block
 * size in both; transport 0 is the first context's. After the last META block
 * the META region holds the park table (CAPSTONE_PARK_BYTES). See
 * docs/plans/delegation-signals.md and docs/plans/delegation-threads.md. */
#define CAPSTONE_DELEGATE_META_BYTES 16384u
/* Contexts besides the first with a transport of their own: the monitor lends
 * each application 8 invocation descriptors (process-abi.h). */
#define CAPSTONE_DELEGATE_CONTEXTS_MAX 7u
#define CAPSTONE_SIGNAL_OFFSET 4096u
#define CAPSTONE_SIGNAL_EVENTS 64u
#define CAPSTONE_SIGNAL_WAIT 1u  /* accepted inside a wait with a temporary mask */
#define CAPSTONE_SIGNAL_DEFER 2u /* its own signal remains blocked until SIGDONE */

/* One accepted signal. `mask` is the mask the domain handler is based on: the
 * wait's temporary mask for a WAIT event, otherwise the task's mask at
 * acceptance. `info` is the kernel's siginfo, RV64 layout, 128 bytes. */
struct capstone_signal_event {
  uint64_t seq;
  uint32_t signo, flags;
  uint64_t mask;
  uint64_t generation;
  unsigned char info[128];
};

/* Written by the launcher; read by the libc between rounds. `recorded` counts
 * every accepted event and is written by the trampoline itself, so the libc
 * can see between rounds that something waits (the hint). `published` counts
 * the events handed over so far; events[0..count) are those published by the
 * round that just ended, in acceptance order. */
struct capstone_signal_block {
  uint64_t recorded;
  uint64_t published;
  uint32_t count, reserved;
  struct capstone_signal_event events[CAPSTONE_SIGNAL_EVENTS];
  /* Appended after events so existing images retain the handover offsets. */
  uint64_t initial_mask, initial_ignored; /* inherited state at launcher startup */
};

/* How one argument of one syscall is interpreted. */
enum capstone_delegate_kind {
  CAPSTONE_ARG_INT = 0,   /* passed as is */
  CAPSTONE_ARG_IN,        /* exchange offset, bytes read by the kernel */
  CAPSTONE_ARG_OUT,       /* exchange offset, bytes written by the kernel */
  CAPSTONE_ARG_INOUT,     /* both */
  CAPSTONE_ARG_STR,       /* exchange offset of a NUL-terminated string */
  CAPSTONE_ARG_OPT_IN,    /* IN, or zero for NULL */
  CAPSTONE_ARG_OPT_OUT,   /* OUT, or zero for NULL */
  CAPSTONE_ARG_OPT_INOUT, /* INOUT, or zero for NULL */
  CAPSTONE_ARG_OPT_STR    /* STR, or zero for NULL (utimensat by fd) */
};

/* Where a buffer argument's length comes from. */
enum capstone_delegate_length {
  CAPSTONE_LEN_NONE = 0,  /* an integer, or a string */
  CAPSTONE_LEN_FIXED,     /* `size` bytes */
  CAPSTONE_LEN_ARG,       /* args[size] bytes */
  CAPSTONE_LEN_ARG_SCALED /* args[size] elements of `scale` bytes */
};

struct capstone_delegate_arg {
  uint8_t kind;   /* enum capstone_delegate_kind */
  uint8_t length; /* enum capstone_delegate_length */
  uint16_t size;  /* bytes, argument index, or element count source */
  uint8_t scale;  /* element size for CAPSTONE_LEN_ARG_SCALED */
  uint8_t reserved;
};

/* The closed exception list. Everything not in a group is delegated. */
enum capstone_delegate_group {
  CAPSTONE_GROUP_DELEGATED = 0,
  CAPSTONE_GROUP_MEMORY,   /* served by the domain allocator or ENOSYS */
  CAPSTONE_GROUP_PROCESS,  /* the spawn service, or ENOSYS for fork */
  CAPSTONE_GROUP_SIGNAL,   /* domain table and launcher mask */
  CAPSTONE_GROUP_RUNTIME,  /* runtime-internal, answered by the launcher itself */
  CAPSTONE_GROUP_UNKNOWN   /* not in the table: ENOSYS, recorded */
};

/* Runtime-internal request numbers, above every Linux number. HELLO is the
 * first request of every run: args[0] is the runtime address of domain_main,
 * args[1] and args[2] the code capability's base and end, so a fault record
 * can be symbolized against the image's link addresses. */
#define CAPSTONE_NR_HELLO UINT64_C(0xC0DE0001)
/* fcntl and ioctl carry an integer or a pointer in their third argument by
 * command. The integer forms travel as the Linux number; the pointer forms
 * the libc uses travel as these, whose third argument is a fixed buffer, and
 * the launcher runs the Linux call with the buffer's address. */
#define CAPSTONE_NR_FCNTL_LOCK UINT64_C(0xC0DE0003)  /* F_GETLK, F_SETLK, F_SETLKW: 32 bytes */
#define CAPSTONE_NR_IOCTL_BUF UINT64_C(0xC0DE0004)   /* a request with a buffer: 64 bytes */
/* Signals: the domain keeps its handlers, the task keeps Linux's state.
 * SIGACTION carries a disposition class instead of a handler pointer, plus
 * the flags Linux applies itself; SIGDONE acknowledges an event's sequence
 * number after its handler ran; SIGPOLL publishes accepted events without a
 * call, the answer to the hint. */
#define CAPSTONE_NR_SIGACTION UINT64_C(0xC0DE0005) /* signo, class, flags */
#define CAPSTONE_NR_SIGDONE UINT64_C(0xC0DE0006)   /* seq */
#define CAPSTONE_NR_SIGPOLL UINT64_C(0xC0DE0007)

/* Contexts (docs/plans/delegation-threads.md).
 * CONTEXT_RESERVE: no arguments; the result is a free transport index, 1 to
 * the descriptor's contexts, or -EAGAIN when every one is in use (-ENOSYS for
 * an application that declares none). The creator puts the index into the
 * new context's start block before CONTEXT_CREATE, so the context has its
 * transport at its first entry.
 * CONTEXT_CREATE: ticket, mode, transport; the seal is already in the
 * requesting context's invocation descriptor, the result is the new context's
 * id or -errno. It consumes a reservation whatever its outcome: THREAD mode
 * names the reserved transport, which the launcher thread serves until the
 * context ends; REGISTER mode names none (0), and that context makes no
 * delegated call.
 * CONTEXT_STEP: id, 0, event; the launcher steps a REGISTER context once and
 * writes a capstone_context_event. CONTEXT_FORGET: id. */
#define CAPSTONE_NR_CONTEXT_CREATE UINT64_C(0xC0DE0008)
#define CAPSTONE_NR_CONTEXT_STEP UINT64_C(0xC0DE0009)
#define CAPSTONE_NR_CONTEXT_FORGET UINT64_C(0xC0DE000A)
#define CAPSTONE_NR_CONTEXT_RESERVE UINT64_C(0xC0DE000B)
#define CAPSTONE_CONTEXT_REGISTER 0u   /* register only; the application steps it */
#define CAPSTONE_CONTEXT_THREAD 1u     /* a launcher thread steps it until it ends */

/* Parking (docs/plans/delegation-threads.md, "Parking"). The domain keeps the
 * lock words; Linux only puts launcher threads to sleep and wakes them. A key
 * is the domain address of a lock word as an integer; the launcher never
 * dereferences it. The park table has one generation word per bucket; only
 * the launcher writes one, and it never wraps (it stops at UINT64_MAX).
 * PARK_WAIT: key, gen, deadline (absolute CLOCK_MONOTONIC nanoseconds, 0 for
 *   none); sleeps only while the key's bucket still has generation gen. The
 *   result is CAPSTONE_PARK_RESULT_WOKEN or _RECHECK, or -ETIMEDOUT or
 *   -EINTR; a signal accepted meanwhile under SA_RESTART ends it as a RETRY
 *   round, after which the caller checks its lock word again.
 * PARK_WAKE: key, n; the result is how many waiters on key it selected.
 * PARK_REQUEUE: src, dst, nwake, nmove; wakes up to nwake waiters on src,
 *   moves up to nmove more to dst, and returns woken plus moved. */
#define CAPSTONE_NR_PARK_WAIT UINT64_C(0xC0DE000C)
#define CAPSTONE_NR_PARK_WAKE UINT64_C(0xC0DE000D)
#define CAPSTONE_NR_PARK_REQUEUE UINT64_C(0xC0DE000E)
#define CAPSTONE_PARK_RESULT_WOKEN 0
#define CAPSTONE_PARK_RESULT_RECHECK 1
#define CAPSTONE_PARK_BUCKETS 256u
#define CAPSTONE_PARK_BYTES 4096u

_Static_assert(CAPSTONE_PARK_BUCKETS * 8u <= CAPSTONE_PARK_BYTES, "the park table fits its page");

/* The bucket of a key, the same on both sides of the boundary. */
static inline unsigned capstone_park_bucket_of(uint64_t key, unsigned buckets) {
  return (unsigned)((key * UINT64_C(0x9E3779B97F4A7C15)) >> 32) & (buckets - 1);
}
struct capstone_context_event {
  uint64_t kind;     /* the driver's step event: returned, preempted, fault, dead, stale */
  uint64_t result;   /* the context's result word */
  uint64_t cause, pc, address;
  uint64_t reserved;
};
#define CAPSTONE_SIGNAL_DEFAULT 0u
#define CAPSTONE_SIGNAL_IGNORE 1u
#define CAPSTONE_SIGNAL_CAUGHT 2u

/* Descriptor flag: the image speaks this ABI. Images without it use the
 * HostCall v0 application runtime; a launcher must accept both. */
#define CAPSTONE_APPLICATION_DELEGATE 2u
#define CAPSTONE_DELEGATE_DEFAULT_EXCHANGE 262144u

struct capstone_delegate_shape {
  uint16_t nr;
  uint8_t group;    /* enum capstone_delegate_group */
  uint8_t argc;
  const char *name;
  struct capstone_delegate_arg args[CAPSTONE_DELEGATE_ARGS];
};

/* RV64 syscall numbers, the asm-generic table. Named here so the shape table
 * and both sides of the boundary agree without a kernel header. */
enum {
  CAPSTONE_SYS_getcwd = 17, CAPSTONE_SYS_dup = 23, CAPSTONE_SYS_dup3 = 24,
  CAPSTONE_SYS_fcntl = 25, CAPSTONE_SYS_ioctl = 29, CAPSTONE_SYS_flock = 32, CAPSTONE_SYS_mkdirat = 34,
  CAPSTONE_SYS_unlinkat = 35, CAPSTONE_SYS_symlinkat = 36, CAPSTONE_SYS_ftruncate = 46,
  CAPSTONE_SYS_faccessat = 48, CAPSTONE_SYS_chdir = 49, CAPSTONE_SYS_openat = 56,
  CAPSTONE_SYS_fchmodat = 53,
  CAPSTONE_SYS_close = 57, CAPSTONE_SYS_pipe2 = 59, CAPSTONE_SYS_getdents64 = 61,
  CAPSTONE_SYS_lseek = 62, CAPSTONE_SYS_read = 63, CAPSTONE_SYS_write = 64,
  CAPSTONE_SYS_readv = 65, CAPSTONE_SYS_writev = 66, CAPSTONE_SYS_pread64 = 67,
  CAPSTONE_SYS_pwrite64 = 68, CAPSTONE_SYS_ppoll = 73,
  CAPSTONE_SYS_preadv = 69, CAPSTONE_SYS_pwritev = 70,
  CAPSTONE_SYS_readlinkat = 78, CAPSTONE_SYS_newfstatat = 79,
  CAPSTONE_SYS_fstat = 80, CAPSTONE_SYS_fsync = 82, CAPSTONE_SYS_fdatasync = 83,
  CAPSTONE_SYS_sync_file_range = 84,
  CAPSTONE_SYS_utimensat = 88, CAPSTONE_SYS_exit = 93,
  CAPSTONE_SYS_exit_group = 94, CAPSTONE_SYS_set_tid_address = 96,
  CAPSTONE_SYS_futex = 98, CAPSTONE_SYS_set_robust_list = 99,
  CAPSTONE_SYS_nanosleep = 101, CAPSTONE_SYS_clock_gettime = 113,
  CAPSTONE_SYS_clock_nanosleep = 115, CAPSTONE_SYS_sched_yield = 124,
  CAPSTONE_SYS_kill = 129, CAPSTONE_SYS_tkill = 130, CAPSTONE_SYS_sigaltstack = 132,
  CAPSTONE_SYS_rt_sigsuspend = 133, CAPSTONE_SYS_rt_sigaction = 134,
  CAPSTONE_SYS_rt_sigprocmask = 135, CAPSTONE_SYS_rt_sigpending = 136,
  CAPSTONE_SYS_rt_sigtimedwait = 137, CAPSTONE_SYS_rt_sigreturn = 139,
  CAPSTONE_SYS_getitimer = 102, CAPSTONE_SYS_setitimer = 103,
  CAPSTONE_SYS_times = 153, CAPSTONE_SYS_uname = 160, CAPSTONE_SYS_umask = 166,
  CAPSTONE_SYS_gettimeofday = 169, CAPSTONE_SYS_getpid = 172,
  CAPSTONE_SYS_getppid = 173, CAPSTONE_SYS_getuid = 174,
  CAPSTONE_SYS_geteuid = 175, CAPSTONE_SYS_getgid = 176,
  CAPSTONE_SYS_getegid = 177, CAPSTONE_SYS_gettid = 178,
  CAPSTONE_SYS_sysinfo = 179, CAPSTONE_SYS_socket = 198,
  CAPSTONE_SYS_brk = 214, CAPSTONE_SYS_munmap = 215, CAPSTONE_SYS_mremap = 216,
  CAPSTONE_SYS_clone = 220, CAPSTONE_SYS_execve = 221, CAPSTONE_SYS_mmap = 222,
  CAPSTONE_SYS_mprotect = 226, CAPSTONE_SYS_madvise = 233,
  CAPSTONE_SYS_wait4 = 260, CAPSTONE_SYS_prlimit64 = 261,
  CAPSTONE_SYS_renameat2 = 276, CAPSTONE_SYS_getrandom = 278,
  CAPSTONE_SYS_vfork = 1071, CAPSTONE_SYS_fork = 1079
};

/* Lookup; NULL for a number outside the table. */
const struct capstone_delegate_shape *capstone_delegate_shape(uint64_t nr);
enum capstone_delegate_group capstone_delegate_group_of(uint64_t nr);

/* Fill an entry for one call. The shape decides which arguments are exchange
 * offsets, so callers pass the raw values and this sets `flags`. Returns
 * EINVAL for a group other than DELEGATED or RUNTIME. */
int capstone_delegate_pack(struct capstone_delegate_entry *entry, uint64_t nr,
                           const uint64_t args[CAPSTONE_DELEGATE_ARGS]);

/* The launcher's check before it touches the exchange region: version, count,
 * group, and every flagged argument inside [0, exchange_bytes) for the length
 * the shape implies. Returns 0, or the errno the request must be answered with
 * (EINVAL for a malformed block, EFAULT for an offset outside the region,
 * ENOSYS for an unknown or excepted number). RUNTIME requests pass. */
int capstone_delegate_validate(const struct capstone_delegate_entry *entry,
                               size_t exchange_bytes);

/* Bytes the argument at `index` covers in the exchange region, per the shape,
 * or 0 when it is not a buffer. Strings report 0: the launcher bounds them
 * with capstone_delegate_string_ok. */
size_t capstone_delegate_arg_bytes(const struct capstone_delegate_shape *shape,
                                   const struct capstone_delegate_entry *entry,
                                   unsigned index);

/* A NUL inside [offset, exchange_bytes). */
int capstone_delegate_string_ok(const char *exchange, size_t exchange_bytes,
                                uint64_t offset);

/* Bytes an output argument may change for this result (short I/O, errors and
 * wait4(WNOHANG) must leave the caller's remaining buffer untouched). */
size_t capstone_delegate_result_bytes(uint64_t nr, unsigned index, size_t bytes,
                                      int64_t result);

#endif
