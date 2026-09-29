/* Delegated syscall ABI v2: the wire block between a domain and its Linux task.
 *
 * One request per block for now; `count` and the entry layout follow io_uring's
 * submission entry so a later kernel-side or io_uring transport changes the
 * transport and not this header. Pointer arguments cross as byte offsets into
 * the exchange region, never as domain addresses and never as capabilities.
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

/* Fixed-width, little-endian, 88 bytes. `flags` bit i says args[i] is an
 * offset into the exchange region; the shape table says how many bytes that
 * offset must cover. `pending` is written by the launcher on every return. */
struct capstone_delegate_entry {
  uint32_t version;
  uint32_t count;
  uint64_t nr;
  uint64_t args[CAPSTONE_DELEGATE_ARGS];
  uint64_t flags;
  int64_t result;
  uint64_t pending;
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
  CAPSTONE_ARG_OPT_INOUT  /* INOUT, or zero for NULL */
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
  CAPSTONE_SYS_fcntl = 25, CAPSTONE_SYS_ioctl = 29, CAPSTONE_SYS_mkdirat = 34,
  CAPSTONE_SYS_unlinkat = 35, CAPSTONE_SYS_ftruncate = 46,
  CAPSTONE_SYS_faccessat = 48, CAPSTONE_SYS_chdir = 49, CAPSTONE_SYS_openat = 56,
  CAPSTONE_SYS_close = 57, CAPSTONE_SYS_pipe2 = 59, CAPSTONE_SYS_getdents64 = 61,
  CAPSTONE_SYS_lseek = 62, CAPSTONE_SYS_read = 63, CAPSTONE_SYS_write = 64,
  CAPSTONE_SYS_readv = 65, CAPSTONE_SYS_writev = 66, CAPSTONE_SYS_pread64 = 67,
  CAPSTONE_SYS_pwrite64 = 68, CAPSTONE_SYS_ppoll = 73,
  CAPSTONE_SYS_readlinkat = 78, CAPSTONE_SYS_newfstatat = 79,
  CAPSTONE_SYS_fstat = 80, CAPSTONE_SYS_fsync = 82, CAPSTONE_SYS_fdatasync = 83,
  CAPSTONE_SYS_utimensat = 88, CAPSTONE_SYS_exit = 93,
  CAPSTONE_SYS_exit_group = 94, CAPSTONE_SYS_set_tid_address = 96,
  CAPSTONE_SYS_futex = 98, CAPSTONE_SYS_set_robust_list = 99,
  CAPSTONE_SYS_nanosleep = 101, CAPSTONE_SYS_clock_gettime = 113,
  CAPSTONE_SYS_clock_nanosleep = 115, CAPSTONE_SYS_sched_yield = 124,
  CAPSTONE_SYS_kill = 129, CAPSTONE_SYS_rt_sigaction = 134,
  CAPSTONE_SYS_rt_sigprocmask = 135, CAPSTONE_SYS_rt_sigreturn = 139,
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

#endif
