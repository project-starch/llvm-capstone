#ifndef CAPSTONE_LAUNCH_H
#define CAPSTONE_LAUNCH_H

#include <stddef.h>
#include <stdint.h>

/* Little-endian wire data. Offsets name NUL-terminated byte strings, never
 * pointers. The same decoder is tested natively and linked into the CRT. */
#define CAPSTONE_LAUNCH_VERSION 2u
#define CAPSTONE_LAUNCH_BYTES 65536u
#define CAPSTONE_LAUNCH_STRINGS 1024u
#define CAPSTONE_APPLICATION_MAGIC UINT64_C(0x315050414e4f5043)
#define CAPSTONE_APPLICATION_RECOVERY 1u

struct capstone_launch_header {
  unsigned char magic[8];
  uint32_t version, bytes, argc, envc;
  uint32_t cwd, stdio_mask, pid, ppid;
  uint32_t uid, euid, gid, egid;
  uint64_t realtime_ns, monotonic_ns, ticks, ticks_per_second;
};

/* What the task knows about itself when it starts the domain. The identity
 * never changes for a domain: it cannot fork or change credentials, and an
 * exec in place keeps the pid. The clock pair is a vDSO in one record: the two
 * clocks and the rdtime counter read together, plus the counter's rate, so the
 * libc answers clock_gettime without a round. A rate of zero means the
 * launcher could not read the timebase and the libc delegates time calls. */
struct capstone_launch_task {
  uint32_t pid, ppid, uid, euid, gid, egid;
  uint64_t realtime_ns, monotonic_ns, ticks, ticks_per_second;
};

struct capstone_application_descriptor {
  uint64_t magic, version, flags, launch_bytes, heap_bytes;
};

/* Descriptor v2: the v1 fields, CAPSTONE_APPLICATION_DELEGATE in flags, and
 * the exchange region the launcher grants. The prefix retains its wire layout;
 * applications with the old 40-byte descriptor are rejected. */
struct capstone_application_descriptor_v2 {
  struct capstone_application_descriptor v1;
  uint64_t exchange_bytes;
};

struct capstone_launch_view {
  unsigned argc, envc, stdio_mask;
  char *cwd;
  struct capstone_launch_task task;
};

/* Return an errno value (zero on success). No allocation or global state. */
int capstone_launch_pack(void *buffer, size_t capacity, int argc,
                         char *const argv[], char *const envp[],
                         const char *cwd, unsigned stdio_mask,
                         const struct capstone_launch_task *task);
int capstone_launch_unpack(void *buffer, size_t capacity,
                           char **argv, size_t argv_slots,
                           char **envp, size_t env_slots,
                           struct capstone_launch_view *view);

#endif
