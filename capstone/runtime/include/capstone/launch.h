#ifndef CAPSTONE_LAUNCH_H
#define CAPSTONE_LAUNCH_H

#include <stddef.h>
#include <stdint.h>

/* Little-endian wire data. Offsets name NUL-terminated byte strings, never
 * pointers. The same decoder is tested natively and linked into the CRT. */
#define CAPSTONE_LAUNCH_VERSION 1u
#define CAPSTONE_LAUNCH_BYTES 65536u
#define CAPSTONE_LAUNCH_STRINGS 1024u
#define CAPSTONE_APPLICATION_MAGIC UINT64_C(0x315050414e4f5043)
#define CAPSTONE_APPLICATION_RECOVERY 1u

struct capstone_launch_header {
  unsigned char magic[8];
  uint32_t version, bytes, argc, envc;
  uint32_t cwd, stdio_mask, reserved0, reserved1;
};

struct capstone_application_descriptor {
  uint64_t magic, version, flags, launch_bytes, heap_bytes;
};

struct capstone_launch_view {
  unsigned argc, envc, stdio_mask;
  char *cwd;
};

/* Return an errno value (zero on success). No allocation or global state. */
int capstone_launch_pack(void *buffer, size_t capacity, int argc,
                         char *const argv[], char *const envp[],
                         const char *cwd, unsigned stdio_mask);
int capstone_launch_unpack(void *buffer, size_t capacity,
                           char **argv, size_t argv_slots,
                           char **envp, size_t env_slots,
                           struct capstone_launch_view *view);

#endif
