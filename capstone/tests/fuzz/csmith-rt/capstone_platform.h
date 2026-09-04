/* The csmith runtime's platform layer for a Capstone domain.
 *
 * Force-included (-include) ahead of csmith.h.  It pre-defines csmith's own
 * platform_generic.h include guard so that stdio-based layer never enters the
 * translation unit, and supplies the two hooks csmith's generated main() calls:
 * platform_main_end receives the program's checksum, which is the whole result of
 * a csmith program, and parks it in a global that fuzz_domain.c returns through the
 * domain's 32-bit result channel.  printf is referenced by csmith's transparent_crc
 * (never called with the default flag) and is stubbed in fuzz_domain.c.
 */
#ifndef CAPSTONE_FUZZ_PLATFORM_H
#define CAPSTONE_FUZZ_PLATFORM_H

#define PLATFORM_GENERIC_H 1
#define NOT_PRINT_CHECKSUM 1

#include <stdint.h>

extern volatile uint32_t capstone_fuzz_checksum;
extern volatile uint32_t capstone_fuzz_stage;
int printf(const char *fmt, ...);

static void platform_main_begin(void) { capstone_fuzz_stage = 1; }

static void platform_main_end(uint32_t crc, int flag) {
  (void)flag;
  capstone_fuzz_checksum = crc;
  capstone_fuzz_stage = 2;
}

#define MB (1 << 20)

#endif
