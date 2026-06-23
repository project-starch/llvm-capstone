/*
 * Freestanding prelude for the Capstone PureCap BEEBS `dtoa` domain build.
 *
 * David Gay's dtoa/strtod source (BEEBS `src/dtoa/libdtoa.c`) normally pulls in
 * the hosted "stdlib.h"/"string.h"/"errno.h"/"math.h".  On the bare-metal
 * Capstone domain there is no hosted libc, so the per-benchmark build strips
 * those includes and this prelude supplies exactly the symbols they declared:
 *
 *   - size_t / NULL                 -> compiler-provided <stddef.h>
 *   - memcpy/memmove/memset/strlen/strcpy/strcmp
 *                                   -> shared adapted/beebs_freestanding_string.c
 *   - floor/ceil/log                -> shared adapted/beebs_softfloat_libm.c
 *   - errno / ERANGE                -> a plain global; strtod writes
 *                                      `errno = ERANGE` only on overflow/underflow
 *                                      paths, which none of the benchmark's test
 *                                      inputs reach, so a definition suffices.
 *
 * The benchmark's `char *nums[]` table is auto-tagged at runtime by the backend
 * constructor-codegen pass (CapstoneCapGlobalInit + start.S __capstone_cap_init),
 * so the upstream benchmark()/verify_benchmark are used unchanged (no tail).
 *
 * Build defines (see build-beebs-dtoa-capstone.sh):
 *   -DLong=int            target is LP64-like (long=64, int=32); David Gay's code
 *                         assumes ULong is 32-bit (it splits an IEEE double into
 *                         two 32-bit words) -- the case its own comment flags.
 *   -DNO_HEX_FP           omit the hex-float parser (decimal-only inputs; dead code).
 *   -DOmit_Private_Memory route every Bigint through malloc_beebs so a single
 *                         16-byte-aligned allocator backs them (a Bigint's first
 *                         field is a 16-byte capability; the build patches
 *                         malloc_beebs to 16-align in place, tag-preserving).
 */
#include <stddef.h> /* size_t, NULL */

#define ERANGE 34
int errno;

void *memcpy(void *, const void *, size_t);
void *memmove(void *, const void *, size_t);
void *memset(void *, int, size_t);
size_t strlen(const char *);
char *strcpy(char *, const char *);
int strcmp(const char *, const char *);

double floor(double);
double ceil(double);
double log(double);
