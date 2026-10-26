/* CoreMark-derived workload, virtual-profile port. The core files are upstream cfa9ab37; core_main.c
 * is compiled with -Dmain=coremark_main so the harness can run several iteration sets in one
 * process (see core_portme.c). One core line is changed, in a build-directory copy (build.py): the
 * list's per-node size in core_list_init, `16+sizeof(struct list_data_s)`, becomes
 * `sizeof(struct list_head_s)+sizeof(struct list_data_s)`. Unchanged, a 32-byte purecap list_head
 * makes the list overrun its third of the data block into the matrix's, and the matrix
 * initialisation overwrites live list pointers (QEMU: capability fault cause=24 on the first
 * list walk). On an LP64 build both expressions are 20, so the native reference is unaffected.
 * `build.py --coremark-unmodified` builds the unchanged file to reproduce the fault.
 *
 * Based on upstream simple/core_portme.h. Changes, and why:
 *  - timing: start_time/stop_time are the harness bracket (clock_gettime + perf counters);
 *    CORE_TICKS are nanoseconds;
 *  - HAS_PRINTF 0: ee_printf is the port's own printf, which also records the CRC report lines,
 *    so the verdict can be read without editing core_main.c (MAIN_RETURN_VAL is fixed by coremark.h);
 *  - ee_ptr_int is an opaque pointer-sized holder: on a 128-bit capability target no integer type
 *    holds a pointer, and the only arithmetic use, align_mem, is rewritten as pointer arithmetic;
 *  - SEED_VOLATILE performance-run seeds (0, 0, 0x66), ITERATIONS fixed at build time, MEM_MALLOC
 *    (one allocation of TOTAL_DATA_SIZE before start_time, freed after stop_time). */
#ifndef CORE_PORTME_H
#define CORE_PORTME_H
#include <stddef.h>

#define HAS_FLOAT 1
#define HAS_TIME_H 1
#define USE_CLOCK 0
#define HAS_STDIO 1
#define HAS_PRINTF 0
typedef unsigned long long CORE_TICKS;

#ifndef COMPILER_VERSION
#ifdef __clang__
#define COMPILER_VERSION "clang " __clang_version__
#elif defined(__GNUC__)
#define COMPILER_VERSION "GCC" __VERSION__
#endif
#endif
#ifndef COMPILER_FLAGS
#define COMPILER_FLAGS "-O2"
#endif
#define MEM_LOCATION "HEAP"

typedef signed short ee_s16;
typedef unsigned short ee_u16;
typedef signed int ee_s32;
typedef double ee_f32;
typedef unsigned char ee_u8;
typedef unsigned int ee_u32;
/* Pointer-sized, never converted to or from a pointer (check_data_types only takes its size). */
typedef struct { unsigned char bytes[sizeof(void *)]; } ee_ptr_int;
typedef size_t ee_size_t;
/* Upstream: (void *)(4 + (((ee_ptr_int)(x) - 1) & ~3)), i.e. x rounded up to 4 through an integer.
 * The same rounding as pointer arithmetic, so a capability keeps its tag and bounds. */
#define align_mem(x) ((void *)((char *)(x) + ((-(unsigned long)(x)) & 3UL)))

#define SEED_METHOD SEED_VOLATILE
#define MEM_METHOD MEM_MALLOC
#define MULTITHREAD 1
#define USE_PTHREAD 0
#define USE_FORK 0
#define USE_SOCKET 0
#define MAIN_HAS_NOARGC 1
#define MAIN_HAS_NORETURN 0

#ifndef ITERATIONS
#error "ITERATIONS must be fixed at build time"
#endif

extern ee_u32 default_num_contexts;
typedef struct CORE_PORTABLE_S {
  ee_u8 portable_id;
} core_portable;
void portable_init(core_portable *p, int *argc, char *argv[]);
void portable_fini(core_portable *p);
int ee_printf(const char *fmt, ...);

#if !defined(PROFILE_RUN) && !defined(PERFORMANCE_RUN) && !defined(VALIDATION_RUN)
#if (TOTAL_DATA_SIZE == 1200)
#define PROFILE_RUN 1
#elif (TOTAL_DATA_SIZE == 2000)
#define PERFORMANCE_RUN 1
#else
#define VALIDATION_RUN 1
#endif
#endif
#endif /* CORE_PORTME_H */
