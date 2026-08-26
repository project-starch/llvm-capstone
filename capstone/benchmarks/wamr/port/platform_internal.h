/* WAMR platform layer for a Capstone domain.
 *
 * The contract is core/shared/platform/include/platform_api_vmcore.h: 24 os_*
 * entry points, of which the RTOS ports implement about twelve and map the rest
 * to macros here. nuttx (478 lines) and riot (693) are the templates; this is
 * smaller than either because a domain is single-threaded and has no syscalls.
 *
 * WHAT IS DELIBERATE, not a shortcut:
 *
 *   os_mmap carves from a STATIC ARENA. A domain has no mmap, and WAMR's own
 *   embedded configuration (WASM_ENABLE_GLOBAL_HEAP_POOL) does exactly this. The
 *   nesting that follows -- one region handed out in software, every sub-block
 *   inheriting its capability -- is not an artefact of the port. It IS the thing
 *   the corpus measures, so the port must not accidentally remove it.
 *
 *   Mutexes are no-ops and threads are a single constant id. Single-threaded by
 *   construction; a domain has one core and no scheduler.
 */
#ifndef _PLATFORM_INTERNAL_H
#define _PLATFORM_INTERNAL_H

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define BH_PLATFORM_CAPSTONE

/* Single-threaded: every handle is a value, never a pointer to kernel state. */
typedef int korp_tid;
typedef int korp_mutex;
typedef int korp_cond;
typedef int korp_thread;
typedef int korp_rwlock;
typedef int korp_sem;

#define OS_THREAD_MUTEX_INITIALIZER 0
#define BH_THREAD_DEFAULT_PRIORITY 0
#define BH_APPLET_PRESERVED_STACK_SIZE (2 * BH_KB)

/* 4 KiB, matching the module's own page size. Not queried: there is nothing to
   query, and a wrong constant here would silently mis-size linear memory. */
#define os_getpagesize() 4096

int capstone_wamr_printf(const char *fmt, ...);
int capstone_wamr_vprintf(const char *fmt, va_list ap);
#define os_printf capstone_wamr_printf
#define os_vprintf capstone_wamr_vprintf

#define BH_HAS_DLFCN 0
#define BH_TIME_T_MAX 0xffffffff
#define CONFIG_HAS_ISATTY 0

/* No filesystem. The types must exist because the vmcore references them even
   when WASM_ENABLE_LIBC_WASI is off; nothing dereferences them. */
typedef int os_file_handle;
typedef void *os_dir_stream;
typedef int os_raw_file_handle;

/* The extension header declares poll/socket entry points unconditionally, so the
   TYPES must exist even with WASI off. nuttx does the same; nothing in a domain
   build ever calls them. */
typedef struct {
    int fd;
    short events;
    short revents;
} os_poll_file_handle;
typedef unsigned long os_nfds_t;

/* The vmcore calls this from a static inline in platform_api_vmcore.h, so it must
   be visible here rather than in the .c file. */
static inline os_file_handle
os_get_invalid_handle(void)
{
    return -1;
}

#endif /* _PLATFORM_INTERNAL_H */
