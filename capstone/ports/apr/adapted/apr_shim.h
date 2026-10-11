/* What apr_pools.c needs from APR, freestanding, in place of the fourteen headers it includes.
 *
 * The same seam ngx_shim.h is for nginx: upstream's allocator byte for byte, and one header that
 * answers what it asks for. APR generates apr.h and apr_private.h from configure, which is a host
 * build we do not have and do not want, so the types and the feature flags are written out here.
 *
 * The two flags that decide how much of the allocator survives are set to zero on purpose, and
 * the census says why: mmap, munmap and sysconf sit behind APR_ALLOCATOR_USES_MMAP, and every
 * mutex and atomic sits behind APR_HAS_THREADS.
 */
#ifndef APR_SHIM_H
#define APR_SHIM_H

#include <stddef.h>
#include <stdint.h>

/* ---- what configure would have written ---------------------------------- */
#define APR_HAS_THREADS            0
#define APR_ALLOCATOR_USES_MMAP    0
#define APR_ALLOCATOR_GUARD_PAGES  0
#define APR_POOL_DEBUG             0
#define APR_POOL_CONCURRENCY_CHECK 0
/* Zero, not one, and the difference is the whole point: apr_pools.c would include the host's
   <stdlib.h> and <unistd.h>, which a freestanding target does not have. What it wants from them
   is declared at the bottom of this file instead. */
#define APR_HAVE_STDLIB_H          0
#define APR_HAVE_STRING_H          0
#define APR_HAVE_UNISTD_H          0
#define HAVE_VALGRIND              0
#define APR_SIZEOF_VOIDP           (int) sizeof(void *)

#define APR_DECLARE(type)          type
#define APR_DECLARE_NONSTD(type)   type
#define APR_THREAD_FUNC
#define APR_INLINE                 inline
#define APR_STRINGIFY(n)           APR_STRINGIFY_HELPER(n)
#define APR_STRINGIFY_HELPER(n)    #n

typedef size_t     apr_size_t;
typedef ptrdiff_t  apr_ssize_t;
typedef int        apr_status_t;
typedef int32_t    apr_int32_t;
typedef uint32_t   apr_uint32_t;
typedef int64_t    apr_int64_t;
typedef uint64_t   apr_uint64_t;
typedef int        apr_os_proc_t;

#define APR_SUCCESS    0
#define APR_ENOPOOL    20000
#define APR_EINVAL     20001
#define APR_ENOMEM     20002
#define APR_ALIGN(size, boundary) \
    (((size) + ((boundary) - 1)) & ~((boundary) - 1))
/* Upstream rounds every apr_palloc result and every header size to 8. The pools port sets this
   to 16 (-DAPR_ALIGN_DEFAULT_BOUNDARY=16): a capability is 16 bytes and must be stored 16-aligned,
   and a 24-byte apr_palloc followed by a cleanup_t -- two function-pointer capabilities -- would
   otherwise trap on the store. The census keeps upstream's 8, so its numbers stay upstream's. */
#ifndef APR_ALIGN_DEFAULT_BOUNDARY
#define APR_ALIGN_DEFAULT_BOUNDARY 8
#endif
#define APR_ALIGN_DEFAULT(size) APR_ALIGN(size, APR_ALIGN_DEFAULT_BOUNDARY)

/* ---- what apr_pools.c needs for the three services that share its file ----
 *
 * apr_pools.c is not only an allocator. The same translation unit carries the subprocess chain,
 * the userdata hash and apr_psprintf, and each wants types from a header this shim replaced. They
 * are given opaque types and declarations rather than implementations, because a port does not
 * touch them and a census must not pretend they are not there.
 */
#include <stdarg.h>                      /* a compiler header, present freestanding */

/* At FILE scope, and it has to be. A struct tag first named inside a prototype is scoped to that
   prototype, so four declarations below would each have declared a different apr_pool_t and the
   compiler would have refused the calls with the two types printed identically. */
struct apr_pool_t;

typedef unsigned char apr_byte_t;
#define APR_UINT32_MAX  ((apr_uint32_t) 0xFFFFFFFFU)

typedef struct apr_proc_t apr_proc_t;    /* the subprocess chain */
typedef enum { APR_KILL_NEVER = 0, APR_KILL_ALWAYS, APR_KILL_AFTER_TIMEOUT,
               APR_JUST_WAIT, APR_KILL_ONLY_ONCE } apr_kill_conditions_e;
apr_status_t apr_proc_wait(apr_proc_t *, int *, int *, int);
apr_status_t apr_proc_kill(apr_proc_t *, int);

typedef struct apr_hash_t apr_hash_t;    /* the userdata table */
#define APR_HASH_KEY_STRING  (-1)
apr_hash_t *apr_hash_make(struct apr_pool_t *);
void apr_hash_set(apr_hash_t *, const void *, apr_ssize_t, const void *);
void *apr_hash_get(apr_hash_t *, const void *, apr_ssize_t);

typedef struct apr_vformatter_buff_t {   /* apr_psprintf */
    char *curpos;
    char *endpos;
} apr_vformatter_buff_t;
int apr_vformatter(int (*flush_func)(apr_vformatter_buff_t *b),
                   apr_vformatter_buff_t *, const char *, va_list);

apr_status_t apr_atomic_init(struct apr_pool_t *);

/* The subprocess chain, which apr_pool_destroy walks. Nothing here is the allocator, and a port
   does not touch it, but it shares the translation unit and so has to name its types. */
typedef apr_int64_t apr_time_t;
#define APR_NOWAIT          0
#define APR_WAIT            1
#define APR_CHILD_NOTDONE   20003
#define SIGTERM             15
#define SIGKILL             9
void apr_sleep(apr_time_t);
char *apr_pstrdup(struct apr_pool_t *, const char *);

/* ---- the level below, which is what a port replaces ---------------------- */
void *malloc(size_t);
void  free(void *);
void *memcpy(void *, const void *, size_t);
void *memset(void *, int, size_t);
size_t strlen(const char *);

#endif
