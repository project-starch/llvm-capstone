/* What apr_buckets_alloc.c needs from apr-util, freestanding, in place of the
 * five headers it includes -- and in place of the ten that apr_buckets.h pulls
 * in behind them.
 *
 * The same seam apr_shim.h is for APR's pools: upstream's allocator byte for
 * byte, and one header that answers what it asks for.
 *
 * THE NODE GEOMETRY IS CARRIED, NOT GUESSED. SMALL_NODE_SIZE is
 * APR_BUCKET_ALLOC_SIZE plus the node header, and APR_BUCKET_ALLOC_SIZE is
 * 2*sizeof(union apr_bucket_structs) -- a union over the bucket TYPE zoo, which
 * lives in apr_buckets.h and not in the file being ported. A shim that invented
 * that size would silently change the allocator's block layout, its small/large
 * split and therefore which frees go to the freelist at all. So the five
 * structs and the union are transcribed verbatim from apr-util 1.6.3's
 * apr_buckets.h, and everything they reference that this file never
 * dereferences is an opaque pointer.
 */
#ifndef APR_BUCKET_SHIM_H
#define APR_BUCKET_SHIM_H

#include "apr_shim.h"
#include "apr_allocator.h"
#include "apr_pools.h"

#define APU_DECLARE_NONSTD(type)  type
#define APU_DECLARE(type)         type
/* <stdlib.h>, which the port's freestanding patch removes: the one thing the
 * allocator takes from it. Hosted, this is libc's; in a domain the port's
 * services supply it, ending the run with a code. */
_Noreturn void abort(void);

/* APR_HAS_MMAP decides whether apr_bucket_mmap is a union member and whether
 * apr_bucket_file carries can_mmap. Neither is the union's largest member --
 * apr_bucket is -- so the value does not move APR_BUCKET_ALLOC_SIZE. That is an
 * assertion the build checks rather than a claim this comment makes: see
 * bucket-geometry.c. */
#ifndef APR_HAS_MMAP
#define APR_HAS_MMAP 1
#endif

/* apr_version.h. The value matters: apr_bucket_alloc_aligned_floor takes a
 * different arm below 1.6.0, so this is the pinned APR version and not a
 * convenience. It must track ports/apr/fetch-apr.sh. */
#define APR_MAJOR_VERSION 1
#define APR_MINOR_VERSION 7
#define APR_PATCH_VERSION 4
#define APR_VERSION_AT_LEAST(major, minor, patch)                              \
  (((major) < APR_MAJOR_VERSION) ||                                            \
   ((major) == APR_MAJOR_VERSION && (minor) < APR_MINOR_VERSION) ||            \
   ((major) == APR_MAJOR_VERSION && (minor) == APR_MINOR_VERSION &&            \
    (patch) <= APR_PATCH_VERSION))

typedef long apr_off_t;

/* Opaque: apr_buckets_alloc.c stores and compares these and never dereferences
 * one, so their contents cannot affect it. Their SIZE can, because they sit in
 * the union, and a pointer is a pointer. */
typedef struct apr_bucket_type_t apr_bucket_type_t;
typedef struct apr_mmap_t apr_mmap_t;
typedef struct apr_file_t apr_file_t;

typedef struct apr_bucket_alloc_t apr_bucket_alloc_t;
typedef struct apr_bucket apr_bucket;

/* apr_ring.h, the one macro the bucket struct uses. */
#define APR_RING_ENTRY(elem)                                                   \
  struct {                                                                     \
    struct elem *next;                                                         \
    struct elem *prev;                                                         \
  }

/* ---- verbatim from apr-util 1.6.3 include/apr_buckets.h ------------------ */
struct apr_bucket {
    APR_RING_ENTRY(apr_bucket) link;
    const apr_bucket_type_t *type;
    apr_size_t length;
    apr_off_t start;
    void *data;
    void (*free)(void *e);
    apr_bucket_alloc_t *list;
};
typedef struct apr_bucket_refcount apr_bucket_refcount;
struct apr_bucket_refcount {
    int refcount;
};
typedef struct apr_bucket_heap apr_bucket_heap;
struct apr_bucket_heap {
    apr_bucket_refcount refcount;
    char *base;
    apr_size_t alloc_len;
    void (*free_func)(void *data);
};
typedef struct apr_bucket_pool apr_bucket_pool;
struct apr_bucket_pool {
    apr_bucket_heap heap;
    const char *base;
    apr_pool_t *pool;
    apr_bucket_alloc_t *list;
};
#if APR_HAS_MMAP
typedef struct apr_bucket_mmap apr_bucket_mmap;
struct apr_bucket_mmap {
    apr_bucket_refcount refcount;
    apr_mmap_t *mmap;
};
#endif
typedef struct apr_bucket_file apr_bucket_file;
struct apr_bucket_file {
    apr_bucket_refcount refcount;
    apr_file_t *fd;
    apr_pool_t *readpool;
#if APR_HAS_MMAP
    int can_mmap;
#endif
    apr_size_t read_size;
};
typedef union apr_bucket_structs apr_bucket_structs;
union apr_bucket_structs {
    apr_bucket b;
    apr_bucket_heap heap;
    apr_bucket_pool pool;
#if APR_HAS_MMAP
    apr_bucket_mmap mmap;
#endif
    apr_bucket_file file;
};
#define APR_BUCKET_ALLOC_SIZE  APR_ALIGN_DEFAULT(2*sizeof(apr_bucket_structs))
/* ---- end verbatim -------------------------------------------------------- */

APU_DECLARE_NONSTD(apr_bucket_alloc_t *) apr_bucket_alloc_create(apr_pool_t *p);
APU_DECLARE_NONSTD(apr_bucket_alloc_t *) apr_bucket_alloc_create_ex(apr_allocator_t *a);
APU_DECLARE_NONSTD(void) apr_bucket_alloc_destroy(apr_bucket_alloc_t *list);
APU_DECLARE_NONSTD(apr_size_t) apr_bucket_alloc_aligned_floor(apr_bucket_alloc_t *list, apr_size_t size);
APU_DECLARE_NONSTD(void *) apr_bucket_alloc(apr_size_t size, apr_bucket_alloc_t *list);
APU_DECLARE_NONSTD(void) apr_bucket_free(void *block);

#endif
