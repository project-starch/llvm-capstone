/* What slabs.c needs from memcached.h, freestanding, in place of that 1114-line
 * header and everything it drags in.
 *
 * The same seam apr_shim.h is for APR and ngx_shim.h for nginx: upstream's
 * allocator byte for byte, and one header that answers what it asks for. The
 * census (../README.md) counted what slabs.c reaches into memcached.h for --
 * the item struct, settings, the stats callback type and a handful of
 * constants -- and this header carries exactly those, each upstream's text
 * cited to its line in the 1.6.45 tree, except the knob marked as such.
 * Safe to include from any program that links the allocator; the system-header
 * replacements slabs.c also needs are in mc_slabs_libc.h, for slabs.c alone. */
#ifndef MC_SLABS_SHIM_H
#define MC_SLABS_SHIM_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "mc_pthread_shim.h"

/* Marks the two regions the freestanding compile does not carry -- the stats
 * formatter and huge-page preallocation -- which the port's first patch gates. */
#define MC_PORT 1

/* memcached ships with -DNDEBUG (Makefile.am:92); this is the same binary. */
#ifndef NDEBUG
#error "memcached builds with -DNDEBUG (Makefile.am:92); pass it here too"
#endif
#define assert(x) ((void)0)

/* ---- memcached.h: constants and types slabs.c reads --------------------- */
#define POWER_SMALLEST 1                                        /* :118 */
#define POWER_LARGEST  256 /* actual cap is 255 */               /* :119 */
#define SLAB_GLOBAL_PAGE_POOL 0 /* magic slab class for storing pages for reassignment */ /* :120 */
/* Upstream aligns chunk sizes to 8 (:121). The port passes 16: a Sublet region
 * must be aligned and sized to whole capabilities, and a chunk is a region.
 * Left at 8 the class table is upstream's, which is what a census wants. */
#ifndef CHUNK_ALIGN_BYTES
#define CHUNK_ALIGN_BYTES 8
#endif
#define MAX_NUMBER_OF_SLAB_CLASSES (63 + 1)                     /* :123 */

typedef unsigned int rel_time_t;                                /* logger.h:14 */
#ifdef LARGE_CLIENT_FLAGS                                       /* :96-101 */
typedef uint64_t client_flags_t;
#else
typedef uint32_t client_flags_t;
#endif

#define ITEM_LINKED 1                                           /* :582-599 */
#define ITEM_CAS 2
#define ITEM_SLABBED 4
#define ITEM_FETCHED 8
#define ITEM_ACTIVE 16
#define ITEM_CHUNKED 32
#define ITEM_CHUNK 64
#define ITEM_HDR 128
#define ITEM_CFLAGS 256
#define ITEM_TOKEN_SENT 512

/**
 * Structure for storing items within memcached.
 */
typedef struct _stritem {                                       /* :613-637 */
    /* Protected by LRU locks */
    struct _stritem *next;
    struct _stritem *prev;
    /* Rest are protected by an item lock */
    struct _stritem *h_next;    /* hash chain next */
    rel_time_t      time;       /* least recent access */
    rel_time_t      exptime;    /* expire time */
    int             nbytes;     /* size of data */
    unsigned short  refcount;
    uint16_t        it_flags;   /* ITEM_* above */
    uint8_t         slabs_clsid;/* which slab class we're in */
    uint8_t         nkey;       /* key length, w/terminating null and padding */
    /* this odd type prevents type-punning issues when we do
     * the little shuffle to save space when not using CAS. */
    union {
        uint64_t cas;
        char end;
    } data[];
    /* if it_flags & ITEM_CAS we have 8 bytes CAS */
    /* then null-terminated key */
    /* then " flags length\r\n" (no terminating null) */
    /* then data with terminating \r\n (no terminating null; it's binary!) */
} item;

/* Header when an item is actually a chunk of another item. */
typedef struct _strchunk {                                      /* :661-673 */
    struct _strchunk *next;     /* points within its own chain. */
    struct _strchunk *prev;     /* can potentially point to the head. */
    struct _stritem  *head;     /* always points to the owner chunk */
    int              size;      /* available chunk space in bytes */
    int              used;      /* chunk space used */
    int              nbytes;    /* used. */
    unsigned short   refcount;  /* used? */
    uint16_t         it_flags;  /* ITEM_* above. */
    uint8_t          slabs_clsid; /* Same as above. */
    uint8_t          orig_clsid; /* For obj hdr chunks slabs_clsid is fake. */
    char data[];
} item_chunk;

#define ITEM_clsid(item) ((item)->slabs_clsid & ~(3<<6))        /* :154 */
/* :687-690, the build without NEED_ALIGN */
#define ITEM_schunk(item) ((char*) &((item)->data) + (item)->nkey + 1 \
         + (((item)->it_flags & ITEM_CFLAGS) ? sizeof(client_flags_t) : 0) \
         + (((item)->it_flags & ITEM_CAS) ? sizeof(uint64_t) : 0))

/* :470, the fields slabs.c reads plus factor, which memcached.c passes to
 * slabs_init from here. Types are upstream's; the port defines the object with
 * upstream's defaults (memcached.c:settings_init) in src/shared/services.c. */
struct settings {
    size_t maxbytes;
    int verbose;
    double factor;            /* chunk size growth factor */
    int chunk_size;
    int item_size_max;        /* Maximum item size */
    int slab_chunk_size_max;  /* Upper end for chunks within slab pages. */
    int slab_page_size;     /* Slab's page units. */
    bool slab_reassign;     /* Whether or not slab reassignment is allowed */
};
extern struct settings settings;                                /* :580 */

typedef void (*ADD_STAT)(const char *key, const uint16_t klen,   /* :208 */
                         const char *val, const uint32_t vlen,
                         const void *cookie);

/* ---- trace.h: the probes of a build without DTrace, :59-68 -------------- */
#define MEMCACHED_SLABS_ALLOCATE(arg0, arg1, arg2)
#define MEMCACHED_SLABS_ALLOCATE_ENABLED() (0)
#define MEMCACHED_SLABS_ALLOCATE_FAILED(arg0)
#define MEMCACHED_SLABS_ALLOCATE_FAILED_ENABLED() (0)
#define MEMCACHED_SLABS_FREE(arg0, arg1)
#define MEMCACHED_SLABS_FREE_ENABLED() (0)
#define MEMCACHED_SLABS_SLABCLASS_ALLOCATE(arg0)
#define MEMCACHED_SLABS_SLABCLASS_ALLOCATE_ENABLED() (0)
#define MEMCACHED_SLABS_SLABCLASS_ALLOCATE_FAILED(arg0)
#define MEMCACHED_SLABS_SLABCLASS_ALLOCATE_FAILED_ENABLED() (0)
#endif
