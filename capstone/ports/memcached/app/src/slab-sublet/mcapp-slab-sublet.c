/* Sublet inside memcached's own allocators (the port plan's stretch S1).
 *
 * Patch 0006 puts the component port's lifetime hooks (ports/memcached/allocators, patch 0002)
 * into the application's slabs.c and cache.c, behind MC_CAPSTONE_SLAB_SUBLET. This file is what
 * sits under them in the server:
 *
 *   the adapter    the component's ledger, metadata heap and Sublet authority
 *                  (src/shared/leases.c, src/shared/metadata.c, src/allocators/sublet/authority.c),
 *                  compiled unchanged; leases.c takes the item layout from memcached.h
 *                  (mc_slabs_shim.h beside this file)
 *   the payload    64 MiB lent LINEAR by the Sublet heap (__capstone_sublet_malloc_linear): slab
 *                  pages are carved from its lower 48 MiB, cache objects from the rest
 *   the metadata   16 MiB from malloc: the ledger's records, the slab lists, cache control blocks
 *   one lock       around every hook. The slab hooks already run under slabs_lock, but cache.c is
 *                  called without it, and the ledger and metadata heap are unlocked statics
 *   the mode       MC_SLAB_SUBLET_MODE, required: 0 spatial (each chunk bounded to itself, no
 *                  revocation), 1 sublet (each release and issue revokes the chunk and mints a fresh
 *                  alias). Absent or anything else ends the server: no mode is ever assumed
 *   a report       one line on stderr at exit, only with MC_SLAB_SUBLET_REPORT=1, so the oracle's
 *                  stderr stays comparable with native's
 *   refusals       the adapter's codes (5xx) and this file's (9xx) as one line on stderr, then
 *                  exit 97: a run that hits one DIFFERS, it never passes */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "port.h"
#include "mcapp-slab-sublet.h"

unsigned long __capstone_sublet_malloc_linear(size_t n, capstone_cap_slot *out);

static pthread_mutex_t mcs_lock = PTHREAD_MUTEX_INITIALIZER;
static unsigned mcs_mode;

_Noreturn void mcp_fail(unsigned code)
{
    fprintf(stderr, "MCAPP-SLAB-SUBLET fail %u\n", code);
    fflush(stderr);
    _Exit(97);
}

static void mcs_report(void)
{
    struct mcp_header h;
    pthread_mutex_lock(&mcs_lock);
    mcp_stats(&h);
    pthread_mutex_unlock(&mcs_lock);
    fprintf(stderr, "MCAPP-SLAB-SUBLET mode=%u pages=%llu chunk_releases=%llu chunk_reuses=%llu "
                    "object_releases=%llu object_reuses=%llu backing_used=%llu metadata=%llu\n",
            mcs_mode, (unsigned long long)h.pages, (unsigned long long)h.chunk_releases,
            (unsigned long long)h.chunk_reuses, (unsigned long long)h.object_releases,
            (unsigned long long)h.object_reuses, (unsigned long long)h.backing_used,
            (unsigned long long)h.metadata);
}

void mcs_init(void)
{
    static capstone_cap_slot payload;
    const char *mode = getenv("MC_SLAB_SUBLET_MODE");
    const char *report = getenv("MC_SLAB_SUBLET_REPORT");

    if (!mode || (strcmp(mode, "0") && strcmp(mode, "1")))
        mcp_fail(903);
    mcs_mode = (unsigned)(mode[0] - '0');
    void *metadata = malloc(MCP_META_BYTES);
    if (!metadata)
        mcp_fail(902);
    if (!__capstone_sublet_malloc_linear(MCP_PAYLOAD_BYTES, &payload))
        mcp_fail(901);
    mcp_meta_init(metadata);
    /* The region is handed over once: the authority keeps it in its own slot from here on. */
    mcp_payload_init(payload.c);
    mcp_set_mode(mcs_mode);
    if (report && !strcmp(report, "1"))
        atexit(mcs_report);
}

#define MCS_LOCKED(type, call) \
    do { type r_; pthread_mutex_lock(&mcs_lock); r_ = (call); pthread_mutex_unlock(&mcs_lock); return r_; } while (0)
#define MCS_LOCKED_VOID(call) \
    do { pthread_mutex_lock(&mcs_lock); (call); pthread_mutex_unlock(&mcs_lock); } while (0)

void *mcs_page_backing(size_t size) { MCS_LOCKED(void *, mcp_page_backing(size)); }
void mcs_page_discard(void *page) { MCS_LOCKED_VOID(mcp_page_discard(page)); }
void *mcs_page_carve(void *page, unsigned id, uint32_t chunk_size, uint32_t perslab)
{
    MCS_LOCKED(void *, mcp_page_carve(page, id, chunk_size, perslab));
}
void *mcs_chunk_at(void *page, unsigned index) { MCS_LOCKED(void *, mcp_chunk_at(page, index)); }
void *mcs_chunk_issue(void *chunk) { MCS_LOCKED(void *, mcp_chunk_issue(chunk)); }
void *mcs_chunk_release(void *chunk, unsigned id) { MCS_LOCKED(void *, mcp_chunk_release(chunk, id)); }

void *mcs_object_backing(size_t size) { MCS_LOCKED(void *, mcp_object_backing(size)); }
void *mcs_object_issue(void *object) { MCS_LOCKED(void *, mcp_object_issue(object)); }
void *mcs_object_release(void *object) { MCS_LOCKED(void *, mcp_object_release(object)); }
void mcs_object_discard(void *object) { MCS_LOCKED_VOID(mcp_object_discard(object)); }

void *mcs_meta_calloc(size_t n, size_t size) { MCS_LOCKED(void *, mcp_meta_calloc(n, size)); }
void mcs_meta_free(void *p) { MCS_LOCKED_VOID(mcp_meta_free(p)); }
char *mcs_meta_strdup(const char *s) { MCS_LOCKED(char *, mcp_meta_strdup(s)); }
void **mcs_meta_grow_pointers(void **old, size_t count, size_t new_count)
{
    MCS_LOCKED(void **, mcp_meta_grow_pointers(old, count, new_count));
}
