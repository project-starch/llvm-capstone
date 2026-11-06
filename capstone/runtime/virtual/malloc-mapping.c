/* Mapping authority for the mallocng.
 * All ordinary allocator state remains in musl. No per-object records. */
#include <errno.h>
#include <stdint.h>
#include <sys/mman.h>
#include <string.h>
#include "vm.h"
#include <heap_sublet.h>

struct heap_mid *__malloc_page_root[512];
static unsigned index_mutex;
static void index_lock(void)
{
    while (__atomic_exchange_n(&index_mutex, 1, __ATOMIC_ACQUIRE))
        if (__capstone_vm_wait()) __builtin_trap();
}
static void index_unlock(void)
{ __atomic_store_n(&index_mutex, 0, __ATOMIC_RELEASE); }

/* Tables are initialized once, before an object from their mapping can be
 * published, and retained for process lifetime. Valid-object lookups therefore
 * never race table creation/destruction. A page entry changes only after its
 * mapping has no live allocations. Mapping changes serialize here, separately
 * from mallocng's own locks; no lookup takes this lock. */
static struct heap_leaf *leaf_for(size_t a, int create)
{
    struct heap_mid *mid = __malloc_page_root[a >> 30];
    if (!mid && create) {
        mid = mmap(0, sizeof *mid, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (mid == MAP_FAILED) return 0;
        __malloc_page_root[a >> 30] = mid;
    }
    if (!mid) return 0;
    struct heap_leaf *leaf = mid->leaf[(a >> 21) & 511];
    if (!leaf && create) {
        leaf = mmap(0, sizeof *leaf, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (leaf == MAP_FAILED) return 0;
        mid->leaf[(a >> 21) & 511] = leaf;
    }
    return leaf;
}

#if CAPSTONE_HEAP_MAPPING_REUSE
/* In-domain reuse of whole group mappings (default off).
 *
 * Without it every group mapping mallocng releases is a munmap, and the next same-size group is a fresh
 * mapping: two context exits. A launcher-side pool (CAPSTONE_VM_MAPCACHE) saves the kernel work behind them,
 * but not the exits. On SPEC gcc train those are 24,107 MAP and 22,421 UNMAP exits of about 51 K. On silicon
 * an exit costs about 20 ms outside the context.
 *
 * Each mapping then carries one more lifetime. The holder is the mapping-wide child of the grant: a derived
 * child carries MANAGE, the grant need not. The owner, the parent of every object, becomes the holder's child.
 * Releasing a mapping into the cache:
 *   - ends the owner's whole subtree with one CREVOKE (heap_end), so every object, nested allocation and
 *     capability derived from them is stale before the bytes can be reissued;
 *   - clears the page index, so heap_access on a stale address traps as for an unmapped one;
 *   - keeps the mapping.
 * The next mapping of exactly the same length, at an alignment that length needs, takes a cached one:
 *   - its bytes are zeroed with scalar stores, which also clear every tag (mallocng treats a fresh mapping as
 *     zero);
 *   - a fresh owner is issued from the same holder and published in the index.
 * The cache is bounded by entries and bytes. Past either bound, or for a mapping it does not know, the path is
 * the original munmap. The kernel's retirement of a mapping ends the holder with everything else. */
#ifdef CAPSTONE_HEAP_MAPPING_REUSE_TEST_LIVE   /* saturation tests only: a small table, a power of two */
#define MR_LIVE ((unsigned)CAPSTONE_HEAP_MAPPING_REUSE_TEST_LIVE)
#else
#define MR_LIVE 16384U          /* live mappings tracked (holders), a power of two */
#endif
#define MR_FREE 4096U           /* cached mappings (region pieces are not counted against MR_FREE_BYTES) */
#define MR_FREE_BYTES (64UL << 20)
/* piece: cut from a heap region, never unmapped on its own.
 * noreuse: an mprotect touched the mapping, so its pages may no longer be PROT_READ|PROT_WRITE. It is never cached:
 * the reuse path's memset would fault on a page it cannot write (review of #306, P2). A whole mapping is released by
 * munmap, a piece is dropped. */
struct mr_rec { size_t base, bytes; void *holder; unsigned piece, noreuse; };
static struct mr_rec mr_live[MR_LIVE], mr_free[MR_FREE];
static unsigned mr_nfree, mr_nlive; static size_t mr_free_bytes;
unsigned long __heap_reuse_hits, __heap_reuse_puts, __heap_reuse_spills, __heap_reuse_untracked,
              __heap_reuse_protected, __heap_reuse_evicted;
static unsigned mr_slot(size_t base) { return (unsigned)((base >> 12) * 2654435761UL) & (MR_LIVE - 1); }
static struct mr_rec *mr_find(size_t base)
{
    for (unsigned i = mr_slot(base), k = 0; k < MR_LIVE; i = (i + 1) & (MR_LIVE - 1), ++k) {
        if (mr_live[i].base == base) return &mr_live[i];
        if (!mr_live[i].base) return 0;
    }
    return 0;
}
static void mr_remove(struct mr_rec *r)
{   /* Backward-shift deletion keeps every probe chain intact. */
    unsigned i = (unsigned)(r - mr_live), j = i;
    /* At most MR_LIVE - 1 other slots: with every slot occupied there is no empty one to stop at (P1). */
    for (unsigned n = 1; n < MR_LIVE; ++n) {
        j = (j + 1) & (MR_LIVE - 1);
        if (!mr_live[j].base) break;
        unsigned k = mr_slot(mr_live[j].base);
        if ((j > i && (k <= i || k > j)) || (j < i && k <= i && k > j)) { mr_live[i] = mr_live[j]; i = j; }
    }
    mr_live[i] = (struct mr_rec){0, 0, 0, 0, 0};
    --mr_nlive;
}
static int mr_record(size_t base, size_t bytes, void *holder, unsigned piece)
{
    for (unsigned i = mr_slot(base), k = 0; k < MR_LIVE; i = (i + 1) & (MR_LIVE - 1), ++k)
        if (!mr_live[i].base) { mr_live[i] = (struct mr_rec){base, bytes, holder, piece, 0}; ++mr_nlive; return 0; }
    ++__heap_reuse_untracked;
    return -1;   /* table full: this mapping is released by munmap, as without the switch */
}
/* Give every cached whole mapping back to the kernel; cached pieces stay (a piece is never unmapped alone). Returns
 * how many it gave back. Called with index_mutex held. */
static unsigned mr_evict_all(void)
{
    unsigned gave = 0;
    for (unsigned i = mr_nfree; i-- > 0;) {
        if (mr_free[i].piece) continue;
        struct mr_rec e = mr_free[i];
        mr_free[i] = mr_free[--mr_nfree]; mr_free_bytes -= e.bytes;
        munmap(e.holder, e.bytes);
        ++__heap_reuse_evicted; ++gave;
    }
    return gave;
}
#endif
#if CAPSTONE_HEAP_MAPPING_REUSE
/* The cursor of the owner published for page a, or 0 for a page of no live mapping. */
static size_t page_owner(size_t a)
{
    struct heap_leaf *leaf = leaf_for(a, 0);
    void *o = leaf ? leaf->owner[(a >> 12) & 511] : 0;
    return o ? __builtin_capstone_cap_get_cursor(o) : 0;
}
#endif
/* mprotect (mapping.c) reports every range it was asked to change, whether or not the service changed it. A tracked
 * mapping it overlaps is marked noreuse; a cached one it overlaps leaves the cache at once: a whole mapping is
 * unmapped, a piece is dropped (its bytes stay in the region, unused). */
void __heap_protected(size_t address, size_t bytes)
{
#if CAPSTONE_HEAP_MAPPING_REUSE && !defined(CAPSTONE_HEAP_MAPPING_REUSE_TEST_KEEP_PROTECTED)
    /* The live mappings the range overlaps come from the page index, page by page over the range only. A scan of the
     * whole table made all of it resident: about 200 pinned pages in every process, each one a fault exit. */
    size_t end = address + bytes < address ? (size_t)-1 : address + bytes;
    if (end > (1UL << 39)) end = 1UL << 39;
    index_lock();
    for (size_t a = address & -4096UL; a < end;) {
        size_t o = page_owner(a);
        if (!o) { a = leaf_for(a, 0) ? a + 4096 : (a | ((1UL << 21) - 1)) + 1; continue; }
        size_t b = a;
        while (b >= 4096 && page_owner(b - 4096) == o) b -= 4096;   /* the mapping's first page */
        struct mr_rec *r = mr_find(b);
        if (r) r->noreuse = 1;
        while (a < end && page_owner(a) == o) a += 4096;
    }
    for (unsigned i = mr_nfree; i-- > 0;)
        if (mr_free[i].base < address + bytes && address < mr_free[i].base + mr_free[i].bytes) {
            struct mr_rec e = mr_free[i];
            mr_free[i] = mr_free[--mr_nfree];
            if (!e.piece) { mr_free_bytes -= e.bytes; munmap(e.holder, e.bytes); }
            ++__heap_reuse_protected;
        }
    index_unlock();
#else
    (void)address; (void)bytes;
#endif
}

#if CAPSTONE_HEAP_MAPPING_REGION
#if !CAPSTONE_HEAP_MAPPING_REUSE
#error "CAPSTONE_HEAP_MAPPING_REGION needs CAPSTONE_HEAP_MAPPING_REUSE"
#endif
/* Group mappings of up to RG_PIECE_MAX are cut from 4 MiB heap regions instead of being mapped one by one (default
 * off). A fresh process otherwise pays one MAP exit per group mapping before the reuse cache has anything: perl -e 1
 * pays 51, op/array.t 214. A piece is a mapping in every other respect: its own holder (a child of the region
 * grant at the piece's offset) and owner, its own index entries, and the reuse cache's revocation on release. A
 * piece is never unmapped on its own: released pieces stay in the cache, whose byte bound does not count them. The
 * rest of a region that cannot fit the next piece is not used. The kernel retires a region with the process. */
#define RG_BYTES (4UL << 20)
#define RG_PIECE_MAX (256UL << 10)
static void *rg_raw; static size_t rg_next, rg_end;
unsigned long __heap_region_maps, __heap_region_pieces, __heap_piece_drops;
static void *rg_take(size_t bytes, size_t align)
{
    index_lock();
    size_t off = (rg_next + align - 1) & -align;
    if (!rg_raw || off + bytes > rg_end) {
        sublet_cap g;
        size_t ra = heap_granule(RG_BYTES); if (ra < 4096) ra = 4096;
        if (cap_vm_acquire(&g, RG_BYTES, ra, PROT_READ|PROT_WRITE, 6, CAP_VM_HEAP)) { index_unlock(); return 0; }
        rg_raw = cap_vm_copyable(&g);
        rg_next = __builtin_capstone_cap_get_cursor(rg_raw); rg_end = rg_next + RG_BYTES;
        ++__heap_region_maps;
        off = (rg_next + align - 1) & -align;
        if (off + bytes > rg_end) { index_unlock(); return 0; }
    }
    void *at = (char *)rg_raw + (ptrdiff_t)(off - __builtin_capstone_cap_get_cursor(rg_raw));
    rg_next = off + bytes; ++__heap_region_pieces;
    index_unlock();
    return at;
}
#endif

void *__heap_mapping(size_t n)
{
    if (!n || n > CV_MAX_REGION_BYTES) { errno=ENOMEM; return MAP_FAILED; }
    size_t bytes=heap_round((n+4095)&-4096UL);
    HEAP_POLICY_SYMBOL(); /* Link-time check: libc and SDK use the same policy. */
    sublet_cap grant;
    size_t align=heap_granule(bytes);
    if (align < 4096) align=4096;
#if CAPSTONE_HEAP_MAPPING_REUSE
    index_lock();
    for (unsigned i = mr_nfree; i-- > 0;) {
        if (mr_free[i].bytes != bytes || (mr_free[i].base & (align - 1))) continue;
        struct mr_rec e = mr_free[i];
        /* A piece is handed out only while tracked: an untracked one would be released by munmap, which would cut
         * a hole in its region. With the table full it stays cached and a whole mapping is used instead. */
        if (mr_record(e.base, e.bytes, e.holder, e.piece) && e.piece) continue;
        mr_free[i] = mr_free[--mr_nfree]; if (!e.piece) mr_free_bytes -= bytes;
#ifndef CAPSTONE_HEAP_MAPPING_REUSE_TEST_NO_ZERO   /* negative control of the reuse checks ONLY */
        memset(e.holder, 0, bytes);
#endif
        void *owner = heap_issue(e.holder, bytes);
        for (size_t a = e.base; a < e.base + bytes; a += 4096) {
            struct heap_leaf *leaf = leaf_for(a, 0);
            if (!leaf || leaf->owner[(a >> 12) & 511]) __builtin_trap();
            leaf->owner[(a >> 12) & 511] = owner;
        }
        ++__heap_reuse_hits;
        index_unlock();
        return owner;
    }
    index_unlock();
#endif
    void *raw=0; unsigned piece=0;
#if CAPSTONE_HEAP_MAPPING_REGION
    if (bytes <= RG_PIECE_MAX && mr_nlive < MR_LIVE && (raw=rg_take(bytes,align))) piece=1;
whole:
#endif
    if (!piece) {
        if (cap_vm_acquire(&grant,bytes,align,PROT_READ|PROT_WRITE,6,CAP_VM_HEAP)) {
#if CAPSTONE_HEAP_MAPPING_REUSE
            /* Out of memory with whole mappings of other sizes cached: give them back and try once more (P3). */
            index_lock();
            unsigned gave = mr_evict_all();
            index_unlock();
            if (!gave || cap_vm_acquire(&grant,bytes,align,PROT_READ|PROT_WRITE,6,CAP_VM_HEAP))
#endif
            return MAP_FAILED;
        }
        raw=cap_vm_copyable(&grant);
    }
    size_t base=__builtin_capstone_cap_get_cursor(raw);
    if (base >= (1UL<<39) || bytes > (1UL<<39)-base) {
        if (!piece) munmap(raw,bytes);
        errno=ENOMEM; return MAP_FAILED;
    }
    index_lock();
    /* Provision the entire index before publishing any owner entries. */
    for (size_t a=base; a<base+bytes; a+=4096) {
        if (!leaf_for(a,1)) {
            int e=errno;
            index_unlock(); if (!piece) munmap(raw,bytes); errno=e; return MAP_FAILED;
        }
    }
    /* mmap's returned capability need not carry MANAGE. This one mapping-wide
     * child does, and becomes the parent of all objects within the mapping. */
#if CAPSTONE_HEAP_MAPPING_REUSE
    void *holder=heap_issue(raw,bytes), *owner=heap_issue(holder,bytes);
    if (mr_record(base,bytes,holder,piece)) {   /* untracked: released by munmap */
#if CAPSTONE_HEAP_MAPPING_REGION
        if (piece) {   /* never untracked (see the reuse path): end its lifetimes, leave it unused, map whole */
            heap_end(holder, owner);
            index_unlock(); ++__heap_piece_drops; piece=0; goto whole;
        }
#endif
    }
#else
    void *owner=heap_issue(raw,bytes);
#endif
    for (size_t a=base; a<base+bytes; a+=4096) {
        struct heap_leaf *leaf=leaf_for(a,0);
        if (leaf->owner[(a>>12)&511]) __builtin_trap();
        leaf->owner[(a>>12)&511]=owner;
    }
    index_unlock();
    return owner;
}

int __heap_unmapping(void *owner,size_t n)
{
    size_t bytes=heap_round((n+4095)&-4096UL);
    size_t base=__builtin_capstone_cap_get_cursor(owner);
    index_lock();
#if CAPSTONE_HEAP_MAPPING_REUSE
    {
        struct mr_rec *r = mr_find(base);
        if (r && !r->noreuse && r->bytes == bytes && mr_nfree < MR_FREE && (r->piece || mr_free_bytes + bytes <= MR_FREE_BYTES)) {
#ifndef CAPSTONE_HEAP_MAPPING_REUSE_TEST_NO_REVOKE   /* positive control of the safety probe ONLY */
            heap_end(r->holder, owner);   /* every object, nested group and derivative of this mapping */
#endif
            for (size_t a=base; a<base+bytes; a+=4096) {
                struct heap_leaf *leaf=leaf_for(a,0);
                if (!leaf) __builtin_trap();
                leaf->owner[(a>>12)&511]=0;
            }
            mr_free[mr_nfree++] = *r; if (!r->piece) mr_free_bytes += bytes; mr_remove(r);
            ++__heap_reuse_puts;
            index_unlock();
            return 0;
        }
        if (r) {
#if CAPSTONE_HEAP_MAPPING_REGION
            if (r->piece) {   /* never unmapped alone: revoke, clear the index, drop it */
                heap_end(r->holder, owner);
                for (size_t a=base; a<base+bytes; a+=4096) leaf_for(a,0)->owner[(a>>12)&511]=0;
                mr_remove(r); ++__heap_piece_drops; index_unlock(); return 0;
            }
#endif
            mr_remove(r); ++__heap_reuse_spills;
        }
    }
#endif
    int rc=munmap(owner,bytes);
    if (!rc) {
        for (size_t a=base; a<base+bytes; a+=4096) {
            struct heap_leaf *leaf=leaf_for(a,0);
            if (!leaf) __builtin_trap();
            leaf->owner[(a>>12)&511]=0;
        }
    }
    index_unlock();
    return rc;
}

/* Same map-copy-unmap behavior as the baseline public mremap, retaining its
 * failure guarantee and capability-preserving memcpy. */
void *__heap_remapping(void *p,size_t old,size_t n)
{
    if (((old+4095)&-4096UL)==((n+4095)&-4096UL)) return p;
    void *q=__heap_mapping(n);
    if (q==MAP_FAILED) return q;
    memcpy(q,p,old<n?old:n);
    if (__heap_unmapping(p,old)) {
        int e=errno; __heap_unmapping(q,n); errno=e; return MAP_FAILED;
    }
    return q;
}
