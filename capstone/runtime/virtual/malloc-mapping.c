/* Mapping authority for the mallocng.
 * All ordinary allocator state remains in musl. No per-object records. */
#include <errno.h>
#include <stdint.h>
#include <sys/mman.h>
#include <string.h>
#include <sched.h>
#include <heap_sublet.h>

struct heap_mid *__malloc_page_root[512];
static unsigned index_mutex;
static void index_lock(void)
{
    while (__atomic_exchange_n(&index_mutex, 1, __ATOMIC_ACQUIRE))
        sched_yield();
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
/* In-process reuse of whole group mappings (default off): the trap route's port of the vexec route's cache
 * (alloc-lean eb73a2bf475c and 3e70344d8c3f), the measured Full SDK's HEAP_MAPPING_REUSE.
 *
 * Without it every group mapping mallocng releases is a munmap, and the next same-size group is a fresh mmap.
 * On the vexec route the holder, a child of the grant, was the owner's parent, and a put ended the owner's subtree
 * with the two-operand CREVOKE. That form is illegal under the Sublet format, and here the owner IS the kernel's
 * depth-1 block: a child reference cannot authorize munmap. So a put renews the owner in place:
 *   - owner' = CRENEW(owner). The record's generation advances and the run below it is released, so every object,
 *     nested allocation and capability derived from the old lifetime, and the old owner itself, stop resolving
 *     before the bytes can be reissued (the reference model's CRENEW; the semantics owner's ruling, 2026-11-25);
 *   - the kernel authorizes munmap and mprotect by the reference the program passes, and the renewed one is that
 *     reference (sublet-blocks-internal.h: "Requests always use the handle's slot, including after CRENEW"), so
 *     owner' is what an eviction unmaps;
 *   - CRENEW can fail without a fault: an untagged 0, with the old lifetime still live. Then the mapping is not
 *     cached; it is released by munmap with the old owner, as without the switch;
 *   - the page index is cleared, so heap_access on a stale address traps as for an unmapped one.
 * The next mapping of exactly the same length takes a cached one:
 *   - its bytes are zeroed by memset, as on the vexec route (mallocng treats a fresh mapping as zero; with
 *     MEMSET_CAP_ZERO the granules are null-capability stores, so tags are cleared either way);
 *   - owner' is published in the index.
 * The cache is bounded by entries and bytes. Past either bound, or for a mapping it does not track, or for one an
 * mprotect touched, the path is the original munmap. */
#ifdef CAPSTONE_HEAP_MAPPING_REUSE_TEST_LIVE   /* saturation tests only: a small table, a power of two */
#define MR_LIVE ((unsigned)CAPSTONE_HEAP_MAPPING_REUSE_TEST_LIVE)
#else
#define MR_LIVE 16384U          /* live mappings tracked, a power of two */
#endif
#define MR_FREE 512U            /* cached mappings */
#define MR_FREE_BYTES (64UL << 20)
/* noreuse: an mprotect touched the mapping, so its pages may no longer be PROT_READ|PROT_WRITE. It is released by
 * munmap, never cached: the reuse path's memset would fault on a page it cannot write. */
struct mr_rec { size_t base, bytes; void *owner; unsigned noreuse; };
static struct mr_rec mr_live[MR_LIVE], mr_free[MR_FREE];
static unsigned mr_nfree; static size_t mr_free_bytes;
unsigned long __heap_reuse_hits, __heap_reuse_puts, __heap_reuse_spills, __heap_reuse_untracked,
              __heap_reuse_protected, __heap_reuse_evicted, __heap_reuse_renew_failed;
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
    /* At most MR_LIVE - 1 other slots: with every slot occupied there is no empty one to stop at. */
    for (unsigned n = 1; n < MR_LIVE; ++n) {
        j = (j + 1) & (MR_LIVE - 1);
        if (!mr_live[j].base) break;
        unsigned k = mr_slot(mr_live[j].base);
        if ((j > i && (k <= i || k > j)) || (j < i && k <= i && k > j)) { mr_live[i] = mr_live[j]; i = j; }
    }
    mr_live[i] = (struct mr_rec){0, 0, 0, 0};
}
static int mr_record(size_t base, size_t bytes, void *owner)
{
    for (unsigned i = mr_slot(base), k = 0; k < MR_LIVE; i = (i + 1) & (MR_LIVE - 1), ++k)
        if (!mr_live[i].base) { mr_live[i] = (struct mr_rec){base, bytes, owner, 0}; return 0; }
    ++__heap_reuse_untracked;
    return -1;   /* table full: this mapping is released by munmap, as without the switch */
}
/* Give every cached mapping back to the kernel. Called with index_mutex held. */
static void mr_evict_all(void)
{
    while (mr_nfree) {
        struct mr_rec e = mr_free[--mr_nfree];
        mr_free_bytes -= e.bytes;
        munmap(e.owner, e.bytes);
        ++__heap_reuse_evicted;
    }
}
/* Give cached mappings back, from the front of the table, until one of `bytes` fits both bounds. Called with
 * index_mutex held. That is put order, oldest first, except that a reuse fills the gap it leaves with the last
 * (newest) entry. Without this a mapping that fits the byte bound by itself was refused whenever the cache already
 * held more than the bound minus its size (505.mcf_r test's 64,094,208-byte arcs_pointer_sorted, 3e70344d8c3f). */
static void mr_make_room(size_t bytes)
{
    while (mr_nfree && (mr_nfree >= MR_FREE || mr_free_bytes + bytes > MR_FREE_BYTES)) {
        struct mr_rec e = mr_free[0];
        memmove(mr_free, mr_free + 1, --mr_nfree * sizeof *mr_free);
        mr_free_bytes -= e.bytes;
        munmap(e.owner, e.bytes);
        ++__heap_reuse_evicted;
    }
}
/* The cursor of the owner published for page a, or 0 for a page of no live mapping. */
static size_t page_owner(size_t a)
{
    struct heap_leaf *leaf = leaf_for(a, 0);
    void *o = leaf ? leaf->owner[(a >> 12) & 511] : 0;
    return o ? __builtin_capstone_cap_get_cursor(o) : 0;
}
#endif
/* mprotect (mapping.c) reports every range it was asked to change, whether or not the kernel changed it. A tracked
 * mapping it overlaps is marked noreuse; a cached one it overlaps is unmapped at once. */
void __heap_protected(size_t address, size_t bytes)
{
#if CAPSTONE_HEAP_MAPPING_REUSE && !defined(CAPSTONE_HEAP_MAPPING_REUSE_TEST_KEEP_PROTECTED)
    /* The live mappings the range overlaps come from the page index, page by page over the range only. */
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
            mr_free[i] = mr_free[--mr_nfree]; mr_free_bytes -= e.bytes;
            munmap(e.owner, e.bytes);
            ++__heap_reuse_protected;
        }
    index_unlock();
#else
    (void)address; (void)bytes;
#endif
}

void *__heap_mapping(size_t n)
{
    if (!n || n > PTRDIFF_MAX - 8192) { errno=ENOMEM; return MAP_FAILED; }
    size_t bytes=heap_round((n+4095)&-4096UL);
    HEAP_POLICY_SYMBOL(); /* Link-time check: libc and SDK use the same policy. */
#if CAPSTONE_HEAP_MAPPING_REUSE
    index_lock();
    for (unsigned i = mr_nfree; i-- > 0;) {
        if (mr_free[i].bytes != bytes) continue;
        struct mr_rec e = mr_free[i];
        mr_free[i] = mr_free[--mr_nfree]; mr_free_bytes -= bytes;
        if (mr_record(e.base, e.bytes, e.owner)) { /* untracked from here: released by munmap */ }
#ifndef CAPSTONE_HEAP_MAPPING_REUSE_TEST_NO_ZERO   /* negative control of the reuse checks ONLY */
        memset(e.owner, 0, bytes);
#endif
        for (size_t a = e.base; a < e.base + bytes; a += 4096) {
            struct heap_leaf *leaf = leaf_for(a, 0);
            if (!leaf || leaf->owner[(a >> 12) & 511]) __builtin_trap();
            leaf->owner[(a >> 12) & 511] = e.owner;
        }
        ++__heap_reuse_hits;
        index_unlock();
        return e.owner;
    }
    index_unlock();
#endif
    void *raw=mmap(0,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
#if CAPSTONE_HEAP_MAPPING_REUSE
    if (raw == MAP_FAILED) {
        /* Out of memory with mappings of other sizes cached: give them back and try once more. */
        index_lock();
        unsigned had = mr_nfree;
        mr_evict_all();
        index_unlock();
        if (had) raw=mmap(0,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    }
#endif
    if (raw == MAP_FAILED) return raw;
    size_t base=__builtin_capstone_cap_get_cursor(raw);
    if (base >= (1UL<<39) || bytes > (1UL<<39)-base) {
        munmap(raw,bytes); errno=ENOMEM; return MAP_FAILED;
    }
    index_lock();
    /* Provision the entire index before publishing any owner entries. */
    for (size_t a=base; a<base+bytes; a+=4096) {
        if (!leaf_for(a,1)) {
            int e=errno;
            index_unlock(); munmap(raw,bytes); errno=e; return MAP_FAILED;
        }
    }
    /* Keep the kernel's depth-1 block as owner. A child reference cannot
     * authorize munmap under the trap-route mapping contract. */
    void *owner=raw;
#if CAPSTONE_HEAP_MAPPING_REUSE
    if (mr_record(base,bytes,owner)) { /* untracked: released by munmap */ }
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
        if (r && !r->noreuse && r->bytes == bytes && bytes <= MR_FREE_BYTES) mr_make_room(bytes);
        if (r && !r->noreuse && r->bytes == bytes && mr_nfree < MR_FREE && mr_free_bytes + bytes <= MR_FREE_BYTES) {
#ifndef CAPSTONE_HEAP_MAPPING_REUSE_TEST_NO_REVOKE   /* positive control of the safety probe ONLY */
            /* every object, nested group and derivative of this mapping, and the old owner, end here */
            void *renewed = heap_renew(owner);
#else
            void *renewed = owner;
#endif
            if (__builtin_capstone_cap_get_tag(renewed)) {
                for (size_t a=base; a<base+bytes; a+=4096) {
                    struct heap_leaf *leaf=leaf_for(a,0);
                    if (!leaf) __builtin_trap();
                    leaf->owner[(a>>12)&511]=0;
                }
                struct mr_rec e = *r; e.owner = renewed;
                mr_free[mr_nfree++] = e; mr_free_bytes += bytes; mr_remove(r);
                ++__heap_reuse_puts;
                index_unlock();
                return 0;
            }
            /* CRENEW's failure result: the old lifetime is still live, so the mapping goes back by munmap with it */
            ++__heap_reuse_renew_failed;
        }
        if (r) { mr_remove(r); ++__heap_reuse_spills; }
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
