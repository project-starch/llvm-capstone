/* Mapping authority for the mallocng.
 * All ordinary allocator state remains in musl. No per-object records. */
#include <errno.h>
#include <stdint.h>
#include <sys/mman.h>
#include <string.h>
#include "vm.h"
#include <heap_sublet.h>

struct heap_mid *__malloc_page_root[512];
/* The mapping's own reference, at the entry of its first page, in an index of the same shape.
 * The owner that mallocng holds is a child one level below it (heap_issue), and munmap must hand
 * back the mapping itself: with root_grants the kernel refuses an object derived inside a mapping,
 * even one over its whole range. Only this file reads it. */
static struct heap_mid *mapping_root[512];
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
static struct heap_leaf *leaf_in(struct heap_mid **root, size_t a, int create)
{
    struct heap_mid *mid = root[a >> 30];
    if (!mid && create) {
        mid = mmap(0, sizeof *mid, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (mid == MAP_FAILED) return 0;
        root[a >> 30] = mid;
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
static struct heap_leaf *leaf_for(size_t a, int create)
{ return leaf_in(__malloc_page_root, a, create); }

void *__heap_mapping(size_t n)
{
    if (!n || n > CV_MAX_REGION_BYTES) { errno=ENOMEM; return MAP_FAILED; }
    size_t bytes=heap_round((n+4095)&-4096UL);
    HEAP_POLICY_SYMBOL(); /* Link-time check: libc and SDK use the same policy. */
    sublet_cap grant;
    size_t align=heap_granule(bytes);
    if (align < 4096) align=4096;
    if (cap_vm_acquire(&grant,bytes,align,PROT_READ|PROT_WRITE,CAP_VM_HEAP))
        return MAP_FAILED;
    void *raw=cap_vm_copyable(&grant);
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
    struct heap_leaf *mapping=leaf_in(mapping_root,base,1);
    if (!mapping) {
        int e=errno;
        index_unlock(); munmap(raw,bytes); errno=e; return MAP_FAILED;
    }
    /* The mapping-wide child becomes the parent of all objects within the
     * mapping. Its CDERIVE gives the failure result, an untagged 0, when the
     * lifetime table has no slot left and cannot grow (the 63-bit format):
     * the mapping goes back and the allocator sees ENOMEM, nothing faults. */
    void *owner=heap_issue(raw,bytes);
    if (!owner) {
        index_unlock(); munmap(raw,bytes); errno=ENOMEM; return MAP_FAILED;
    }
    for (size_t a=base; a<base+bytes; a+=4096) {
        struct heap_leaf *leaf=leaf_for(a,0);
        if (leaf->owner[(a>>12)&511]) __builtin_trap();
        leaf->owner[(a>>12)&511]=owner;
    }
    if (mapping->owner[(base>>12)&511]) __builtin_trap();
    mapping->owner[(base>>12)&511]=raw;
    index_unlock();
    return owner;
}

int __heap_unmapping(void *owner,size_t n)
{
    size_t bytes=heap_round((n+4095)&-4096UL);
    size_t base=__builtin_capstone_cap_get_cursor(owner);
    index_lock();
    struct heap_leaf *mapping=leaf_in(mapping_root,base,0);
    void *raw=mapping ? mapping->owner[(base>>12)&511] : 0;
    if (!raw) __builtin_trap();
    int rc=munmap(raw,bytes);
    if (!rc) {
        for (size_t a=base; a<base+bytes; a+=4096) {
            struct heap_leaf *leaf=leaf_for(a,0);
            if (!leaf) __builtin_trap();
            leaf->owner[(a>>12)&511]=0;
        }
        mapping->owner[(base>>12)&511]=0;
    }
    index_unlock();
    return rc;
}
