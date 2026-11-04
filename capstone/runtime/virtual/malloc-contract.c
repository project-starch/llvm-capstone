#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#ifdef __CAPSTONE_PURECAP__
#include "vm.h"
#include <malloc_capstone.h>
#include <capstone/capability.h>
#endif
static void *volatile saved;
/* Keeps the denial cases' size class populated, so that freeing the probed object never
 * empties its group and nothing above the object (group, outer slot, mapping) is released:
 * the probe then reads the OBJECT's lifetime and nothing else's. The anchor and the probe
 * are the first two allocations of their class in this process, so mallocng places them in
 * the class's first group (four slots of 112 bytes for 100-byte requests); the witness
 * below is the distance between them, which a changed policy would break. */
static void *volatile anchored;
__attribute__((noinline)) static void fault_store(volatile char *p)
{
#ifdef __CAPSTONE_PURECAP__
    __asm__ volatile(".global cap_malloc_fault_store\ncap_malloc_fault_store:\nsb zero, 0(%0)"
                     : : "r"(p) : "memory");
#else
    *p = 0;
#endif
}
#ifdef __CAPSTONE_PURECAP__
extern unsigned long __capstone_malloc_bind_refused;
extern unsigned long __capstone_malloc_bind_first[6];
extern unsigned long __capstone_malloc_ng_null;
extern unsigned long __capstone_malloc_ids_failed;
extern unsigned long __capstone_malloc_map_calls;
extern unsigned long __capstone_malloc_map_refused_early;
extern unsigned long __capstone_malloc_map_refused_acquire;
extern unsigned long __capstone_malloc_map_last[4];
#endif
/* An allocation whose result is only tested against NULL may be assumed to succeed when malloc and
 * realloc are treated as builtins (upstream LLVM semantics, not a fork defect), so `!realloc(p, huge)`
 * can fold to false and the failure path is never observed. Every check that expects an allocation
 * to FAIL passes the result through this barrier, so the contract means the same with or without
 * -fno-builtin. Found 2026-10-20: the v6 kit's SDK lacked -fno-builtin and this check read a
 * folded false (MALLOC_FAIL:failed VM growth preserves original) while the runtime set ENOMEM. */
static void *kept(void *q)
{ __asm__ volatile("" : "+r"(q) : : "memory"); return q; }
static void require(int ok, const char *what)
{
    if (ok) return;
    fprintf(stderr, "MALLOC_FAIL:%s\n", what);
#ifdef __CAPSTONE_PURECAP__
    /* Which step refused. Without this a NULL from malloc says only "out of memory". */
    fprintf(stderr, "MALLOC_WHY bind_refused=%lu ng_null=%lu ids_failed=%lu"
                    " map_calls=%lu map_early=%lu map_acquire=%lu"
                    " last_bytes=%lu last_align=%lu last_errno=%lu asked=%lu\n",
            __capstone_malloc_bind_refused, __capstone_malloc_ng_null,
            __capstone_malloc_ids_failed, __capstone_malloc_map_calls,
            __capstone_malloc_map_refused_early, __capstone_malloc_map_refused_acquire,
            __capstone_malloc_map_last[0], __capstone_malloc_map_last[1],
            __capstone_malloc_map_last[2], __capstone_malloc_map_last[3]);
    fprintf(stderr, "MALLOC_WHY2 first_slot=%lu start=%lu end=%lu stride=%lu base=%lu group_end=%lu\n",
            __capstone_malloc_bind_first[0], __capstone_malloc_bind_first[1],
            __capstone_malloc_bind_first[2], __capstone_malloc_bind_first[3],
            __capstone_malloc_bind_first[4], __capstone_malloc_bind_first[5]);
#endif
    fflush(stderr);
    exit(1);
}
static void anchored_with(const void *p)
{
    const char *a = anchored, *q = p;
    require(a && q && (a < q ? q - a : a - q) < 4096, "anchor and probe share a group");
}
/* A denial probe that returned. On the qualified heap that is a failure. A bounds-only heap
 * (sdk.json heap_lifetimes false) cannot deny a stale object, so run-malloc.py passes
 * may-survive there and expects this marker from exactly the cases it pre-registers. */
static int survived(const char *mode, int may_survive)
{
    if (may_survive) { printf("MALLOC_SURVIVED:%s\n", mode); fflush(stdout); return 0; }
    require(0, "denial survived"); return 1;
}
static void basic(void)
{
    char *p = calloc(23, 7);
    require(p != NULL, "calloc");
    for (unsigned i = 0; i < 161; ++i) require(!p[i], "calloc zero");
    errno = EDOM; free(p); require(errno == EDOM, "free preserves errno");
    void *aligned = NULL;
    require(!posix_memalign(&aligned, 4096, 5001), "aligned allocation");
    require(!((uintptr_t)aligned & 4095), "alignment"); free(aligned);
    p = malloc(100); require(p != NULL, "in-place original"); memset(p, 42, 100);
    uintptr_t old = (uintptr_t)p;
    p = realloc(p, 101);
    require(p && (uintptr_t)p == old && p[99] == 42, "musl in-place growth");
    errno = 0;
    require(!kept(realloc(p, SIZE_MAX)) && errno == ENOMEM && p[99] == 42,
            "failed realloc preserves original");
    free(p);
    p = malloc(200000); require(p != NULL, "large original"); p[199999] = 33;
#ifdef __CAPSTONE_PURECAP__
    errno = 0;
    require(!kept(realloc(p, CV_MAX_REGION_BYTES + 1)) && errno == ENOMEM && p[199999] == 33,
            "failed VM growth preserves original");
#endif
    p = realloc(p, 200016); require(p && p[199999] == 33, "large realloc");
    p = realloc(p, 400000); require(p && p[199999] == 33, "mremap growth");
    p = realloc(p, 200000); require(p && p[199999] == 33, "mremap shrink");
    free(p);
    p = malloc(200003); require(p != NULL, "precise bounds"); saved = p;
    ((char *)saved)[200002] = 91;
    require(((char *)saved)[200002] == 91, "last requested byte survives reload");
    free(p);
    puts("MALLOC_BASIC_OK");
}
static void tagged(void)
{
    char *value = malloc(32); require(value != NULL, "value"); value[0] = 71;
    char **container = calloc(1, 64); require(container != NULL, "container");
    container[0] = value;
    uintptr_t old = (uintptr_t)container;
    container = realloc(container, 65536);
    require(container != NULL, "realloc container");
#ifdef __CAPSTONE_PURECAP__
    /* The qualified mallocng size classes force this move. The native
     * reference libc may legally grow the allocation in place. */
    require((uintptr_t)container != old, "moving realloc trigger");
#else
    (void)old;
#endif
    require(container[0][0] == 71, "moving realloc preserves capability");
    free(container[0]); free(container);
#ifdef __CAPSTONE_PURECAP__
    sublet_cap owner;
    require(!cap_vm_acquire(&owner, 4096, 4096, PROT_READ | PROT_WRITE,
                            7, CAP_VM_MAPPING), "linear VM grant");
    unsigned long base = sublet_base(&owner);
    container = calloc(1, 64); require(container != NULL, "linear container");
    sublet_move(&owner, (sublet_cap *)container);
    container = realloc(container, 65536); require(container != NULL, "linear container move");
    require(sublet_base((sublet_cap *)container) == base, "linear tag transfer");
    require(!__capstone_vm_unmap(base, 4096), "linear VM grant release");
    free(container);
#endif
    puts("MALLOC_TAG_COPY_OK");
}
static void population(void)
{
    /* vp2 caps the flat lifetime table at 65536 records (a 16-bit node index), which 70000
     * exceeds by construction. Parameterised so the same source runs at 70000 under QEMU and at
     * 60000 on silicon; the default is unchanged. The count used is printed and recorded. */
    unsigned long COUNT = 70000UL;
    {
        const char *e = getenv("CAPSTONE_MALLOC_COUNT");
        if (e && *e) { char *end = 0; unsigned long n = strtoul(e, &end, 10);
                       if (end && !*end && n) COUNT = n; }
    }
    printf("MALLOC_COUNT_USED %lu\n", COUNT); fflush(stdout);
    char **p = malloc(COUNT * sizeof(*p)); require(p != NULL, "population array");
    for (unsigned i = 0; i < COUNT; ++i) {
        p[i] = malloc(17); require(p[i] != NULL, "population allocation"); p[i][0] = i % 127;
    }
    for (unsigned i = 0; i < COUNT; ++i) {
        require(p[i][0] == i % 127, "population contents"); free(p[i]);
    }
    free(p); puts("MALLOC_POPULATION_OK");
}
static void churn(void)
{
    /* This is a node-recycling stress test. Keep each size class populated
     * and leave holes, so it does not mostly measure Linux map/unmap work. */
    void *anchor[254];
    for (unsigned i=0; i<254; ++i) {
        anchor[i] = malloc(17 + i/2); require(anchor[i] != NULL, "churn anchor");
    }
    for (unsigned i=1; i<254; i+=2) { free(anchor[i]); anchor[i] = NULL; }
    for (unsigned i = 0; i < 200000; ++i) {
        char *p = malloc(17 + i % 127); require(p != NULL, "churn allocation");
        p[0] = 11; free(p);
        if (!(i % 50000)) { printf("MALLOC_CHURN_PROGRESS:%u\n", i); fflush(stdout); }
    }
    for (unsigned i=0; i<254; i+=2) free(anchor[i]);
    puts("MALLOC_CHURN_OK");
}
static void hot(int force_service)
{
    /* Hold the group live, warm offset cycling and leave enough free node
     * capacity for this directed fast-path check. Output is outside the loop. */
    void *anchor[64];
    for (unsigned i=0; i<64; ++i) { anchor[i] = malloc(100); require(anchor[i]!=NULL, "hot setup"); }
    free(anchor[32]); anchor[32] = NULL;
    for (unsigned i=0; i<32; ++i) { void *p = malloc(100); require(p!=NULL, "hot warmup"); free(p); }
    puts("MALLOC_HOT_BEGIN"); fflush(stdout);
    for (unsigned i=0; i<32; ++i) {
        char *p = malloc(100); require(p!=NULL, "hot allocation");
        p[99] = 7; require(p[99]==7, "hot contents"); free(p);
#ifdef __CAPSTONE_PURECAP__
        if (force_service && i == 0) __capstone_vm_wait();
#endif
    }
    puts("MALLOC_HOT_END"); fflush(stdout);
    for (unsigned i=0; i<64; ++i) free(anchor[i]);
}
/* Option (iii): a remapped mapping can come back page-aligned but not granule-aligned, which
 * __capstone_vm_remap cannot be asked to avoid -- it takes an address and two lengths and no
 * alignment. The runtime then refuses it and realloc falls through to malloc-plus-copy, so the
 * cost is a copy and never a widened bound. This REPORTS how often that happened across one grow,
 * and checks the contents survive whichever path was taken. mremap returns page-aligned addresses,
 * so at these sizes (granule 8 KiB at 6 MiB, 16 KiB at 12 MiB) the copy is the expected path; the
 * number is printed so the acceptance runs record it rather than anyone inferring it. */
#ifdef __CAPSTONE_PURECAP__
extern unsigned long __capstone_malloc_remap_unaligned;
#endif
static void remap_align(void)
{
    size_t small = 6u << 20, large = 12u << 20;
#ifdef __CAPSTONE_PURECAP__
    unsigned long before = __capstone_malloc_remap_unaligned;
#else
    unsigned long before = 0;
#endif
    char *p = malloc(small);
    require(p != NULL, "remap-align setup");
    memset(p, 0x5A, small);
    char *q = realloc(p, large);
    require(q != NULL, "remap-align growth");
    /* every byte of the old object must still be there, copied or remapped */
    for (size_t i = 0; i < small; i += 4096)
        require((unsigned char)q[i] == 0x5A, "remap-align contents");
    require((unsigned char)q[small - 1] == 0x5A, "remap-align last byte");
    /* and the grown object must be writable to its end */
    q[large - 1] = 0x3C;
    require((unsigned char)q[large - 1] == 0x3C, "remap-align tail writable");
#ifdef __CAPSTONE_PURECAP__
    printf("MALLOC_REMAP_UNALIGNED %lu\n", __capstone_malloc_remap_unaligned - before);
#else
    printf("MALLOC_REMAP_UNALIGNED %lu\n", before);
#endif
    free(q);
}

#ifdef __CAPSTONE_PURECAP__
extern unsigned long __capstone_malloc_map_ok[2];
extern unsigned long __capstone_malloc_map_admitted;
extern unsigned long __capstone_malloc_map_last[4];
extern unsigned long __capstone_malloc_map_refused_acquire;
extern unsigned long __capstone_malloc_map_ok_base;
extern unsigned long __capstone_malloc_map_last_base;
#endif
/* Separates is_power_of_2(bytes) from "bytes > 16384" as the module's arena predicate. Both score
 * identically on every size measured so far, because each one falls the same side of both. These
 * two invert the predictions. No require(): a failure here is a READING, not a fault, and both
 * readings must survive into one run. */
static void predicate(void)
{
    /* group needs 32768 -- a power of two, ABOVE 16384 */
    void *a = malloc(9341);
    printf("MALLOC_PRED pow2_above=%s\n", a ? "ALLOCATED" : "NULL");
    fflush(stdout);
    if (a) { memset(a, 0x11, 9341); free(a); }
    /* group needs 12288 -- NOT a power of two, BELOW 16384 */
    void *b = malloc(2029);
    printf("MALLOC_PRED nonpow2_below=%s\n", b ? "ALLOCATED" : "NULL");
    fflush(stdout);
    if (b) { memset(b, 0x22, 2029); free(b); }
#ifdef __CAPSTONE_PURECAP__
    /* The REFUSED mapping's own size, so the NULL arm is a measurement rather than a
     * prediction from the group model -- which is all the admitted-side figure made it. */
    printf("MALLOC_PRED admitted=%lu last_ok_bytes=%lu last_ok_align=%lu"
           " refused=%lu last_refused_bytes=%lu last_refused_align=%lu last_refused_errno=%lu\n",
           __capstone_malloc_map_admitted, __capstone_malloc_map_ok[0],
           __capstone_malloc_map_ok[1], __capstone_malloc_map_refused_acquire,
           __capstone_malloc_map_last[0], __capstone_malloc_map_last[1],
           __capstone_malloc_map_last[2]);
    /* The decisive reading: if an ADMITTED mapping's base is not naturally aligned to its own
     * size, add()'s alignment disjunct cannot be what refuses the others. */
    printf("MALLOC_PRED ok_base=%lu ok_base_misaligned_vs_size=%lu\n",
           __capstone_malloc_map_ok_base, __capstone_malloc_map_last_base);
#endif
    fflush(stdout);
}
/* Produce the one row the record is missing: a POWER-OF-TWO mapping refused with
 * naturally_aligned=0. The alignment half of add():466 currently rests on the same arm giving
 * opposite answers across runs plus a 1-in-4 board result, which is behavioural inference, not a
 * printed disjunct.
 *
 * It does not need a lucky boot. A power-of-two mapping is refused whenever mmap does not land it
 * on its own size, roughly half the time, so FORCING many group mappings in one process makes the
 * row near-certain. Each size class holds only cnt slots, so keeping every object live forces a
 * fresh mapping every cnt allocations.
 *
 * Deliberately model-free: it does not assume which group a request lands in -- that assumption is
 * what made the earlier arm measure the wrong thing. It allocates across a spread of classes and
 * lets the launcher's CAPSTONE_VM_ADD_REFUSED rows report the real sizes and bases.
 */
static void align_probe(void)
{
    enum { KEEP = 600 };
    static void *live[KEEP];
    static const size_t sizes[] = {520, 1000, 2029, 3000, 4100, 6144, 9000, 12000};
    unsigned n = 0, ok = 0, failed = 0;
    for (unsigned round = 0; round < KEEP / (sizeof sizes / sizeof *sizes); ++round) {
        for (unsigned i = 0; i < sizeof sizes / sizeof *sizes && n < KEEP; ++i, ++n) {
            live[n] = malloc(sizes[i]);
            if (live[n]) { ok++; memset(live[n], 0x5A, 1); } else failed++;
        }
    }
    printf("MALLOC_ALIGN_PROBE attempted=%u ok=%u failed=%u\n", n, ok, failed);
    fflush(stdout);
    for (unsigned i = 0; i < n; ++i) if (live[i]) free(live[i]);
    puts("MALLOC_ALIGN_PROBE_DONE");
    fflush(stdout);
}
#ifdef __CAPSTONE_PURECAP__
extern unsigned long __capstone_malloc_recycle_hits;
extern unsigned long __capstone_malloc_recycle_kept;
extern unsigned long __capstone_malloc_recycle_unmapped;
extern int malloc_trim(size_t);
/* A RECYCLED group mapping (heap.c "Recycled group mappings"). 20000 bytes at zero usage is
 * mallocng's single-slot group (alloc_group's cnt==1 branch), which it always releases when the
 * object is freed (okay_to_free: the stride is below the class's); its allocation must map (the
 * group is new, not one a startup allocation already holds: the first version of this test used
 * 5000 bytes, whose class a startup object occupies, and failed on exactly that). The release
 * keeps the mapping (kept advances); the next allocation of that size takes it (hits advances, no
 * map) and lands on the freed object's bytes. All of these are required, so a run in which the
 * mapping was not reused fails here instead of reading as a pass. The stale cases then
 * use the old object's capability, which must fault under either heap configuration: releasing
 * the group revoked the group lifetime every object descends from. recycle-tags fills the old
 * object with tagged capabilities to a live object (and reads one back, so the tag reader is
 * shown to see a tag), and after reuse reads every granule of the new object: none may be
 * tagged. Release and reuse must reach the launcher for nothing: the gate brackets them with
 * MALLOC_RECYCLE_BEGIN/END and allows only the END marker's own write. */
enum { RECYCLE_SIZE = 20000 };
static char *recycled(char *old, unsigned long kept, unsigned long hits, int bracket)
{
    if (bracket) { puts("MALLOC_RECYCLE_BEGIN"); fflush(stdout); }
    free(old);
    unsigned long kept_after = __capstone_malloc_recycle_kept;
    unsigned long maps = __capstone_malloc_map_calls;
    char *fresh = malloc(RECYCLE_SIZE);
    unsigned long hits_after = __capstone_malloc_recycle_hits, maps_after = __capstone_malloc_map_calls;
    if (bracket) { puts("MALLOC_RECYCLE_END"); fflush(stdout); }
    printf("MALLOC_RECYCLE kept=%lu->%lu hits=%lu->%lu unmapped=%lu old=0x%lx fresh=0x%lx\n", kept,
           kept_after, hits, hits_after, __capstone_malloc_recycle_unmapped,
           (unsigned long)(uintptr_t)saved, (unsigned long)(uintptr_t)fresh);
    fflush(stdout);
    require(kept_after == kept + 1, "recycle: the released group's mapping was kept");
    require(fresh && hits_after == hits + 1 && maps_after == maps,
            "recycle: the next group took the kept mapping");
    uintptr_t a = (uintptr_t)saved, b = (uintptr_t)fresh;
    require(a < b + RECYCLE_SIZE && b < a + RECYCLE_SIZE, "recycle: reuse overlaps the freed object");
    return fresh;
}
static int recycle_outside(const char *mode, size_t n);
static int recycle_root(void);
static int recycle_limits(void);
static int recycle_path(const char *mode, int may_survive);
static int recycle_case(const char *mode, int may_survive)
{
    if (!strcmp(mode, "recycle-header")) return recycle_outside(mode, RECYCLE_SIZE);
    if (!strcmp(mode, "recycle-slack")) return recycle_outside(mode, 0);   /* a multi-slot group */
    if (!strcmp(mode, "recycle-root")) return recycle_root();
    if (!strcmp(mode, "recycle-limits")) return recycle_limits();
    if (!strcmp(mode, "recycle-geometry") || !strcmp(mode, "recycle-nested") ||
        !strcmp(mode, "recycle-aligned") || !strcmp(mode, "recycle-realloc"))
        return recycle_path(mode, may_survive);
    char *live = malloc(32); require(live != NULL, "recycle live object"); live[0] = 5;
    unsigned long kept = __capstone_malloc_recycle_kept, hits = __capstone_malloc_recycle_hits;
    unsigned long maps = __capstone_malloc_map_calls;
    char *old = malloc(RECYCLE_SIZE); require(old != NULL, "recycle setup"); saved = old;
    require(__capstone_malloc_map_calls == maps + 1 && __capstone_malloc_recycle_hits == hits,
            "recycle: the old object's group is a new mapping");
    if (!strcmp(mode, "recycle-tags")) {
        unsigned granules = RECYCLE_SIZE / 16;
        for (unsigned i = 0; i < granules; ++i) ((char *volatile *)old)[i] = live;
        require(__builtin_capstone_cap_get_tag(((char *volatile *)old)[granules / 2]),
                "recycle-tags: the reader sees a stored tag");
        char *fresh = recycled(old, kept, hits, 1);
        unsigned tagged = 0;
        for (unsigned i = 0; i < granules; ++i)
            tagged += __builtin_capstone_cap_get_tag(((char *volatile *)fresh)[i]) != 0;
        printf("MALLOC_RECYCLE_TAGS tagged=%u of %u\n", tagged, granules);
        require(!tagged, "recycle-tags: no tag of the old group survives into the new object");
        require(live[0] == 5, "recycle-tags: the live object is untouched");
        puts("MALLOC_FAULT_READY:recycle-tags"); fflush(stdout);
        free(fresh); free(live);
        /* The released group's mapping is kept again; malloc_trim gives it back. */
        unsigned long maps = __capstone_malloc_map_calls;
        require(malloc_trim(0) == 1 && malloc_trim(0) == 0, "recycle-tags: malloc_trim releases kept mappings");
        char *after = malloc(RECYCLE_SIZE);
        require(after && __capstone_malloc_map_calls == maps + 1, "recycle-tags: after a trim the group maps again");
        free(after);
        puts("MALLOC_SURVIVED:recycle-tags"); fflush(stdout);
        return 0;
    }
    char *fresh = recycled(old, kept, hits, 0); (void)fresh;
    if (!strcmp(mode, "recycle-stale")) {
        puts("MALLOC_FAULT_READY:recycle-stale"); fflush(stdout); fault_store(saved);
    } else if (!strcmp(mode, "recycle-free")) {
        puts("MALLOC_FAULT_READY:recycle-free"); fflush(stdout); free(saved);
    } else return 2;
    return survived(mode, may_survive);
}
/* The review cases (2026-11-03). Sizes from mallocng's own geometry at zero usage, which the
 * fresh-map precondition checks rather than assumes: 6000 bytes is a three-slot group of 6544-byte
 * slots in a 20480-byte mapping (832 bytes of trailing slack), 17000 and 20000 are single-slot groups
 * of two different classes in 20480-byte mappings (a 32-byte header), posix_memalign(4096, 20000)
 * is a single slot in 24576 bytes, 300 bytes is a nested group. */
#include <pthread.h>
extern unsigned long __capstone_malloc_map_ok[2];
static int whole_group(const void *p)   /* CAPSTONE_HEAP_OBJECT_BOUNDS=0: the object reaches its group */
{ return __builtin_capstone_cap_get_base((void *)p) < __builtin_capstone_cap_get_cursor((void *)p); }
static char *fresh_malloc(size_t n, const char *what)
{
    unsigned long maps = __capstone_malloc_map_calls, hits = __capstone_malloc_recycle_hits;
    char *p = malloc(n);
    require(p && __capstone_malloc_map_calls == maps + 1 && __capstone_malloc_recycle_hits == hits, what);
    return p;
}
static void free_kept(void *p, const char *what)
{
    unsigned long kept = __capstone_malloc_recycle_kept;
    free(p);
    require(__capstone_malloc_recycle_kept == kept + 1, what);
}
static char *reused_malloc(size_t n, const char *what)
{
    unsigned long maps = __capstone_malloc_map_calls, hits = __capstone_malloc_recycle_hits;
    char *p = malloc(n);
    require(p && __capstone_malloc_map_calls == maps && __capstone_malloc_recycle_hits == hits + 1, what);
    return p;
}
/* A fresh MULTI-slot group: the first candidate size whose allocation maps a new group whose slot
 * is at most half the mapping. Which classes a startup allocation already occupies, and so which
 * first group gets which count, is not assumed (6000 bytes mapped a single 8192-byte slot in the
 * first run); unsuitable candidates stay allocated so their groups cannot feed the cache. */
static char *fresh_multi_slot(unsigned long *length, size_t *size)
{
    static const size_t sizes[] = {6000, 5000, 9000, 10000, 4500, 7500, 3000, 2500};
    for (unsigned i = 0; i < sizeof sizes / sizeof *sizes; ++i) {
        unsigned long maps = __capstone_malloc_map_calls, hits = __capstone_malloc_recycle_hits;
        char *p = malloc(sizes[i]); require(p != NULL, "multi-slot candidate");
        unsigned long l = __capstone_malloc_map_ok[0];
        struct capstone_malloc_slot *r = __capstone_malloc_find(p);
        size_t stride = r ? __builtin_capstone_cap_get_end(r->parent) - __builtin_capstone_cap_get_base(r->parent) : 0;
        printf("MALLOC_RECYCLE_MULTI size=%lu maps=+%lu length=%lu stride=%lu\n", (unsigned long)sizes[i],
               __capstone_malloc_map_calls - maps, l, (unsigned long)stride);
        if (__capstone_malloc_map_calls == maps + 1 && __capstone_malloc_recycle_hits == hits &&
            stride && 2 * stride <= l && l >= 16384) { *length = l; *size = sizes[i]; return p; }
    }
    require(0, "a fresh multi-slot group of at least 16384 bytes");
    return NULL;
}
/* Every 16-byte granule of [c + from, c + to), c's own bounds clipped. */
static unsigned fill_caps(char *c, size_t from, size_t to, void *value)
{
    unsigned n = 0;
    for (size_t o = (from + 15) & ~(size_t)15; o + 16 <= to; o += 16, ++n)
        *(void *volatile *)(c + o) = value;
    return n;
}
static unsigned count_tags(char *c, size_t from, size_t to)
{
    unsigned n = 0;
    for (size_t o = (from + 15) & ~(size_t)15; o + 16 <= to; o += 16)
        n += __builtin_capstone_cap_get_tag(*(void *volatile *)(c + o)) != 0;
    return n;
}
/* recycle-header / recycle-slack: with object bounds off the program holds the group's authority;
 * it stores live capabilities into everything of the group outside its own slot -- the header
 * after the allocator's first granule, the untaken slots, the trailing slack -- and after release
 * and reuse the new group's authority must read none of them. With object bounds on the program
 * cannot reach those bytes (outside=0) and the case reads its own object only. */
static int recycle_outside(const char *mode, size_t n)
{
    char *live = malloc(32); require(live != NULL, "recycle live object");
    unsigned long length = 0;
    char *p = n ? fresh_malloc(n, "recycle-outside: a new mapping") : fresh_multi_slot(&length, &n);
    saved = p;
    unsigned outside = 0;
    size_t base = __builtin_capstone_cap_get_base(p), end = __builtin_capstone_cap_get_end(p);
    if (whole_group(p)) {
        struct capstone_malloc_slot *r = __capstone_malloc_find(p);
        require(r != NULL, "recycle-outside: slot record");
        size_t s0 = __builtin_capstone_cap_get_base(r->parent), s1 = __builtin_capstone_cap_get_end(r->parent);
        char *g = p - (__builtin_capstone_cap_get_cursor(p) - base);
        outside = fill_caps(g, 16, s0 - base, live) + fill_caps(g, s1 - base, end - base, live);
        require(outside && count_tags(g, 16, s0 - base) + count_tags(g, s1 - base, end - base) == outside,
                "recycle-outside: the reader sees the stored tags");
    }
    free_kept(p, "recycle-outside: the released group's mapping was kept");
    char *q = reused_malloc(n, "recycle-outside: the next group took the kept mapping");
    size_t qb = __builtin_capstone_cap_get_base(q), qe = __builtin_capstone_cap_get_end(q);
    char *qg = q - (__builtin_capstone_cap_get_cursor(q) - qb);
    unsigned tagged = count_tags(qg, 0, qe - qb);
    printf("MALLOC_RECYCLE_OUTSIDE %s outside=%u tagged=%u of %lu group=%d\n", mode, outside, tagged,
           (unsigned long)((qe - qb) / 16), whole_group(q));
    fflush(stdout);
    require(!tagged, "recycle-outside: no tag of the old group survives into the new group");
    require(!whole_group(q) || qb == base, "recycle-outside: the same mapping");
    printf("MALLOC_FAULT_READY:%s\n", mode);
    free(q); free(live);
    printf("MALLOC_SURVIVED:%s\n", mode); fflush(stdout);
    return 0;
}
/* mallocng's struct meta under patch 0009 (meta.h): prev, next and mem capabilities, the two
 * masks, the bitfield word with last_idx in bits 0-4 at byte 56, cap_parent at 64, cap_slots (the
 * group's slot records, heap.c "Slot-record arrays") at 80, cap_count at 96; 112 bytes. The cases
 * below check cap_slots against the record they found before trusting the rest. */
enum { META_BYTES = 112, META_LAST_IDX = 56, META_CAP_SLOTS = 80, META_CAP_COUNT = 96 };
static struct capstone_malloc_slot *meta_slots(const struct capstone_malloc_slot *r, const char *what)
{
    struct capstone_malloc_slot *s = *(struct capstone_malloc_slot *const volatile *)((const char *)r->group + META_CAP_SLOTS);
    require(s && __builtin_capstone_cap_get_cursor(s) == __builtin_capstone_cap_get_cursor(r - r->index), what);
    return s;
}
static unsigned meta_word(const struct capstone_malloc_slot *r, size_t offset)
{ return (unsigned)*(const volatile unsigned long *)((const char *)r->group + offset); }
/* recycle-root: nothing an exported entry point hands out survives a group's release. Every
 * capability in the meta that __capstone_malloc_find's record points to and in its group's slot
 * records that covers the group header (only the group's own capability does, and with object
 * bounds off each slot's issued copy of it) is counted: a root kept in the meta or a record would
 * add one. The one found is then used after the release and reuse: it must fault. */
static int recycle_root(void)
{
    char *p = fresh_malloc(RECYCLE_SIZE, "recycle-root: a new mapping"); saved = p;
    struct capstone_malloc_slot *r = __capstone_malloc_find(p);
    require(r && r->group, "recycle-root: slot record");
    size_t s0 = __builtin_capstone_cap_get_base(r->parent), obj_end = __builtin_capstone_cap_get_cursor(p) + RECYCLE_SIZE;
    /* The meta itself, from the record's group pointer, and the records its cap_slots points to:
     * the group's bound records, last_idx + 1 of them. */
    struct capstone_malloc_slot *slots = meta_slots(r, "recycle-root: the meta points to the record's array");
    unsigned bound = (meta_word(r, META_LAST_IDX) & 31) + 1;
    const char *regions[2] = { (const char *)r->group, (const char *)slots };
    size_t spans[2] = { META_BYTES, bound * sizeof *slots };
    unsigned found = 0; void *exposed = NULL;
    for (int k = 0; k < 2; ++k)
        for (size_t o = 0; o + 16 <= spans[k]; o += 16) {
            void *c = *(void *volatile *)(regions[k] + o);
            if (!__builtin_capstone_cap_get_tag(c)) continue;
            if (__builtin_capstone_cap_get_base(c) < s0 && __builtin_capstone_cap_get_end(c) >= obj_end) {
                ++found; exposed = c;
            }
        }
    unsigned expected = whole_group(p) ? 2 : 1;   /* the meta's group capability (+ the slot's issued copy) */
    printf("MALLOC_RECYCLE_ROOT found=%u expected=%u meta_bytes=%lu record_bytes=%lu\n", found, expected,
           (unsigned long)spans[0], (unsigned long)spans[1]);
    fflush(stdout);
    require(found == expected, "recycle-root: no capability beyond the group's own covers the group");
    free_kept(p, "recycle-root: the released group's mapping was kept");
    char *q = reused_malloc(RECYCLE_SIZE, "recycle-root: the next group took the kept mapping"); (void)q;
    saved = exposed;
    puts("MALLOC_FAULT_READY:recycle-root"); fflush(stdout);
    fault_store((char *)saved + (s0 - __builtin_capstone_cap_get_base(exposed)));
    return 1;
}
static volatile unsigned long trims_done;
static void *trim_loop(void *arg)
{
    volatile int *stop = arg;
    while (!*stop) { malloc_trim(0); ++trims_done; }
    return NULL;
}
/* recycle-limits: more releases of one length than the cache holds; the excess is unmapped. Then
 * a second thread trims continuously while this one allocates and frees that length. */
static int recycle_limits(void)
{
    enum { N = 6, PER_LENGTH = 4 };
    char *p[N];
    for (unsigned i = 0; i < N; ++i) p[i] = fresh_malloc(RECYCLE_SIZE, "recycle-limits: a new mapping each");
    unsigned long kept = __capstone_malloc_recycle_kept, unmapped = __capstone_malloc_recycle_unmapped;
    for (unsigned i = 0; i < N; ++i) free(p[i]);
    require(__capstone_malloc_recycle_kept == kept + PER_LENGTH &&
            __capstone_malloc_recycle_unmapped == unmapped + (N - PER_LENGTH),
            "recycle-limits: four kept, the rest unmapped");
    unsigned long hits = __capstone_malloc_recycle_hits, maps = __capstone_malloc_map_calls;
    for (unsigned i = 0; i < N; ++i) { p[i] = malloc(RECYCLE_SIZE); require(p[i] != NULL, "recycle-limits: again"); }
    require(__capstone_malloc_recycle_hits == hits + PER_LENGTH && __capstone_malloc_map_calls == maps + (N - PER_LENGTH),
            "recycle-limits: four reused, the rest mapped");
    for (unsigned i = 0; i < N; ++i) free(p[i]);
    volatile int stop = 0; pthread_t t;
    require(!pthread_create(&t, NULL, trim_loop, (void *)&stop), "recycle-limits: trim thread");
    /* At least 2000 rounds, and on until the trim thread has run 100 times (one hart: the threads
     * interleave by preemption), bounded. */
    for (unsigned i = 0; i < 2000 || (trims_done < 100 && i < 2000000); ++i) {
        char *q = malloc(RECYCLE_SIZE); require(q != NULL, "recycle-limits: under trim");
        q[0] = 1; q[RECYCLE_SIZE - 1] = 2; require(q[0] == 1 && q[RECYCLE_SIZE - 1] == 2, "recycle-limits: contents");
        free(q);
    }
    stop = 1;
    require(!pthread_join(t, NULL), "recycle-limits: join");
    unsigned long trims = trims_done;
    require(trims >= 100, "recycle-limits: the trim thread ran concurrently");
    malloc_trim(0);
    printf("MALLOC_RECYCLE_LIMITS hits=%lu kept=%lu unmapped=%lu trims=%lu\n", __capstone_malloc_recycle_hits,
           __capstone_malloc_recycle_kept, __capstone_malloc_recycle_unmapped, trims);
    puts("MALLOC_FAULT_READY:recycle-limits"); puts("MALLOC_SURVIVED:recycle-limits"); fflush(stdout);
    return 0;
}
/* recycle-geometry / -nested / -aligned / -realloc: the group is released by another path, its
 * mapping is reused, and the old object's capability must fault. */
static int recycle_path(const char *mode, int may_survive)
{
    char *live = malloc(32); require(live != NULL, "recycle live object");
    char *fresh = NULL;
    if (!strcmp(mode, "recycle-geometry")) {
        /* a three-slot group's mapping reused by a single-slot group of another class */
        unsigned long length; size_t size;
        char *a = fresh_multi_slot(&length, &size); saved = a;
        require(fill_caps(a, 0, size, live) == size / 16, "recycle-geometry: fill");
        free_kept(a, "recycle-geometry: kept");
        /* A single-slot group of the same mapping length: the request that leaves room for the
         * 4-byte footer and a header of up to 64 bytes (the granule below 64 KiB). */
        size_t single = length - 68;
        printf("MALLOC_RECYCLE_GEOMETRY length=%lu multi=%lu single=%lu\n", length, (unsigned long)size,
               (unsigned long)single); fflush(stdout);
        fresh = reused_malloc(single, "recycle-geometry: a single-slot group of another class took it");
        struct capstone_malloc_slot *r = __capstone_malloc_find(fresh);
        require(r && __builtin_capstone_cap_get_end(r->parent) - __builtin_capstone_cap_get_base(r->parent) >= single,
                "recycle-geometry: the new group is a single slot");
        require(!count_tags(fresh, 0, single), "recycle-geometry: the new object is clean");
    } else if (!strcmp(mode, "recycle-nested")) {
        /* The object must be in a NESTED group whose host chain ends in a mapped group holding
         * nothing else, so that freeing the object retires the nested group, then its host's
         * mapped group, whose mapping the next nested group's chain then reuses. mallocng nests a
         * class whose first group is at most half a page; the chains for 240..800-byte objects end
         * in a two-slot group of 2032-byte slots in one page (class 23), which startup objects also
         * use. So first fill class 23's free slots (2000-byte objects, kept live) until one maps a
         * group of its own, and free that one: no class-23 group then has a free slot, and the next
         * nested chain must take a new class-23 group, from the kept mapping. Nesting is not taken
         * on trust: the object's group must sit inside a host slot whose record says it holds a
         * nested group (heap.c SLOT_NESTED, 2). CAPSTONE_RECYCLE_NESTED_SIZE forces a candidate
         * (1100 bytes, a directly mapped seven-slot group, is the control that must fail here). */
        static const size_t sizes[] = {600, 300, 400, 240, 800, 160, 128};
        const char *forced = getenv("CAPSTONE_RECYCLE_NESTED_SIZE");
        size_t only = forced ? strtoul(forced, NULL, 10) : 0;
        for (unsigned fill = 0; fill < 64; ++fill) {
            unsigned long maps = __capstone_malloc_map_calls;
            char *c = malloc(2000); require(c != NULL, "recycle-nested: class-23 filler");
            if (__capstone_malloc_map_calls != maps) { free(c); break; }
        }
        char *p = NULL; size_t n = 0;
        for (unsigned i = 0; i < (only ? 1 : sizeof sizes / sizeof *sizes) && !p; ++i) {
            n = only ? only : sizes[i];
            unsigned long maps = __capstone_malloc_map_calls, hits = __capstone_malloc_recycle_hits;
            char *c = malloc(n); require(c != NULL, "recycle-nested: candidate");
            struct capstone_malloc_slot *r = __capstone_malloc_find(c);
            size_t s0 = __builtin_capstone_cap_get_base(r->parent);
            size_t stride = __builtin_capstone_cap_get_end(r->parent) - s0;
            size_t hdr = stride >= 4096 ? (size_t)1 << (63 - __builtin_clzl(stride) - 9) : 16;
            if (hdr < 16) hdr = 16;
            size_t group = s0 - r->index * stride - hdr;   /* the nested group's base = its host slot's key */
            struct capstone_malloc_slot *host = __capstone_malloc_find((void *)group);
            int nested = host && host->kind == 2;
            int fresh_chain = __capstone_malloc_map_calls == maps + 1 || __capstone_malloc_recycle_hits == hits + 1;
            printf("MALLOC_RECYCLE_NESTED size=%lu stride=%lu nested=%d new_chain=%d\n", (unsigned long)n,
                   (unsigned long)stride, nested, fresh_chain); fflush(stdout);
            if (nested && fresh_chain) p = c;
        }
        require(p != NULL, "recycle-nested: an object in a nested group whose host chain is new");
        saved = p; free_kept(p, "recycle-nested: the host group's mapping was kept");
        fresh = reused_malloc(n, "recycle-nested: the next nested group's host took it");
    } else if (!strcmp(mode, "recycle-aligned")) {
        void *a = NULL;
        unsigned long maps = __capstone_malloc_map_calls;
        require(!posix_memalign(&a, 4096, RECYCLE_SIZE) && __capstone_malloc_map_calls == maps + 1,
                "recycle-aligned: a new mapping");
        saved = a; free_kept(a, "recycle-aligned: kept");
        unsigned long hits = __capstone_malloc_recycle_hits;
        require(!posix_memalign((void **)&fresh, 4096, RECYCLE_SIZE) && __capstone_malloc_recycle_hits == hits + 1,
                "recycle-aligned: reused");
    } else if (!strcmp(mode, "recycle-realloc")) {
        char *a = fresh_malloc(RECYCLE_SIZE, "recycle-realloc: a new mapping"); saved = a;
        unsigned long kept = __capstone_malloc_recycle_kept;
        char *moved = realloc(a, 60000);
        require(moved && (uintptr_t)moved != (uintptr_t)saved && __capstone_malloc_recycle_kept == kept + 1,
                "recycle-realloc: the move released and kept the old group");
        fresh = reused_malloc(RECYCLE_SIZE, "recycle-realloc: reused");
    } else return 2;
    (void)fresh;
    printf("MALLOC_FAULT_READY:%s\n", mode); fflush(stdout);
    fault_store(saved);
    return survived(mode, may_survive);
}
#endif
#ifdef __CAPSTONE_PURECAP__
/* calloc-clean (2026-11-03): calloc skips its memset for an object whose slot was clean when it
 * was taken -- a fresh mapping's slot nobody took before, or a reused mapping's slot, scrubbed at
 * that take (heap.c calloc). Every calloc here must still read zero and untagged over its whole
 * extent: (1) a fresh mapping's single slot, which takes the skip; (2) a slot of a live group that
 * an earlier object filled with bytes and capabilities and freed, which must not (the memset
 * zeroes it); (3) the first slot's mapping after its group was released and kept, reused: scrubbed,
 * so the skip again. With object bounds off the program holds its group's authority and could write
 * a slot nobody took, so that arm never skips. Each step's condition (a fresh map, the dirty
 * bytes' reuse, a recycle hit) is required, so a run that never creates it fails, not passes. */
extern unsigned long __capstone_malloc_calloc_clean;
static void zero_untagged(const char *p, size_t n, const char *what)
{
    for (size_t i = 0; i < n; ++i) require(!p[i], what);
    for (size_t o = 0; o + 16 <= n; o += 16)
        require(!__builtin_capstone_cap_get_tag(*(void *const volatile *)(p + o)), what);
}
static void dirty(char *p, size_t n, void *value)
{
    memset(p, 0xa5, n);
    for (size_t o = 0; o + 16 <= n; o += 16) *(void *volatile *)(p + o) = value;
    require(__builtin_capstone_cap_get_tag(*(void *volatile *)p), "calloc-clean: the reader sees a stored tag");
}
static int calloc_clean(void)
{
    char *live = malloc(32); require(live != NULL, "calloc-clean live object");
    /* (2) first, in the 100-byte class that the denial cases use: 112-byte slots, offset 0 */
    anchored = calloc(1, 100); require(anchored != NULL, "calloc-clean anchor");
    char *old = calloc(1, 100); require(old != NULL, "calloc-clean old"); anchored_with(old);
    int bounds_off = whole_group(old);
    zero_untagged(old, 100, "calloc-clean: a first-use object is zero");
    dirty(old, 100, live);
    uintptr_t address = (uintptr_t)old;
    free(old);
    static char *held[4096]; unsigned count = 0, reused = 0, cleaned = 0;
    while (count < 4096) {
        unsigned long before = __capstone_malloc_calloc_clean;
        held[count] = calloc(1, 100); require(held[count] != NULL, "calloc-clean reuse search");
        zero_untagged(held[count], 100, "calloc-clean: every object is zero and untagged");
        if ((uintptr_t)held[count++] == address) { reused = 1; cleaned = __capstone_malloc_calloc_clean != before; break; }
    }
    require(reused, "calloc-clean: the dirty slot was reused");
    require(!cleaned, "calloc-clean: a reused slot is zeroed, not skipped");
    /* (1) and (3) */
    unsigned long maps = __capstone_malloc_map_calls, hits = __capstone_malloc_recycle_hits;
    unsigned long skipped = __capstone_malloc_calloc_clean;
    char *first = calloc(1, RECYCLE_SIZE);
    require(first && __capstone_malloc_map_calls == maps + 1 && __capstone_malloc_recycle_hits == hits,
            "calloc-clean: a fresh mapping");
    require(__capstone_malloc_calloc_clean == skipped + !bounds_off, "calloc-clean: a fresh slot skips the memset");
    zero_untagged(first, RECYCLE_SIZE, "calloc-clean: a fresh slot is zero and untagged");
    dirty(first, RECYCLE_SIZE, live);
    unsigned long kept = __capstone_malloc_recycle_kept;
    free(first);
    require(__capstone_malloc_recycle_kept == kept + 1, "calloc-clean: the released mapping was kept");
    maps = __capstone_malloc_map_calls; skipped = __capstone_malloc_calloc_clean;
    char *again = calloc(1, RECYCLE_SIZE);
    require(again && __capstone_malloc_map_calls == maps && __capstone_malloc_recycle_hits == hits + 1,
            "calloc-clean: the kept mapping was reused");
    require((uintptr_t)again < (uintptr_t)first + RECYCLE_SIZE && (uintptr_t)first < (uintptr_t)again + RECYCLE_SIZE,
            "calloc-clean: the reuse overlaps the dirtied object");
    require(__capstone_malloc_calloc_clean == skipped + !bounds_off, "calloc-clean: a scrubbed slot skips the memset");
    zero_untagged(again, RECYCLE_SIZE, "calloc-clean: a reused mapping's slot is zero and untagged");
    printf("MALLOC_CALLOC_CLEAN reused_after=%u skipped=%lu bounds_off=%d\n", count, __capstone_malloc_calloc_clean,
           bounds_off);
    puts("MALLOC_FAULT_READY:calloc-clean");
    puts("MALLOC_SURVIVED:calloc-clean"); fflush(stdout);
    return 0;
}
/* The directed cases of the review (2026-11-03), each a survivor under every heap configuration.
 * calloc-shrink: an object filled with bytes and capabilities, shrunk in place, then freed; the
 * calloc that reuses its address must zero the whole extent again, the tail the shrink gave up
 * included (the slot is not clean, so no skip). */
static int calloc_shrink(void)
{
    char *live = malloc(32); require(live != NULL, "calloc-shrink live object");
    anchored = malloc(100); char *p = malloc(100); require(anchored && p, "calloc-shrink setup"); anchored_with(p);
    dirty(p, 100, live);
    uintptr_t address = (uintptr_t)p;
    char *q = realloc(p, 90);
    require(q && (uintptr_t)q == address, "calloc-shrink: the shrink is in place");
    free(q);
    static char *held[4096]; unsigned count = 0, found = 0, skipped = 0;
    while (count < 4096 && !found) {
        unsigned long before = __capstone_malloc_calloc_clean;
        held[count] = calloc(1, 100); require(held[count] != NULL, "calloc-shrink reuse search");
        zero_untagged(held[count], 100, "calloc-shrink: every object is zero and untagged");
        found = (uintptr_t)held[count++] == address; skipped = __capstone_malloc_calloc_clean != before;
    }
    require(found, "calloc-shrink: the shrunk object's slot was reused");
    require(!skipped, "calloc-shrink: a reused slot is zeroed, not skipped");
    printf("MALLOC_CALLOC_SHRINK reused_after=%u\n", count);
    puts("MALLOC_FAULT_READY:calloc-shrink"); puts("MALLOC_SURVIVED:calloc-shrink"); fflush(stdout);
    return 0;
}
/* calloc-geometry: a multi-slot group's object filled and freed, its mapping kept; a calloc of a
 * single-slot size of the same mapping length takes the mapping with other slot boundaries. Its
 * whole extent must read zero and untagged, and it skips the memset (the slot was scrubbed). */
static int calloc_geometry(void)
{
    char *live = malloc(32); require(live != NULL, "calloc-geometry live object");
    unsigned long length; size_t size;
    char *a = fresh_multi_slot(&length, &size);
    int bounds_off = whole_group(a);
    dirty(a, size, live);
    free_kept(a, "calloc-geometry: kept");
    size_t single = length - 68;   /* as recycle-geometry: room for the footer and a header up to 64 */
    unsigned long maps = __capstone_malloc_map_calls, hits = __capstone_malloc_recycle_hits;
    unsigned long skipped = __capstone_malloc_calloc_clean;
    char *c = calloc(1, single);
    require(c && __capstone_malloc_map_calls == maps && __capstone_malloc_recycle_hits == hits + 1,
            "calloc-geometry: a single-slot group took the kept mapping");
    struct capstone_malloc_slot *r = __capstone_malloc_find(c);
    require(r && __builtin_capstone_cap_get_end(r->parent) - __builtin_capstone_cap_get_base(r->parent) >= single,
            "calloc-geometry: the new group is a single slot");
    require(__capstone_malloc_calloc_clean == skipped + !bounds_off, "calloc-geometry: a scrubbed slot skips the memset");
    zero_untagged(c, single, "calloc-geometry: the new object is zero and untagged");
    printf("MALLOC_CALLOC_GEOMETRY length=%lu multi=%lu single=%lu\n", length, (unsigned long)size,
           (unsigned long)single);
    puts("MALLOC_FAULT_READY:calloc-geometry"); puts("MALLOC_SURVIVED:calloc-geometry"); fflush(stdout);
    return 0;
}
/* meta-reuse: a group binds slot records that are all zero (musl patch 0009 free_meta, heap.c
 * "Slot-record arrays"). A group's records are an array of 1, 2, 4, 8, 16 or 32, the smallest that
 * holds its slot count (cap_count), reused through one free list per size, so an array is bound
 * in turn by groups of different counts of one size. The bound count is the meta's own last_idx + 1,
 * read from its header (META_* above). Counting the records that name the meta would not do: a
 * record left uncleared still names it (lane S's first negative control read exactly that).
 * Records below the bound must name the meta; every other record of the array, up to its size,
 * must be all zero bytes (no capability, no tag). Arrays are seen bound with many slots, then by a
 * group with fewer (sizes of another class reuse them), then with more again; both transitions must
 * be observed on some array, and the invariant holds on every array scanned. */
#define META_SEEN 512
static struct { size_t array; unsigned bound, fewer, more; } seen[META_SEEN];
static unsigned nseen, transitions_few, transitions_many, scanned, bound_tags;
static unsigned scan_meta(void *p, const char *what)
{
    struct capstone_malloc_slot *r = __capstone_malloc_find(p);
    require(r && r->group && r->index < 32, what);
    struct capstone_malloc_slot *s = meta_slots(r, "meta-reuse: the meta points to the record's array");
    size_t meta = __builtin_capstone_cap_get_cursor(r->group);
    unsigned bound = (meta_word(r, META_LAST_IDX) & 31) + 1, count = meta_word(r, META_CAP_COUNT), size = 1;
    while (size < count) size <<= 1;
    require(r->index < bound, "meta-reuse: the record is inside its meta's bound");
    require(count == bound && size <= 32, "meta-reuse: the array was taken for the group's count");
    for (unsigned k = 0; k < size; ++k) {
        if (k < bound) {
            require(s[k].group && __builtin_capstone_cap_get_cursor(s[k].group) == meta,
                    "meta-reuse: a bound record names its meta");
            bound_tags += __builtin_capstone_cap_get_tag(s[k].parent) != 0; continue;
        }
        const unsigned char *b = (const unsigned char *)&s[k];
        for (size_t i = 0; i < sizeof s[k]; ++i) require(!b[i], "meta-reuse: a record past the bound is zero");
        for (size_t o = 0; o + 16 <= sizeof s[k]; o += 16)
            require(!__builtin_capstone_cap_get_tag(*(void *const volatile *)(b + o)), "meta-reuse: a record past the bound is untagged");
    }
    ++scanned;
    return bound;
}
static void note_meta(void *p)
{
    unsigned bound = scan_meta(p, "meta-reuse: slot record");
    struct capstone_malloc_slot *r = __capstone_malloc_find(p);
    size_t array = __builtin_capstone_cap_get_cursor(r - r->index);
    unsigned i = 0;
    while (i < nseen && seen[i].array != array) ++i;
    if (i == nseen) { if (nseen == META_SEEN) return; seen[nseen].array = array; seen[nseen++].bound = bound; return; }
    if (bound < seen[i].bound && !seen[i].fewer) { seen[i].fewer = 1; ++transitions_few; }
    if (bound > seen[i].bound && seen[i].fewer && !seen[i].more) { seen[i].more = 1; ++transitions_many; }
    seen[i].bound = bound;
}
/* Three rounds of 600 small objects, freed after each round. 28 and 200 bytes, then 44 and 120,
 * then 28 and 200 again: mallocng's counts for these classes grow with the class's usage
 * (small_cnt_tab, med_cnt_tab), and the classes of the middle round bind the arrays the first
 * round's groups released with other counts of the same array size. */
static int meta_reuse(void)
{
    enum { SMALL = 600 };
    static char *small[SMALL];
    static const size_t sizes[3][2] = { { 28, 200 }, { 44, 120 }, { 28, 200 } };
    for (int round = 0; round < 3; ++round) {
        for (unsigned i = 0; i < SMALL; ++i) {
            small[i] = malloc(sizes[round][i % 2]); require(small[i] != NULL, "meta-reuse small");
            note_meta(small[i]);
        }
        for (unsigned i = 0; i < SMALL; ++i) free(small[i]);
    }
    printf("MALLOC_META_REUSE arrays=%u scanned=%u many_to_few=%u few_to_many=%u bound_tags=%u\n", nseen, scanned,
           transitions_few, transitions_many, bound_tags);
    require(bound_tags > 0, "meta-reuse: the reader sees the bound records' tags");
    require(transitions_few > 0 && transitions_many > 0, "meta-reuse: an array went many -> few -> many slots");
    puts("MALLOC_FAULT_READY:meta-reuse"); puts("MALLOC_SURVIVED:meta-reuse"); fflush(stdout);
    return 0;
}
/* Direct group calls isolate group release from per-object revocation. This runs alone in
 * one thread; the public malloc cases above cover locking and mallocng's placement policy.
 * The runner requires shared backing when the image contains the coarse-region implementation.
 * The same source on older heaps exercises dedicated mappings, with that geometry reported. */
/* The shared heap supplies the strong counter. Older heaps have no regions to unmap. */
unsigned long __capstone_malloc_coarse_unmapped __attribute__((weak));
static int coarse_case(const char *mode, int page_extents)
{
    enum { GROUPS = 20, SPAN = 3360 };
    static struct capstone_malloc_slot records[GROUPS][32];
    static char *groups[GROUPS];
    static char live[16];
    malloc_trim(0);
    unsigned long maps = __capstone_malloc_map_calls;
    size_t span = 0;
    for (unsigned i = 0; i < GROUPS; ++i) {
        groups[i] = __capstone_malloc_group_map(records[i], 4096, 176, 19);
        require(groups[i] && groups[i] != (void *)-1, "coarse: group mapping");
        size_t length = __builtin_capstone_cap_get_end(groups[i]) -
                        __builtin_capstone_cap_get_base(groups[i]);
        if (!i) span = length;
        require(length == span && (span == SPAN || span == 4096), "coarse: group bounds");
    }
    unsigned long mapped = __capstone_malloc_map_calls - maps;
    int shared = mapped == 2;
    require(mapped == (shared ? 2 : GROUPS), "coarse: backing map count");
    require(span == (shared && !page_extents ? SPAN : 4096), "coarse: selected extent geometry");
    size_t a = __builtin_capstone_cap_get_cursor(groups[0]);
    size_t b = __builtin_capstone_cap_get_cursor(groups[1]);
    if (shared) {
        require(!(a & 65535), "coarse: aligned region");
        unsigned first = page_extents ? 16 : 19;
        for (unsigned i = 0; i < first; ++i)
            require(__builtin_capstone_cap_get_cursor(groups[i]) == a + i * span,
                    "coarse: consecutive extents in first region");
        size_t last = __builtin_capstone_cap_get_cursor(groups[first]);
        require(!(last & 65535) && last != a, "coarse: second region");
        require(((a + span - 1) / 4096 == b / 4096) == !page_extents,
                "coarse: selected page sharing");
    }
    printf("MALLOC_COARSE_LAYOUT %s groups=20 maps=%lu span=%lu\n",
           shared ? "shared" : "dedicated", mapped, (unsigned long)span);
    groups[1][16] = 77;
    char *stale = capstone_cap_derive(groups[0], 16, 16);
    unsigned written = fill_caps(groups[0], 0, span, live);
    require(written == span / 16 && count_tags(groups[0], 0, span) == written,
            "coarse: tag reader sees every stored capability");
    unsigned long unmapped = __capstone_malloc_coarse_unmapped;
    __capstone_malloc_group_unmap(groups[0], 4096);
    /* Dedicated mappings must stay cached until reuse. A shared region must survive trim
     * while a sibling is live, including the page that the released extent shared with it. */
    if (shared) require(malloc_trim(0) == 0 && __capstone_malloc_coarse_unmapped == unmapped,
                        "coarse: trim retains partial regions");
    require(groups[1][16] == 77, "coarse: sibling survives release and trim");
    memset(records[0], 0, sizeof records[0]);
    groups[0] = __capstone_malloc_group_map(records[0], 4096, 304, 11);
    require(groups[0] && groups[0] != (void *)-1 &&
            __builtin_capstone_cap_get_cursor(groups[0]) == a &&
            __builtin_capstone_cap_get_end(groups[0]) - a == span,
            "coarse: same extent reused with different geometry");
    /* With object bounds off, header and untaken slots are reachable immediately. */
    if (records[0][0].issued) {
        unsigned tags = count_tags(groups[0], 0, span), bytes = 0;
        for (size_t o = 0; o < span; ++o) bytes += groups[0][o] != 0;
        printf("MALLOC_COARSE_WHOLE tags=%u bytes=%u\n", tags, bytes);
        require(!tags && !bytes, "coarse: whole extent clean before any slot is taken");
    }
    if (!strcmp(mode, "coarse-stale") || !strcmp(mode, "coarse-bounds")) {
        printf("MALLOC_FAULT_READY:%s\n", mode); fflush(stdout);
        fault_store(!strcmp(mode, "coarse-stale") ? stale : groups[0] + span);
        return survived(mode, 1);
    }
    unsigned tags = 0, bytes = 0;
    for (unsigned i = 0; i < 11; ++i) {
        char *p = __capstone_malloc_slot_take(&records[0][i]);
        require(p && (records[0][i].taken_clean || records[0][i].issued),
                "coarse: reused slot is clean for calloc");
        tags += count_tags(p, 0, 304);
        for (unsigned o = 0; o < 304; ++o) bytes += p[o] != 0;
    }
    printf("MALLOC_COARSE_SCRUB tags=%u bytes=%u slots=11 stride=304\n", tags, bytes);
    require(!tags && !bytes, "coarse: geometry-changing reuse clears bytes and tags");
    require(groups[1][16] == 77, "coarse: sibling survives reuse scrub");
    for (unsigned i = 0; i < GROUPS; ++i) __capstone_malloc_group_unmap(groups[i], 4096);
    require(malloc_trim(0) == 1 && malloc_trim(0) == 0, "coarse: trim returns empty backing");
    unsigned long released = __capstone_malloc_coarse_unmapped - unmapped;
    require(released == (shared ? 2 : 0), "coarse: trim returns both empty regions");
    printf("MALLOC_COARSE_TRIM empty_backing_returned=1 regions=%lu\n", released);
    printf("MALLOC_SURVIVED:%s\n", mode); fflush(stdout);
    return 0;
}
#endif
int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "ok";
    int may_survive = argc > 2 && !strcmp(argv[2], "may-survive");
#ifdef __CAPSTONE_PURECAP__
    if (!strcmp(mode, "coarse-reuse") || !strcmp(mode, "coarse-stale") ||
        !strcmp(mode, "coarse-bounds"))
        return coarse_case(mode, argc > 2 && !strcmp(argv[2], "page-extents"));
    if (!strncmp(mode, "recycle-", 8)) return recycle_case(mode, may_survive);
    if (!strcmp(mode, "calloc-clean")) return calloc_clean();
    if (!strcmp(mode, "calloc-shrink")) return calloc_shrink();
    if (!strcmp(mode, "calloc-geometry")) return calloc_geometry();
    if (!strcmp(mode, "meta-reuse")) return meta_reuse();
#endif
    if (!strcmp(mode, "hot")) { hot(0); return 0; }
    if (!strcmp(mode, "hot-service-control")) { hot(1); return 0; }
    if (!strcmp(mode, "population")) { population(); return 0; }
    if (!strcmp(mode, "churn")) { churn(); return 0; }
    if (!strcmp(mode, "remap-align")) { remap_align(); return 0; }
    if (!strcmp(mode, "predicate")) { predicate(); return 0; }
    if (!strcmp(mode, "align-probe")) { align_probe(); return 0; }
    if (!strcmp(mode, "ok")) { basic(); tagged(); return 0; }
    if (!strcmp(mode, "reuse-stale") || !strcmp(mode, "reuse-free")) {
        /* Let upstream offset cycling choose reuse; never force its policy. */
        anchored = malloc(100);
        char *old = malloc(100);
        require(anchored && old, "reuse setup"); anchored_with(old);
        uintptr_t address = (uintptr_t)old;
        saved = old; free(old);
        void *held[4096]; unsigned count = 0;
        while (count < 4096) {
            held[count] = malloc(100); require(held[count] != NULL, "reuse search");
            if ((uintptr_t)held[count++] == address) break;
        }
        require(count && (uintptr_t)held[count - 1] == address, "same address reuse trigger");
        puts(!strcmp(mode, "reuse-stale") ? "MALLOC_FAULT_READY:reuse-stale" :
                                           "MALLOC_FAULT_READY:reuse-free");
        fflush(stdout);
        if (!strcmp(mode, "reuse-free")) free(saved); else fault_store(saved);
        return survived(mode, may_survive);
    }
    if (!strcmp(mode, "remap-stale")) {
        char *old = malloc(200000); require(old != NULL, "remap stale setup"); saved = old;
        char *next = realloc(old, 400000); require(next != NULL, "remap stale growth");
        puts("MALLOC_FAULT_READY:remap-stale"); fflush(stdout); fault_store(saved);
        return survived(mode, may_survive);
    }
    anchored = malloc(100); require(anchored != NULL, "anchor allocation");
    char *p = malloc(100); require(p != NULL, "fault allocation"); saved = p;
    anchored_with(p);
#ifdef __CAPSTONE_PURECAP__
    /* THE NESTED-ALLOCATOR PAIR (heap.c __capstone_malloc_derive / _reclaim), in the shape PHP's
     * emalloc uses it: a 48-byte header the allocator keeps, a payload child the caller gets.
     * nested-ok is the positive control and must complete: derive, write, reclaim gives the object
     * back at its own address, the header is still writable, a fresh child over the same bytes
     * works, and the object frees. The three denials: the reclaimed child faults when used
     * (nested-stale, 25 at the store); a grandchild the caller derived for itself is not the
     * allocator's child and is refused (nested-grandchild, 29 at cap_malloc_reclaim); a child
     * reclaimed twice is dead the second time (nested-double, 25 at cap_malloc_reclaim). */
    if (!strncmp(mode, "nested-", 7)) {
        char *child = __capstone_malloc_derive(p, 48, 40);
        require(child != NULL, "nested derive");
        require(__builtin_capstone_cap_get_base(child) == (uintptr_t)p + 48 &&
                __builtin_capstone_cap_get_end(child) == (uintptr_t)p + 88, "nested child bounds");
        memset(child, 'c', 40);
        if (!strcmp(mode, "nested-grandchild")) {
            char *grand = __capstone_malloc_derive(child, 0, 16);
            require(grand != NULL, "nested grandchild derive");
            puts("MALLOC_FAULT_READY:nested-grandchild"); fflush(stdout);
            __capstone_malloc_reclaim(grand, 48);
            return survived(mode, may_survive);
        }
        require(__capstone_malloc_object(child, 48) == p, "nested object lookup finds the block");
        char *back = __capstone_malloc_reclaim(child, 48);
        require(back != NULL && __builtin_capstone_cap_get_cursor(back) == (uintptr_t)p,
                "nested reclaim returns the object at its start");
        if (!strcmp(mode, "nested-stale")) {
            puts("MALLOC_FAULT_READY:nested-stale"); fflush(stdout); fault_store(child);
            return survived(mode, may_survive);
        }
        if (!strcmp(mode, "nested-double")) {
            puts("MALLOC_FAULT_READY:nested-double"); fflush(stdout);
            __capstone_malloc_reclaim(child, 48);
            return survived(mode, may_survive);
        }
        if (!strcmp(mode, "nested-ok")) {
            char *again;
            puts("MALLOC_FAULT_READY:nested-ok"); fflush(stdout);
            memset(back, 'h', 48);                        /* the header, after the child ended */
            again = __capstone_malloc_derive(back, 48, 40); /* the cache hit: a fresh child */
            require(again != NULL, "nested re-derive");
            memset(again, 'd', 40);
            require(__capstone_malloc_reclaim(again, 48) == back, "nested second reclaim");
            free(back);
            puts("MALLOC_SURVIVED:nested-ok"); fflush(stdout);
            return 0;
        }
        return 2;
    }
#endif
    if (!strcmp(mode, "inplace-stale")) {
        uintptr_t old = (uintptr_t)p;
        p = realloc(p, 101);
        require(p && (uintptr_t)p == old, "in-place stale trigger");
        puts("MALLOC_FAULT_READY:inplace-stale"); fflush(stdout);
        fault_store(saved);
    } else if (!strcmp(mode, "stale")) {
        free(p); puts("MALLOC_FAULT_READY:stale"); fflush(stdout);
        fault_store(saved);
    } else if (!strcmp(mode, "stale-free")) {
        free(p); puts("MALLOC_FAULT_READY:stale-free"); fflush(stdout); free(saved);
    } else if (!strcmp(mode, "bounds")) {
        /* This request is not exactly representable in the compressed codec. */
        free(p); p = malloc(200003); require(p != NULL, "precise allocation");
        saved = p; puts("MALLOC_FAULT_READY:bounds"); fflush(stdout);
#ifdef __CAPSTONE_PURECAP__
        /* The published bounds are rounded up to what the codec represents, so the first byte
         * past the request can lie inside them; the store goes to the capability's real end. */
        size_t reach = __builtin_capstone_cap_get_end(saved) - __builtin_capstone_cap_get_cursor(saved);
#else
        size_t reach = 200003;
#endif
        fault_store((volatile char *)saved + reach);
    } else if (!strcmp(mode, "interior-free")) {
        puts("MALLOC_FAULT_READY:interior-free"); fflush(stdout); free(p + 1);
    }
    else return 2;
    return survived(mode, may_survive);
}
