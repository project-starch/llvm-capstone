#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#ifdef __CAPSTONE_PURECAP__
#include "vm.h"
#endif
static void *volatile saved;
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
    require(!realloc(p, SIZE_MAX) && errno == ENOMEM && p[99] == 42,
            "failed realloc preserves original");
    free(p);
    p = malloc(200000); require(p != NULL, "large original"); p[199999] = 33;
#ifdef __CAPSTONE_PURECAP__
    errno = 0;
    require(!realloc(p, 512UL<<20) && errno == ENOMEM && p[199999] == 33,
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
int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "ok";
    if (!strcmp(mode, "hot")) { hot(0); return 0; }
    if (!strcmp(mode, "hot-service-control")) { hot(1); return 0; }
    if (!strcmp(mode, "population")) { population(); return 0; }
    if (!strcmp(mode, "churn")) { churn(); return 0; }
    if (!strcmp(mode, "remap-align")) { remap_align(); return 0; }
    if (!strcmp(mode, "predicate")) { predicate(); return 0; }
    if (!strcmp(mode, "ok")) { basic(); tagged(); return 0; }
    if (!strcmp(mode, "reuse-stale") || !strcmp(mode, "reuse-free")) {
        /* Let upstream offset cycling choose reuse; never force its policy. */
        char *anchor = malloc(100), *old = malloc(100);
        require(anchor && old, "reuse setup");
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
        return 1;
    }
    if (!strcmp(mode, "remap-stale")) {
        char *old = malloc(200000); require(old != NULL, "remap stale setup"); saved = old;
        char *next = realloc(old, 400000); require(next != NULL, "remap stale growth");
        puts("MALLOC_FAULT_READY:remap-stale"); fflush(stdout); fault_store(saved); return 1;
    }
    char *p = malloc(100); require(p != NULL, "fault allocation"); saved = p;
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
        fault_store((volatile char *)saved + 200003);
    } else if (!strcmp(mode, "interior-free")) {
        puts("MALLOC_FAULT_READY:interior-free"); fflush(stdout); free(p + 1);
    }
    else return 2;
    require(0, "denial survived"); return 1;
}
