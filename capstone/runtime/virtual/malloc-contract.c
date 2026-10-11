#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#ifdef __CAPSTONE_PURECAP__
#include "vm.h"
extern unsigned long __capstone_sublet_malloc_linear(size_t, sublet_cap *);
extern void __capstone_sublet_free_linear(unsigned long);
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
static void require(int ok, const char *what)
{ if (!ok) { fprintf(stderr, "MALLOC_FAIL:%s\n", what); exit(1); } }
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
    require(container && (uintptr_t)container != old, "moving realloc trigger");
    require(container[0][0] == 71, "moving realloc preserves capability");
    free(container[0]); free(container);
#ifdef __CAPSTONE_PURECAP__
    sublet_cap owner;
    unsigned long base = __capstone_sublet_malloc_linear(32, &owner);
    require(base != 0, "linear allocation");
    container = calloc(1, 64); require(container != NULL, "linear container");
    sublet_move(&owner, (sublet_cap *)container);
    container = realloc(container, 65536); require(container != NULL, "linear container move");
    require(sublet_base((sublet_cap *)container) == base, "linear tag transfer");
    __capstone_sublet_free_linear(base); free(container);
#endif
    puts("MALLOC_TAG_COPY_OK");
}
static void population(void)
{
    enum { COUNT = 70000 };
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
int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "ok";
    if (!strcmp(mode, "hot")) { hot(0); return 0; }
    if (!strcmp(mode, "hot-service-control")) { hot(1); return 0; }
    if (!strcmp(mode, "population")) { population(); return 0; }
    if (!strcmp(mode, "churn")) { churn(); return 0; }
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
