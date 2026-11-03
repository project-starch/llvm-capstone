/* EXPERIMENT (lane M, never lands): page-movement torture test for a tag-aware guest Linux.
 *
 * A domain with live and stale capabilities spread over many pages: a heap arena of SLOT_PAGES
 * pages and a .bss array of BSS_PAGES pages (an eager grant), each granule a pointer slot of a
 * known class, plus objects that point at each other. Between rounds the domain yields (the
 * experiment launcher forks a sleeping child on sched_yield when CAPSTONE_VM_TORTURE_FORK is
 * set, so every page becomes copy-on-write) and sleeps (the guest driver compacts memory
 * meanwhile); then it writes one scalar word in every page (breaking COW while the child
 * lives), rewrites a share of its live slots with capability stores, and verifies every slot:
 *
 *   live    tagged (LCC type NONLIN), same 16 bytes, and the pointee's id reads back;
 *   stale   a pointer to a FREED object: reads as recorded, same bytes (never dereferenced
 *           here; mode "stale" dereferences one and must die);
 *   scalar  a granule that held a pointer and was overwritten by a scalar store: untagged,
 *           same bytes (a tag must not come back);
 *   zero    never written: untagged, zero.
 *
 * Usage: torture [rounds [sleep_ms]]     -> exit 0 only if every check of every round held
 *        torture stale                   -> dereferences a stale pointer; must not return
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sched.h>

#define SLOT_PAGES 2048            /* 8 MiB of slots in a heap arena */
#define BSS_PAGES 256              /* 1 MiB of slots in .bss (an eager image grant) */
#define SLOTS_PER_PAGE 256
#define NOBJ 16384

enum { ZERO, LIVE, STALE, SCALAR, SCRATCH };

struct obj { uint64_t id; struct obj *next; uint64_t pad[3]; };

static void *bss_slots[BSS_PAGES * SLOTS_PER_PAGE] __attribute__((aligned(4096)));
static struct obj *objs[NOBJ];
static uint8_t freed[NOBJ];

static inline unsigned long cap_type(void *p) {
    unsigned long t;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x04, %0, %1, x1" : "=r"(t) : "r"(p));
    return t;
}

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint64_t next_rand(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

struct region { void **slots; size_t n; uint8_t *cls; uint32_t *target; uint64_t *bytes; uint8_t *type; };

static void fill(struct region *r) {
    for (size_t i = 0; i < r->n; i++) {
        unsigned k = (unsigned)(next_rand() % NOBJ);
        unsigned c = (unsigned)(next_rand() % 8);
        if (i % SLOTS_PER_PAGE == SLOTS_PER_PAGE - 1) c = 99;            /* scratch */
        r->target[i] = k;
        if (c == 99) r->cls[i] = SCRATCH;
        else if (c < 4) { r->slots[i] = freed[k] ? NULL : objs[k]; r->cls[i] = freed[k] ? ZERO : LIVE; }
        else if (c < 6) { r->cls[i] = ZERO; }
        else if (c == 6) { r->slots[i] = objs[k]; r->cls[i] = STALE; }   /* freed below if not yet */
        else { r->slots[i] = objs[k];
               ((volatile uint64_t *)&r->slots[i])[0] = 0x5ca1a400000000ull | i;  /* scalar over a pointer */
               r->cls[i] = SCALAR; }
    }
}

/* The reference state: bytes and LCC type of every slot, taken once before any round. A live
 * slot must read NONLIN, a scalar or zero slot untagged; a stale slot is recorded as read
 * (whether a load of a pointer with an ended lifetime comes back tagged is the ISA's business,
 * not this test's): after every round each slot must read exactly as recorded. */
static long snapshot(struct region *r) {
    long setup_bad = 0;
    for (size_t i = 0; i < r->n; i++) {
        volatile uint64_t *w = (volatile uint64_t *)&r->slots[i];
        r->bytes[2 * i] = w[0]; r->bytes[2 * i + 1] = w[1];
        r->type[i] = (uint8_t)cap_type(r->slots[i]);
        if ((r->cls[i] == LIVE && r->type[i] != 1) ||
            ((r->cls[i] == SCALAR || r->cls[i] == ZERO) && r->type[i] != 7)) {
            if (setup_bad < 5) printf("SETUP slot %zu class %d type %u\n", i, r->cls[i], r->type[i]);
            setup_bad++;
        }
    }
    return setup_bad;
}

struct counts { long live, stale, scalar, zero, bad_type, bad_bytes, bad_deref; };

static void verify(struct region *r, struct counts *c, const char *name, int round) {
    for (size_t i = 0; i < r->n; i++) {
        if (r->cls[i] == SCRATCH) continue;
        volatile uint64_t *w = (volatile uint64_t *)&r->slots[i];
        void *p = r->slots[i];
        unsigned long t = cap_type(p);
        if (w[0] != r->bytes[2 * i] || w[1] != r->bytes[2 * i + 1]) {
            if (c->bad_bytes < 5) printf("round %d %s slot %zu class %d BYTES %016llx%016llx was %016llx%016llx\n",
                                         round, name, i, r->cls[i], (unsigned long long)w[1], (unsigned long long)w[0],
                                         (unsigned long long)r->bytes[2 * i + 1], (unsigned long long)r->bytes[2 * i]);
            c->bad_bytes++;
        }
        if (t != r->type[i]) {
            if (c->bad_type < 5) printf("round %d %s slot %zu class %d TYPE %lu\n", round, name, i, r->cls[i], t);
            c->bad_type++;
            continue;
        }
        switch (r->cls[i]) {
        case LIVE: c->live++;
            if (((struct obj *)p)->id != r->target[i]) {
                if (c->bad_deref < 5) printf("round %d %s slot %zu DEREF id %llu want %u\n", round, name, i,
                                             (unsigned long long)((struct obj *)p)->id, r->target[i]);
                c->bad_deref++;
            }
            break;
        case STALE: c->stale++; break;
        case SCALAR: c->scalar++; break;
        default: c->zero++; break;
        }
    }
}

static void touch_pages(struct region *r, int round) {
    /* one scalar store per page (the scratch granule): breaks COW while the forked child lives */
    for (size_t i = SLOTS_PER_PAGE - 1; i < r->n; i += SLOTS_PER_PAGE)
        ((volatile uint64_t *)&r->slots[i])[0] = 0x7000000000000000ull | ((uint64_t)round << 32) | i;
    /* capability stores of the same values into a share of the live slots (STC on moved pages) */
    for (size_t i = (size_t)round % 7; i < r->n; i += 7)
        if (r->cls[i] == LIVE) { void *v = r->slots[i]; ((void *volatile *)r->slots)[i] = v; }
}

static void sleep_ms(long ms) { struct timespec ts = {ms / 1000, (ms % 1000) * 1000000}; nanosleep(&ts, NULL); }

int main(int argc, char **argv) {
    int rounds = 20; long ms = 1500;
    int stale_mode = argc > 1 && !strcmp(argv[1], "stale");
    if (argc > 1 && !stale_mode) rounds = atoi(argv[1]);
    if (stale_mode) rounds = argc > 2 ? atoi(argv[2]) : 0;   /* stale ROUNDS: move pages first */
    else if (argc > 2) ms = atol(argv[2]);
    for (unsigned k = 0; k < NOBJ; k++) {
        objs[k] = malloc(sizeof(struct obj));
        if (!objs[k]) { printf("malloc failed\n"); return 2; }
        objs[k]->id = k;
    }
    for (unsigned k = 0; k < NOBJ; k++) objs[k]->next = objs[(k * 7919u + 1) % NOBJ];
    struct region h, b;
    h.n = (size_t)SLOT_PAGES * SLOTS_PER_PAGE; b.n = (size_t)BSS_PAGES * SLOTS_PER_PAGE;
    h.slots = malloc(h.n * sizeof(void *)); b.slots = bss_slots;
    h.cls = malloc(h.n); b.cls = malloc(b.n);
    h.target = malloc(h.n * 4); b.target = malloc(b.n * 4);
    h.bytes = malloc(h.n * 16); b.bytes = malloc(b.n * 16);
    h.type = malloc(h.n); b.type = malloc(b.n);
    if (!h.slots || !h.cls || !b.cls || !h.target || !b.target || !h.bytes || !b.bytes || !h.type || !b.type) { printf("malloc failed\n"); return 2; }
    memset(h.slots, 0, h.n * sizeof(void *));
    fill(&h); fill(&b);
    /* free every object a STALE slot names (and only those): their lifetimes end */
    long nfreed = 0;
    for (int pass = 0; pass < 2; pass++) {
        struct region *r = pass ? &b : &h;
        for (size_t i = 0; i < r->n; i++)
            if (r->cls[i] == STALE && !freed[r->target[i]]) { free(objs[r->target[i]]); freed[r->target[i]] = 1; nfreed++; }
    }
    /* a LIVE slot whose object got freed afterwards is STALE too */
    for (int pass = 0; pass < 2; pass++) {
        struct region *r = pass ? &b : &h;
        for (size_t i = 0; i < r->n; i++) if (r->cls[i] == LIVE && freed[r->target[i]]) r->cls[i] = STALE;
    }
    /* object-to-object pointers: rewrite to live successors so objects stay a live graph */
    for (unsigned k = 0; k < NOBJ; k++) if (!freed[k]) {
        unsigned n = (k + 1) % NOBJ; while (freed[n]) n = (n + 1) % NOBJ;
        objs[k]->next = objs[n];
    }
    long setup_bad = snapshot(&h) + snapshot(&b);
    long stale_tagged = 0;
    for (size_t i = 0; i < h.n; i++) if (h.cls[i] == STALE && h.type[i] == 1) stale_tagged++;
    if (setup_bad) { printf("TORTURE SETUP-FAIL %ld\n", setup_bad); return 5; }
    if (stale_mode && rounds == 0) {
        for (size_t i = 0; i < h.n; i++) if (h.cls[i] == STALE) {
            printf("STALE-DEREF slot %zu type %lu\n", i, cap_type(h.slots[i])); fflush(stdout);
            volatile uint64_t v = ((struct obj *)h.slots[i])->id;
            printf("REVIVED: stale pointer read id %llu\n", (unsigned long long)v);
            return 3;
        }
        return 4;
    }
    printf("READY pid=%d heap=%p heap_bytes=%zu bss=%p bss_bytes=%zu objs_freed=%ld stale_tagged_heap=%ld\n", getpid(),
           (void *)h.slots, h.n * sizeof(void *), (void *)b.slots, b.n * sizeof(void *), nfreed, stale_tagged);
    fflush(stdout);
    long total_bad = 0;
    for (int r = 0; r < rounds; r++) {
        sched_yield();          /* the experiment launcher forks a sleeping child here */
        sleep_ms(ms);           /* the guest driver compacts memory meanwhile */
        touch_pages(&h, r); touch_pages(&b, r);
        sleep_ms(ms / 3);
        struct counts c = {0};
        verify(&h, &c, "heap", r); verify(&b, &c, "bss", r);
        /* the object graph: walk 4096 steps of next pointers from a live object */
        unsigned k = 0; while (freed[k]) k++;
        struct obj *o = objs[k]; long walk_bad = 0;
        for (int s = 0; s < 4096; s++) { struct obj *n = o->next; if (cap_type(n) != 1) { walk_bad++; break; } o = n; }
        long bad = c.bad_type + c.bad_bytes + c.bad_deref + walk_bad;
        total_bad += bad;
        printf("round %d live %ld stale %ld scalar %ld zero %ld bad_type %ld bad_bytes %ld bad_deref %ld walk_bad %ld\n",
               r, c.live, c.stale, c.scalar, c.zero, c.bad_type, c.bad_bytes, c.bad_deref, walk_bad);
        fflush(stdout);
    }
    printf("TORTURE %s total_bad=%ld\n", total_bad ? "FAIL" : "PASS", total_bad);
    if (stale_mode) {   /* after the rounds: the pages moved; a stale pointer must still fault */
        fflush(stdout);
        for (size_t i = 0; i < h.n; i++) if (h.cls[i] == STALE && h.type[i] == 1) {
            printf("STALE-DEREF slot %zu type %lu\n", i, cap_type(h.slots[i])); fflush(stdout);
            volatile uint64_t v = ((struct obj *)h.slots[i])->id;
            printf("REVIVED: stale pointer read id %llu\n", (unsigned long long)v);
            return 3;
        }
        return 4;
    }
    return total_bad ? 1 : 0;
}
