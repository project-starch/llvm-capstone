/* mallocng runs in the application's Capstone context. This file supplies
 * ownership operations, not allocation policy. Only VM growth/release and
 * node pressure cross the existing service boundary. */
#include <errno.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <malloc_capstone.h>
#include <capstone/capability.h>
#include "vm.h"
#include "vm-abi.h"
#include <capstone/representable.h>

/* The two switches of the spatial-versus-temporal measurement (runtime/application CMake
 * options CAPSTONE_HEAP_LIFETIMES and CAPSTONE_HEAP_SCRUB_FREE, recorded in sdk.json). The
 * default, LIFETIMES=1 and SCRUB_FREE=0, is the profile. LIFETIMES=0: a public object, and a
 * nested group, is a SHRINK copy of the slot's authority over the same bytes instead of a
 * CDERIVE child, so free revokes nothing and only a mapping's retirement (munmap, mremap)
 * ends a stale capability; the lifetime per mapping (mapping_group) remains, so ensure_ids
 * stays. SCRUB_FREE=0: free leaves the payload's bytes and tags in place (calloc still
 * zeroes), so a reused slot's previous contents are the next owner's to read; a stale
 * capability among them still faults on use, because its lifetime ended. SCRUB_FREE=1 zeroes
 * the payload before the CREVOKE. */
#ifndef CAPSTONE_HEAP_LIFETIMES
#define CAPSTONE_HEAP_LIFETIMES 1
#endif
#ifndef CAPSTONE_HEAP_SCRUB_FREE
#define CAPSTONE_HEAP_SCRUB_FREE 0
#endif
/* The third switch (CMake option CAPSTONE_HEAP_OBJECT_BOUNDS, sdk.json heap_object_bounds), for the
 * T0a arm of the microbenchmark study. OBJECT_BOUNDS=0: malloc, calloc, realloc and aligned_alloc
 * return the GROUP's authority -- the capability bind_slots was handed, over the whole mapping (or the
 * whole nested group) -- with its cursor moved to the object, instead of a child over the object's
 * own bytes. No per-object or per-slot bounds reach the program. Under LIFETIMES=0 a slot's `issued`
 * field is otherwise dead, so it holds that group capability for the slot's whole life: bind_slots
 * sets it, and nothing clears it. The slot parents, the index key (the object's cursor) and every
 * pointer mallocng sees are unchanged, so free and realloc find the object exactly as before. At 1
 * (the default) this file compiles to the same object as before the switch existed. */
#ifndef CAPSTONE_HEAP_OBJECT_BOUNDS
#define CAPSTONE_HEAP_OBJECT_BOUNDS 1
#endif
#ifndef CAPSTONE_HEAP_COARSE_PAGES
#define CAPSTONE_HEAP_COARSE_PAGES 0
#endif
#if !CAPSTONE_HEAP_OBJECT_BOUNDS && CAPSTONE_HEAP_LIFETIMES
#error "CAPSTONE_HEAP_OBJECT_BOUNDS=0 requires CAPSTONE_HEAP_LIFETIMES=0: a per-object lifetime is a CDERIVE child, which has bounds of its own"
#endif

/* The address index: INDEX_BUCKETS chains of slot records, keyed by an object's start. The bucket
 * count bounds nothing; chains grow with the live population. 1,024 buckets (lane Y, 2026-11-03),
 * not 4,096: the table is 16-byte capability heads in .bss, and the multiplicative hash spreads even
 * a few hundred live objects over every one of its pages, so 4,096 buckets faulted all 16 pages
 * (SPEC's gcc test) where 1,024 fault 4. Since one lookup per call (below), only free, realloc,
 * malloc_usable_size and the trio search a chain, and slot_release and a moved key unlink one, so a
 * load four times higher costs a few chain steps on those calls and none on allocation. */
#define INDEX_BITS 10U
#define INDEX_BUCKETS (1U << INDEX_BITS)
static struct capstone_malloc_slot *index_heads[INDEX_BUCKETS];
static sublet_cap pending_mapping;
static unsigned heap_mutex;
enum { SLOT_INTERNAL, SLOT_PUBLIC, SLOT_NESTED };
static void lock_heap(void)
{
    while (__atomic_exchange_n(&heap_mutex, 1, __ATOMIC_ACQUIRE))
        if (__capstone_vm_wait()) __builtin_trap();
}
static void unlock_heap(void)
{ __atomic_store_n(&heap_mutex, 0, __ATOMIC_RELEASE); }
static unsigned bucket(size_t key)
{ return ((key >> 4) * 11400714819323198485UL) >> (64U - INDEX_BITS); }
static void unindex(struct capstone_malloc_slot *r)
{
    if (!r->key) return;
    struct capstone_malloc_slot **p = &index_heads[bucket(r->key)];
    while (*p && *p != r) p = &(*p)->next;
    if (!*p) __builtin_trap();
    *p = r->next;
    r->next = NULL; r->key = 0;
}
/* THE ONE LOOKUP PER CALL (lane S, 2026-11-03). The index maps an object's start to its slot
 * record; it is how the runtime gets from a pointer the PROGRAM passed (free, realloc,
 * malloc_usable_size, the trio) to the allocator's own authority over that object, which the
 * program's capability does not carry. Each such call looks the pointer up once (checked() below)
 * and hands the record on; mallocng's representation hooks take the record as an argument
 * (cap_prefix(p, s), musl patch 0009) instead of finding it again. Before, mallocng found the
 * record of its own backend pointer by this hash up to seven times per call -- 38,657 calls on
 * SPEC's gcc test, six per enframe and four per free, against 7,000 malloc/free/realloc calls --
 * although enframe holds the record (g->cap_slots[idx]) and free receives it from here.
 * Allocation needs no lookup at all: enframe keys the record it took as its last step, and
 * publish takes that record (`keyed`), checking its key against the returned pointer. */
static struct capstone_malloc_slot *keyed;   /* the record slot_key keyed last; under the heap lock */
struct capstone_malloc_slot *__capstone_malloc_find(const void *p)
{
    size_t key = __builtin_capstone_cap_get_cursor((void *)p);
    for (struct capstone_malloc_slot *r = index_heads[bucket(key)]; r; r = r->next)
        if (r->key == key) return r;
    return NULL;
}
void __capstone_malloc_slot_key(struct capstone_malloc_slot *r, void *p, size_t n)
{
    size_t key = __builtin_capstone_cap_get_cursor(p);
    keyed = r;
    r->nominal = n;
    if (r->key == key) return;   /* an in-place realloc keeps its start */
    unindex(r);
    r->key = key;
    unsigned b = bucket(key);
    r->next = index_heads[b]; index_heads[b] = r;
}
/* The record of the object mallocng just returned at p: the one slot_key keyed last, which is
 * p's for every path that returns an object (enframe, an in-place or remapping realloc and
 * aligned_alloc key it as their last step). The key comparison makes a mismatch a lookup, not
 * a wrong record. */
static struct capstone_malloc_slot *record_of_result(void *p)
{
    struct capstone_malloc_slot *r = keyed;
    if (!r || r->key != __builtin_capstone_cap_get_cursor(p)) r = __capstone_malloc_find(p);
    return r;
}
/* Q-21: THE APPLICATION RUNTIME NEVER READS A CSR FOR AVAILABILITY.
 *
 * This used to begin `csrr n, 0xcc0` (urevavail) and return early when n >= need +
 * CV_NODE_RESERVE, purely to SKIP the service call below. The read is a QEMU-only construct:
 * urevavail does not exist on the RTL (csr_regfile.sv falls through to illegal), so every image
 * carrying it passes under QEMU and takes an illegal-instruction fault on silicon at its FIRST
 * malloc. vp2's boot 3-T1 read exactly that -- cause 2 at contract.dom VA 0x20ccc, which is this
 * instruction inside malloc, with tval the instruction's own encoding. Q-21's rule had been applied
 * to the kernel module and not here, and the application runtime cannot see the table header at
 * all, so it had no second source to fall back on.
 *
 * Dropping the pre-check is BEHAVIOUR-PRESERVING, not a trade: the module's ensure_nodes() opens
 * with `if (available >= required) return 0;` over the table header, which is the authoritative
 * copy of the same quantity the CSR mirrored. So the decision is unchanged and merely made one
 * level in. The cost is a service call on the path that previously returned without one.
 *
 * CORRECTION (2026-10-26): that cost was PER ALLOCATION, not per mapping as this comment used to say.
 * malloc, realloc and aligned_alloc each called ensure_ids(256) unconditionally, so every one of them
 * left the context for a full delegated round (thousands of instructions; #282 profiles one round at
 * 6,890) in both measurement arms, T1 included, where a malloc takes no record at all.
 *
 * So the heap now counts instead of asking. A successful call leaves at least IDS_ASK records
 * available (the module's ensure_nodes guarantees available >= required); the heap then takes up to
 * IDS_ASK - CV_NODE_RESERVE - 1 of them, one per CDERIVE it issues (take_id), before it asks again,
 * so it never drives the table to the hardware's pressure threshold on its own. That is one service
 * call per ~255 records taken, and none for an allocation that takes none (T1's SHRINK). Records
 * taken outside the heap (a ported program's own nested allocator, CSNODE) are not counted here;
 * they are what the hardware's kind-5 pressure escape is for, which the module answers with
 * ensure_nodes and a retry (capstone_vm.c, CV_STEP, event 5) -- the backstop this hint never
 * replaced. Frees do not refill the count: a CREVOKE retires a record and frees it only at a
 * collection, so counting takes alone errs on the side of asking early.
 *
 * Availability is not a thing the application polls. The signals are the hardware's kind-5 pressure
 * escape and the module's collection; a hint, if one is ever wanted, is a service call like the one
 * below and never a CSR. */
#define IDS_ASK (CV_NODE_RESERVE + 256)
/* The most records one heap call takes before its next ensure_ids: a mapping's root and its group
 * lifetime (group_map, see "Recycled group mappings" below), a nested group and the object's own
 * child. A group on a NEW mapping takes one record more than before recycling (root and lifetime,
 * where one mapping group was); a group on a kept mapping takes one (the lifetime). */
#define IDS_PER_CALL 4
static size_t ids_budget;   /* records the heap may still take before asking; under the heap lock */
static int ensure_ids(size_t need)
{
    if (ids_budget >= need) return 0;
    long rc = __capstone_vm_nodes(IDS_ASK);
    if (rc < 0) { errno = -rc; return -1; }
    ids_budget = IDS_ASK - CV_NODE_RESERVE - 1;
    return 0;
}
static inline void take_id(void)
{
    if (ids_budget) --ids_budget;
}
/* Slot-record arrays (lane Y, 2026-11-03). Each mallocng meta points to the slot records of its
 * group, one per slot, instead of embedding 32 of them: 32 x 96 bytes made every meta 3,152 bytes,
 * so a meta-area page held one meta whatever the group's slot count, and SPEC's gcc test faulted
 * 268 meta pages for 268 metas (88 of its 151 live groups have one slot). The records are the same
 * records with the same contents and the same lifetime -- a meta's, from alloc_group to free_meta --
 * only stored elsewhere: arrays of 1, 2, 4, 8, 16 or 32 records, the smallest that holds the group's
 * slot count, cut from chunks this file maps (16 KiB, doubling to 256 KiB) and reused through one
 * free list per size. Nothing but the metas and this file's free lists points to them, as nothing
 * but mallocng's meta areas held them before.
 *
 * The invariant mallocng relies on is unchanged: a group binds records that are all zero. A chunk
 * is fresh zero memory; free_meta zeroes a meta's records before returning them; a free array's
 * first record carries the free-list link in `next`, which this file clears when it hands the
 * array out. A group writes only its own count of records (bind_slots, alloc_group's loop), so
 * the records an array holds past that count stay zero for its next group. Every caller is under
 * the heap lock (mallocng runs only inside malloc, free, realloc and aligned_alloc below). */
#define SLOT_CLASSES 6U   /* 1 << 0 .. 1 << 5 records */
#define SLOT_CHUNK_MIN (16UL << 10)
#define SLOT_CHUNK_MAX (256UL << 10)
static struct capstone_malloc_slot *slots_free_heads[SLOT_CLASSES];
static char *slots_next;
static size_t slots_left, slots_chunk;
static unsigned slots_class(unsigned count)
{
    unsigned c = 0;
    if (!count || count > 1U << (SLOT_CLASSES - 1)) __builtin_trap();
    while ((1U << c) < count) ++c;
    return c;
}
struct capstone_malloc_slot *__capstone_malloc_slots_new(unsigned count)
{
    unsigned c = slots_class(count);
    struct capstone_malloc_slot *s = slots_free_heads[c];
    if (s) {
        slots_free_heads[c] = s->next;
        s->next = NULL;
        return s;
    }
    size_t bytes = ((size_t)1 << c) * sizeof *s;
    if (slots_left < bytes) {
        /* The rest of the old chunk (less than 3 KiB) is left unused. */
        size_t chunk = slots_chunk ? 2 * slots_chunk : SLOT_CHUNK_MIN;
        if (chunk > SLOT_CHUNK_MAX) chunk = SLOT_CHUNK_MAX;
        char *m = mmap(NULL, chunk, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) return NULL;
        slots_chunk = chunk; slots_next = m; slots_left = chunk;
    }
    s = (struct capstone_malloc_slot *)slots_next;
    slots_next += bytes; slots_left -= bytes;
    return s;
}
void __capstone_malloc_slots_free(struct capstone_malloc_slot *s, unsigned count)
{
    unsigned c = slots_class(count);
    s->next = slots_free_heads[c];
    slots_free_heads[c] = s;
}
/* Keep musl's group geometry. Slot parents are narrowed copies of one
 * managed group, not linear partitions or separate lifetime nodes. The
 * caller receives only a freshly derived child over its requested bytes. */
/* The group header bind_slots places slots after. Derived from the MAPPING size, so it is an
 * over-approximation of the header the stride itself needs -- granule() is monotone -- which makes
 * the single slot's stride below provably small enough rather than exactly right. Measured over
 * 300,003 sizes, the over-approximation costs 0 bytes. */
static size_t group_header(size_t bytes)
{
    size_t g = capstone_repr_granule(bytes > 4 ? bytes - 4 : 1);
    return g > 16 ? g : 16;
}
/* Cause B probe. A refused shrink leaves an unusable slot, mallocng then fails to find one, and
 * malloc returns NULL -- indistinguishable at the call site from any other out-of-memory, which is
 * what made the first run uninformative. Counted, never printed: a print from inside the allocator
 * can re-enter it. */
unsigned long __capstone_malloc_bind_refused;
unsigned long __capstone_malloc_bind_first[6];
unsigned long __capstone_malloc_ng_null;
/* Split from __capstone_malloc_ng_null, which conflated them and left the first run ambiguous. */
unsigned long __capstone_malloc_ids_failed;
/* The choke point. map_calls not advancing across a failed malloc means mallocng never asked for
 * a mapping, which no errno can tell you. */
unsigned long __capstone_malloc_map_calls;
unsigned long __capstone_malloc_map_refused_early;
unsigned long __capstone_malloc_map_refused_acquire;
unsigned long __capstone_malloc_map_last[4];   /* bytes, align, errno, rounded request */
/* The BASE, which is what add()'s second disjunct tests: (address & (bytes - 1)). Without it a
 * power-of-two size refused with EINVAL cannot be told from a misaligned one -- the reading that
 * refuted "the predicate is is_power_of_2". */
unsigned long __capstone_malloc_map_last_base;
unsigned long __capstone_malloc_map_ok_base;
/* The last ADMITTED mapping. The exact-bounds-ON arm cannot show a non-power-of-two mapping being
 * accepted while the gate is armed -- turning the profile on disables the gate -- so the only way
 * to see an admission is to record it on the success path. */
unsigned long __capstone_malloc_map_ok[2];     /* bytes, align */
unsigned long __capstone_malloc_map_admitted;
/* `stale`: the slots may hold a previous group's bytes and tags (a reused mapping); each is zeroed
 * at its first take. `clean`: the slots' bytes are all zero and untagged (a fresh mapping, or a
 * nested group inside a slot that was clean when taken); calloc then need not zero an object at a
 * slot's first take. Never both. */
static void *bind_slots(struct capstone_malloc_slot *slots, void *group,
                        size_t stride, unsigned count, unsigned char stale,
                        unsigned char clean)
{
    size_t base = __builtin_capstone_cap_get_cursor(group);
    /* Option (iii): the slot array starts after a GRANULE-sized group header, not a 16-byte one,
     * so each slot's base is a multiple of its own bounds granule and the capability derived over
     * an object in it is exactly representable. mallocng's enframe pads identically
     * (meta.h cap_hdr_pad), and the pad fits in the page rounding alloc_group already does --
     * asserted there, because the margin is exact for some class and count pairs.
     *
     * The granule of the STRIDE, the length of the slot capability shrunk below, not of stride - 4,
     * the largest object. They differ at the power-of-two strides (32768, 65536, 131072), where the
     * old rule left every slot of a multi-slot group half a granule off: the encoder widened the slot
     * downward, the bounds check below counted it without refusing, and backend_pointer() -- which
     * takes the slot's base as its start -- handed mallocng the object's address plus the shift, so
     * free() trapped (gcc train on the twin, a class-43 slot bound as [start - 64, start + 65600)).
     * meta.h cap_hdr_pad (musl patch 0009) takes the same granule. */
    size_t granule = capstone_repr_granule(stride);
    size_t hdr = granule > 16 ? granule : 16;
    for (unsigned i = 0; i < count; ++i) {
        size_t start = base + hdr + i * stride;
        size_t end = base + hdr + (i + 1) * stride;
        /* Offset in integers first (lane runtime/cursor-window, 2026-10-19). `group + start - base`
         * let the compiler keep a capability with its cursor near 0, far outside its bounds, live
         * across this loop. On QEMU's compact profile a preemption saves it compressed and the resume
         * decodes different bounds (capstone_supervisor.c virtual_store_reg/virtual_load_reg); on
         * silicon a bounds-deriving op issued while it is out of window refuses with 29 (RTL arm
         * 4eecdf450). Forming `start - base` first keeps every intermediate cursor in the group. */
        slots[i].parent = __builtin_capstone_cap_shrink(
            (char *)group + (start - base), start, end);
        if (__builtin_capstone_cap_get_base(slots[i].parent) != start ||
            __builtin_capstone_cap_get_end(slots[i].parent) != end) {
            if (!__capstone_malloc_bind_refused++) {
                __capstone_malloc_bind_first[0] = i;
                __capstone_malloc_bind_first[1] = start;
                __capstone_malloc_bind_first[2] = end;
                __capstone_malloc_bind_first[3] = stride;
                __capstone_malloc_bind_first[4] = base;
                __capstone_malloc_bind_first[5] =
                    __builtin_capstone_cap_get_end(group);
            }
        }
#if CAPSTONE_HEAP_OBJECT_BOUNDS
        slots[i].issued = NULL;
#else
        slots[i].issued = group;
#endif
        slots[i].stale = stale;
#if CAPSTONE_HEAP_OBJECT_BOUNDS
        slots[i].clean = clean;
#else
        /* The program holds the group's authority here and can write into a slot nobody took. */
        (void)clean; slots[i].clean = 0;
#endif
    }
    return group;
}
/* A child of a slot's authority over [base + offset, base + offset + size): the CDERIVE
 * child with a lifetime of its own, or without lifetimes the narrowed copy that bind_slots
 * makes for the slot itself (the same rights, the slot's lifetime). */
static void *issue_child(void *parent, size_t offset, size_t size)
{
#if CAPSTONE_HEAP_LIFETIMES
    take_id();
    return capstone_cap_derive(parent, offset, size);
#else
    size_t start = __builtin_capstone_cap_get_base(parent) + offset;
    void *at = (char *)parent + (ptrdiff_t)(start - __builtin_capstone_cap_get_cursor(parent));
    /* CDERIVE refuses bounds the encoder cannot represent (cause 29); SHRINK rounds outward on
     * silicon. A silently widened child would reach into its neighbours where the qualified
     * arm faults, so the spatial arm would under-report the overflows it exists to measure. */
    void *child = __builtin_capstone_cap_shrink(at, start, start + size);
    if (__builtin_capstone_cap_get_base(child) != start ||
        __builtin_capstone_cap_get_end(child) != start + size)
        __builtin_trap();
    return child;
#endif
}
static void end_child(void *parent, void *child)
{
#if CAPSTONE_HEAP_LIFETIMES
    capstone_cap_revoke_child(parent, child);
#else
    (void)parent; (void)child;
#endif
}
static void *mapping_group(sublet_cap *grant, size_t bytes)
{
    take_id();
    return capstone_cap_derive(cap_vm_copyable(grant), 0, bytes);
}
/* The length __capstone_malloc_map maps for a request of `bytes` (at most CV_MAX_REGION_BYTES).
 * Option (iii): above 4 MiB the bounds granule EXCEEDS a page (8 KiB at 4 MiB, 128 KiB at
 * 64 MiB), so a page-aligned mapping is not enough -- the encoder truncates an unaligned base
 * downward and the object would gain authority below itself. Measured: 4, 8, 16, 64 and 88 MiB
 * all decode inexact at a page-aligned base, while 2 MiB and 4 MiB - 4 KiB decode exact.
 * cap_vm_acquire already takes an alignment, so this is the alignment argument rather than an
 * over-map and trim, and `bytes` is rounded to the granule so both edges are multiples. */
static size_t mapping_length(size_t bytes)
{
    return capstone_repr_round((bytes + 4095) & -4096UL);
}

/* Recycled group mappings (decision D10, 2026-11-03). mallocng releases an empty mapped group
 * (free.c okay_to_free, unchanged) and maps a new one when the class needs a group again; each
 * release was an UNMAP service, each new group a MAP service, and every page of the new mapping
 * faulted again (in SPEC's gcc test most of the launcher's returns were these). A released group's
 * MAPPING is kept here instead, registered and resident, and the next group_map of the same length
 * takes it without any service. What makes that sound is that a mapping's authority never enters the
 * program's pointer flow; the program holds only object capabilities, each a descendant of one group
 * LIFETIME (the trust boundary below says what that does and does not cover):
 *
 *   mapping grant (kernel)
 *     +-- root: CDERIVE child, kept for the mapping's life in THIS FILE'S static state only
 *           +-- group lifetime: CDERIVE child of the root, one per group (bind_slots' parent)
 *                 +-- slot parents (narrowed copies) -> objects, nested groups (CDERIVE)
 *
 * TRUST BOUNDARY (decision 2026-11-03, after review). What the heap's temporal guarantee covers is
 * the program's own pointer flow: a pointer the program obtained from malloc, calloc, realloc,
 * aligned_alloc or the nested-allocator trio, and kept past the free (or past its group's release),
 * cannot reach a later object -- including after the mapping it pointed into has been reused for a
 * new group. What it does NOT cover is code that reads allocator state on purpose: code that follows
 * the image-wide gp capability into libc's statics, or calls the SDK-internal entry points
 * (__capstone_malloc_find and the other __capstone_malloc_* functions musl and some ports call).
 * Such code reaches slot records, metas and whole-group capabilities, and through this file's
 * statics the mapping roots too. That boundary is pre-existing -- on dev the same code already
 * reaches mallocng's metas and their whole-group capabilities -- and recycling leaves it where it
 * was, with one difference: a root now outlives the groups derived from it, so such code could
 * derive a sibling of a group that survives the group's release. The root is therefore kept out of
 * every record and return value (root_heads/recycle below; malloc-contract.c recycle-root), so no
 * entry point hands it out; it is still inside gp's bounds. Closing the boundary means moving
 * allocator state out of what gp covers, a separate item.
 *
 * (1) Nothing derived from the old group survives: releasing the group CREVOKEs its lifetime
 *     from the root, which ends every descendant at once -- objects, nested groups and the
 *     nested-allocator trio's children -- whether or not each was already revoked by free
 *     (prepare_free), an in-place or moving realloc (publish, mallocng's free) or the nested
 *     group's release. Without per-object lifetimes it is the one point where stale objects
 *     end, exactly as the unmap's retirement was.
 * (2) Nothing of the old group's metadata is reused: the new group has a fresh lifetime (a new
 *     CDERIVE of the root), bind_slots re-derives every slot parent from it, and the meta and
 *     its slot records are fresh (mallocng's free_meta zeroes them). The root is the one
 *     capability kept across groups, and only the allocator holds it.
 * (3) The bytes a program can reach are clean: every slot of a group placed in a recycled
 *     mapping is marked stale, and its first take (enframe, before any child over it is
 *     derived) zeroes the slot, which clears every tag in it; a plain store clears its
 *     granule's tag. Objects and nested groups lie inside slots, so no issued capability covers
 *     an unscrubbed granule. The header and the slack after the last slot are read only as
 *     scalars by the allocator. Scrubbing at first take rather than at reuse touches only the
 *     slots that are used again, so no page the new group does not use is faulted in.
 *     CAPSTONE_HEAP_OBJECT_BOUNDS=0 hands the program the GROUP's authority, which also reaches
 *     the header, the untaken slots and the slack, so that arm zeroes the whole mapping when it
 *     is reused, before the new group is bound.
 * (4) Bounded: RECYCLE_PER_LENGTH mappings of one length, RECYCLE_ENTRIES in all, at most
 *     RECYCLE_BYTES including empty shared regions, none above RECYCLE_MAX_ENTRY; a release beyond a bound is unmapped as
 *     before; malloc_trim unmaps every kept mapping. Kept mappings stay registered arenas, so
 *     the module's collection sweeps their
 *     pages like any live heap, and at exit its teardown retires them (vm_release -> retire),
 *     the same path an unmap takes. Large (sc 63) and remapped mappings have no root entry and
 *     are always unmapped, as is a group whose root entry could not be allocated. All of this runs under the heap lock (malloc, free and realloc hold it
 *     around mallocng). */
#define RECYCLE_PER_LENGTH 4U
#define RECYCLE_ENTRIES 32U
#define RECYCLE_BYTES (1UL << 20)
#define RECYCLE_MAX_ENTRY (256UL << 10)
static struct { void *root; size_t bytes; } recycle[RECYCLE_ENTRIES];
static unsigned recycle_count;
static size_t recycle_bytes;
static size_t coarse_empty_bytes;
/* The roots of live mapped groups, by mapping base: entered by group_map, taken out by
 * group_unmap. Entries come from a static seed and then from pages of their own (mmap), which
 * only this table points to; a mapping whose entry cannot be had is simply not recyclable. */
struct root_entry { struct root_entry *next; void *root; size_t base; };
#define ROOT_BUCKETS 64U
static struct root_entry *root_heads[ROOT_BUCKETS];
static struct root_entry *root_free;
static struct root_entry root_seed[32];
static unsigned root_seeded;
static unsigned root_bucket(size_t base)
{ return (unsigned)(((base >> 12) * 11400714819323198485UL) >> 58); }
static void root_free_push(struct root_entry *e)
{ e->root = NULL; e->base = 0; e->next = root_free; root_free = e; }
static struct root_entry *root_entry_new(void)
{
    if (!root_seeded) {
        root_seeded = 1;
        for (unsigned i = 0; i < sizeof root_seed / sizeof *root_seed; ++i) root_free_push(&root_seed[i]);
    }
    if (!root_free) {
        int e = errno;
        char *page = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        errno = e;
        if (page == MAP_FAILED) return NULL;
        for (size_t o = 0; o + sizeof(struct root_entry) <= 4096; o += sizeof(struct root_entry))
            root_free_push((struct root_entry *)(page + o));
    }
    struct root_entry *r = root_free;
    root_free = r->next;
    return r;
}
static void root_enter(void *root, size_t base)
{
    struct root_entry *e = root_entry_new();
    if (!e) return;
    e->root = root; e->base = base;
    unsigned b = root_bucket(base);
    e->next = root_heads[b]; root_heads[b] = e;
}
static void *root_remove(size_t base)
{
    for (struct root_entry **p = &root_heads[root_bucket(base)]; *p; p = &(*p)->next)
        if ((*p)->base == base) {
            struct root_entry *e = *p;
            void *root = e->root;
            *p = e->next;
            root_free_push(e);
            return root;
        }
    return NULL;
}
/* Read by malloc-contract.c's recycle cases: groups placed in a kept mapping, releases kept,
 * releases unmapped because a bound was full or the mapping too large. */
unsigned long __capstone_malloc_recycle_hits;
unsigned long __capstone_malloc_recycle_kept;
unsigned long __capstone_malloc_recycle_unmapped;
/* CAPSTONE_HEAP_RECYCLE_TRACE=1: one stderr line per kept mapping taken (hit) or released into
 * the cache (kept), so a run can say whether it went through a reused mapping even when it ends in
 * a fault, before any exit-time report. write(2) of a stack buffer, never stdio, so the line cannot
 * re-enter the allocator. Unset, it costs one load per event. */
static int recycle_trace = -1;
static void recycle_note(const char *what, size_t bytes)
{
    if (recycle_trace < 0) {
        const char *e = getenv("CAPSTONE_HEAP_RECYCLE_TRACE");
        recycle_trace = e && e[0] == '1';
    }
    if (!recycle_trace) return;
    char line[128];
    int n = snprintf(line, sizeof line, "CAPSTONE_HEAP_RECYCLE %s bytes=%lu hits=%lu kept=%lu unmapped=%lu\n",
                     what, (unsigned long)bytes, __capstone_malloc_recycle_hits,
                     __capstone_malloc_recycle_kept, __capstone_malloc_recycle_unmapped);
    if (n > 0) write(2, line, (size_t)n < sizeof line ? (size_t)n : sizeof line - 1);
}
static void *recycle_take(size_t bytes)
{
    for (unsigned i = recycle_count; i--; )
        if (recycle[i].bytes == bytes) {
            void *root = recycle[i].root;
            recycle_bytes -= bytes;
            recycle[i] = recycle[--recycle_count];
            recycle[recycle_count].root = NULL;
            recycle[recycle_count].bytes = 0;
            return root;
        }
    return NULL;
}
static int recycle_fits(size_t bytes)
{
    if (bytes > RECYCLE_MAX_ENTRY || recycle_count >= RECYCLE_ENTRIES ||
        recycle_bytes + coarse_empty_bytes + bytes > RECYCLE_BYTES) return 0;
    unsigned same = 0;
    for (unsigned i = 0; i < recycle_count; ++i) same += recycle[i].bytes == bytes;
    return same < RECYCLE_PER_LENGTH;
}
/* A group's lifetime: a fresh CDERIVE child of the mapping's root, over the whole mapping. */
static void *group_lifetime(void *root, size_t bytes)
{
    take_id();
    return capstone_cap_derive(root, 0, bytes);
}
/* Small multi-slot groups share 64 KiB regions. A region contains equal-sized extents, each
 * large enough for the group's header and slots, rounded to a 16-byte granule and then to
 * representable bounds. mallocng still chooses every slot/count/stride; only backing placement changes.
 * Single/two-slot groups and mappings above 16 KiB keep the existing recycling path. In
 * particular, every group that can reach mallocng's mremap path remains a separate mapping.
 *
 * The root and occupancy maps are private, just like root_heads. Each occupied extent has its
 * own CDERIVE child, which is all bind_slots sees. A release CREVOKEs that child before clearing
 * its occupancy bit. `dirty` survives release: reusing that extent must scrub its slots (or the
 * whole group with object bounds off). Equal extents may hold different group geometries.
 * Slot records remain lane Y's zeroed arrays, and program pointers still use the address index.
 *
 * Region bases are 64 KiB aligned, so a group's address finds its region through a hash table,
 * without walking all live regions. A second table contains only regions with free extents.
 * Empty regions share RECYCLE_BYTES with the dedicated-mapping cache; malloc_trim returns all
 * of them. munmap performs the VM retirement before Linux can reuse any region page. No partial
 * region is unmapped or advised away. All operations hold heap_mutex. */
#define COARSE_REGION_BYTES (64UL << 10)
#define COARSE_GROUP_MAX (16UL << 10)
#define COARSE_BUCKETS 64U
struct coarse_region {
    struct coarse_region *hash_next, *avail_next, *avail_prev;
    void *root;
    size_t base, span;
    unsigned long free, dirty, mask;
};
static struct coarse_region *coarse_heads[COARSE_BUCKETS];
static struct coarse_region *coarse_avail[COARSE_BUCKETS];
static struct coarse_region *coarse_free;
static struct coarse_region coarse_seed[8];
static unsigned coarse_seeded;
unsigned long __capstone_malloc_coarse_regions;
unsigned long __capstone_malloc_coarse_groups;
unsigned long __capstone_malloc_coarse_reused;
unsigned long __capstone_malloc_coarse_released;
unsigned long __capstone_malloc_coarse_unmapped;
static void *malloc_map_aligned(size_t bytes, size_t align);
static unsigned coarse_bucket(size_t key)
{ return (unsigned)((key * 11400714819323198485UL) >> 58); }
static void coarse_avail_add(struct coarse_region *r)
{
    unsigned b = coarse_bucket(r->span >> 4);
    r->avail_prev = NULL; r->avail_next = coarse_avail[b];
    if (r->avail_next) r->avail_next->avail_prev = r;
    coarse_avail[b] = r;
}
static void coarse_avail_remove(struct coarse_region *r)
{
    if (r->avail_prev) r->avail_prev->avail_next = r->avail_next;
    else coarse_avail[coarse_bucket(r->span >> 4)] = r->avail_next;
    if (r->avail_next) r->avail_next->avail_prev = r->avail_prev;
    r->avail_next = r->avail_prev = NULL;
}
static struct coarse_region *coarse_entry_new(void)
{
    if (!coarse_seeded) {
        coarse_seeded = 1;
        for (unsigned i = 0; i < sizeof coarse_seed / sizeof *coarse_seed; ++i) {
            coarse_seed[i].hash_next = coarse_free; coarse_free = &coarse_seed[i];
        }
    }
    if (!coarse_free) {
        int e = errno;
        char *page = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        errno = e;
        if (page == MAP_FAILED) return NULL;
        for (size_t o = 0; o + sizeof(struct coarse_region) <= 4096; o += sizeof(struct coarse_region)) {
            struct coarse_region *r = (struct coarse_region *)(page + o);
            r->hash_next = coarse_free; coarse_free = r;
        }
    }
    struct coarse_region *r = coarse_free;
    coarse_free = r->hash_next;
    memset(r, 0, sizeof *r);
    return r;
}
static void coarse_entry_free(struct coarse_region *r)
{
    memset(r, 0, sizeof *r);
    r->hash_next = coarse_free; coarse_free = r;
}
/* Failure keeps the empty region registered and available, so trim can retry. */
static int coarse_drop(struct coarse_region *r)
{
    if (r->free != r->mask) __builtin_trap();
    if (munmap(r->root, COARSE_REGION_BYTES)) return 0;
    unsigned b = coarse_bucket(r->base >> 16);
    struct coarse_region **p = &coarse_heads[b];
    while (*p != r) {
        if (!*p) __builtin_trap();
        p = &(*p)->hash_next;
    }
    *p = r->hash_next;
    coarse_avail_remove(r);
    coarse_empty_bytes -= COARSE_REGION_BYTES;
    coarse_entry_free(r);
    ++__capstone_malloc_coarse_unmapped;
    return 1;
}
static void *coarse_take(size_t bytes, size_t stride, unsigned count, unsigned char *stale)
{
    if (count < 3 || bytes > COARSE_GROUP_MAX) return NULL;
    size_t hdr = capstone_repr_granule(stride);
    if (hdr < 16) hdr = 16;
    /* Check before multiplication, even though mallocng supplies bounded counts and strides. */
    if (bytes < hdr || !stride || count > (bytes - hdr) / stride) return NULL;
    size_t span = capstone_repr_round((hdr + stride * count + 15) & -16UL);
    if (span < 1024 || span > bytes) return NULL;
#if CAPSTONE_HEAP_COARSE_PAGES
    /* Merge nearby geometries into one region bin and keep extents on page
     * boundaries. Each group still owns only its derived extent. A reused
     * extent keeps its dirty bit even when the new stride/count differ. */
    span = (span + 4095UL) & -4096UL;
    if (span > bytes) return NULL;
#endif
    unsigned b = coarse_bucket(span >> 4);
    struct coarse_region *r;
    for (r = coarse_avail[b]; r && r->span != span; r = r->avail_next) {}
    if (!r) {
        r = coarse_entry_new();
        if (!r) return NULL;
        if (malloc_map_aligned(COARSE_REGION_BYTES, COARSE_REGION_BYTES) == MAP_FAILED) {
            coarse_entry_free(r); return NULL;
        }
        r->root = mapping_group(&pending_mapping, COARSE_REGION_BYTES);
        r->base = __builtin_capstone_cap_get_cursor(r->root);
        r->span = span;
        unsigned capacity = COARSE_REGION_BYTES / span;
        r->mask = capacity == 64 ? ~0UL : (1UL << capacity) - 1;
        r->free = r->mask;
        unsigned h = coarse_bucket(r->base >> 16);
        r->hash_next = coarse_heads[h]; coarse_heads[h] = r;
        coarse_avail_add(r);
        coarse_empty_bytes += COARSE_REGION_BYTES;
        ++__capstone_malloc_coarse_regions;
    }
    if (r->free == r->mask) coarse_empty_bytes -= COARSE_REGION_BYTES;
    unsigned slot = __builtin_ctzl(r->free);
    unsigned long bit = 1UL << slot;
    *stale = (r->dirty & bit) != 0;
    r->free &= ~bit; r->dirty |= bit;
    if (!r->free) coarse_avail_remove(r);
    ++__capstone_malloc_coarse_groups;
    __capstone_malloc_coarse_reused += *stale;
    take_id();
    return capstone_cap_derive(r->root, slot * span, span);
}
static int coarse_release(void *group, size_t bytes)
{
    size_t address = __builtin_capstone_cap_get_cursor(group);
    size_t base = address & -COARSE_REGION_BYTES;
    struct coarse_region *r = coarse_heads[coarse_bucket(base >> 16)];
    for (; r && r->base != base; r = r->hash_next) {}
    if (!r) return 0;
    size_t offset = address - base;
    unsigned slot = offset / r->span;
    if (offset % r->span || slot >= 64 || !(r->mask & (1UL << slot)) ||
        (r->free & (1UL << slot)) || bytes != mapping_length(r->span) ||
        __builtin_capstone_cap_get_base(group) != address ||
        __builtin_capstone_cap_get_end(group) - address != r->span) __builtin_trap();
    capstone_cap_revoke_child(r->root, group);
    if (!r->free) coarse_avail_add(r);
    r->free |= 1UL << slot;
    ++__capstone_malloc_coarse_released;
    if (r->free == r->mask) {
        coarse_empty_bytes += COARSE_REGION_BYTES;
        if (coarse_empty_bytes + recycle_bytes > RECYCLE_BYTES) coarse_drop(r);
    }
    return 1;
}
void __capstone_malloc_group_unmap(void *group, size_t bytes)
{
    if (coarse_release(group, bytes)) return;
    void *root = root_remove(__builtin_capstone_cap_get_cursor(group));
    if (root && recycle_fits(bytes)) {
        capstone_cap_revoke_child(root, group);
        recycle[recycle_count].root = root;
        recycle[recycle_count].bytes = bytes;
        ++recycle_count; recycle_bytes += bytes;
        ++__capstone_malloc_recycle_kept;
        recycle_note("kept", bytes);
        return;
    }
    if (root) ++__capstone_malloc_recycle_unmapped;
    /* The unmap retires the mapping's grant, and with it the root and every descendant. */
    munmap(group, bytes);
}
static void *malloc_map_aligned(size_t bytes, size_t align)
{
    __capstone_malloc_map_calls++;
    __capstone_malloc_map_last[3] = bytes;
    if (!bytes || bytes > CV_MAX_REGION_BYTES) {
        __capstone_malloc_map_refused_early++;
        errno = ENOMEM; return MAP_FAILED;
    }
    bytes = mapping_length(bytes);
    size_t granule = capstone_repr_granule(bytes);
    if (align < granule) align = granule;
    if (align < 4096UL) align = 4096UL;
    if (cap_vm_acquire(&pending_mapping, bytes, align,
                       PROT_READ | PROT_WRITE, 6, CAP_VM_HEAP)) {
        __capstone_malloc_map_refused_acquire++;
        __capstone_malloc_map_last[0] = bytes;
        __capstone_malloc_map_last[1] = align;
        __capstone_malloc_map_last[2] = (unsigned long)errno;
        return MAP_FAILED;
    }
    __capstone_malloc_map_admitted++;
    __capstone_malloc_map_ok[0] = bytes;
    __capstone_malloc_map_ok[1] = align;
    __capstone_malloc_map_ok_base = sublet_base(&pending_mapping);
    /* Does the ADMITTED mapping satisfy add()'s second disjunct? If a mapping is admitted whose
     * base is NOT naturally aligned to its own size, the alignment half cannot be what refuses. */
    __capstone_malloc_map_last_base =
        (__capstone_malloc_map_ok_base & (bytes - 1)) ? 1UL : 0UL;
    /* A scalar placeholder crosses only the private backend, until attach.
     * The linear grant remains in its consuming slot throughout alloc_meta. */
    return (void *)sublet_base(&pending_mapping);
}
void *__capstone_malloc_map(size_t bytes)
{ return malloc_map_aligned(bytes, 4096UL); }
void *__capstone_malloc_group_attach(struct capstone_malloc_slot *slots,
                                    void *p, size_t bytes)
{
    size_t base = sublet_base(&pending_mapping);
    if (base != (size_t)p) __builtin_trap();
    /* stride = bytes - header, NOT bytes - 16. The single slot starts after the granule-sized
     * header, so a stride measured from a 16-byte one ran past the end of the mapping and the
     * shrink in bind_slots could not be derived -- which is what returned NULL for every
     * allocation on this path. */
    return bind_slots(slots, mapping_group(&pending_mapping, bytes),
                      bytes - group_header(bytes), 1, 0, 1);
}
void *__capstone_malloc_group_map(struct capstone_malloc_slot *slots,
                                 size_t bytes, size_t stride, unsigned count)
{
    unsigned char reused = 0;
    void *small = coarse_take(bytes, stride, count, &reused);
    if (small) {
#if !CAPSTONE_HEAP_OBJECT_BOUNDS
        if (reused) {
            memset(small, 0, __builtin_capstone_cap_get_end(small) -
                             __builtin_capstone_cap_get_base(small));
            reused = 0;
        }
#endif
        return bind_slots(slots, small, stride, count, reused, !reused);
    }
    /* A kept mapping of the length __capstone_malloc_map would map, or a new mapping. */
    size_t mapped = bytes && bytes <= CV_MAX_REGION_BYTES ? mapping_length(bytes) : 0;
    void *r = mapped ? recycle_take(mapped) : NULL;
    unsigned char stale = r != NULL;   /* a fresh mapping is zero and untagged: clean */
    if (r) { ++__capstone_malloc_recycle_hits; recycle_note("hit", mapped); }
    else {
        void *p = __capstone_malloc_map(bytes);
        if (p == MAP_FAILED) return p;
        mapped = sublet_end(&pending_mapping) - sublet_base(&pending_mapping);
        r = mapping_group(&pending_mapping, mapped);
    }
    root_enter(r, __builtin_capstone_cap_get_cursor(r));
    void *group = group_lifetime(r, mapped);
#if !CAPSTONE_HEAP_OBJECT_BOUNDS
    /* The program holds this whole group's authority in this arm (condition 3 above). */
    if (stale) { memset(group, 0, mapped); stale = 0; }
#endif
    return bind_slots(slots, group, stride, count, stale, !stale);
}
void *__capstone_malloc_group_nested(struct capstone_malloc_slot *slots,
                                    size_t stride, unsigned count,
                                    struct capstone_malloc_slot *parent)
{
    size_t base = parent->key;
#if CAPSTONE_HEAP_OBJECT_BOUNDS
    parent->issued = issue_child(parent->parent,
        base - __builtin_capstone_cap_get_base(parent->parent), 16 + stride * count);
    parent->kind = SLOT_NESTED;
    /* The host slot was taken just before (alloc_group's enframe); if it was clean then, so is
     * the nested group's storage, which mallocng has written only with its header and zeros. */
    return bind_slots(slots, parent->issued, stride, count, 0, parent->taken_clean);
#else
    /* The host slot keeps its group authority in `issued`; the nested group is a fresh shrink. */
    void *nested = issue_child(parent->parent,
        base - __builtin_capstone_cap_get_base(parent->parent), 16 + stride * count);
    parent->kind = SLOT_NESTED;
    return bind_slots(slots, nested, stride, count, 0, 0);
#endif
}
void *__capstone_malloc_slot_take(struct capstone_malloc_slot *r)
{
    void *p = r->parent;
    r->taken_clean = r->clean | r->stale;
    r->clean = 0;
    /* First take of a slot in a recycled mapping: clear the previous group's bytes and tags
     * before any capability over them is issued (condition 3 at group_unmap). */
    if (r->stale) {
        memset(p, 0, __builtin_capstone_cap_get_end(p) - __builtin_capstone_cap_get_base(p));
        r->stale = 0;
    }
    /* Not indexed here: enframe keys the record once, at the object's final start, and passes
     * the record to every prefix access in between (cap_prefix(p, s)). */
    return p;
}
void *__capstone_malloc_slot_start(struct capstone_malloc_slot *r)
{
    return r->parent;
}
static void *backend_pointer(struct capstone_malloc_slot *r)
{ return (char *)r->parent + (r->key - __builtin_capstone_cap_get_base(r->parent)); }  /* offset first; see bind_slots */
void __capstone_malloc_slot_release(struct capstone_malloc_slot *r)
{ unindex(r); r->kind = SLOT_INTERNAL; }
void __capstone_malloc_nested_release(struct capstone_malloc_slot *r)
{
#if CAPSTONE_HEAP_OBJECT_BOUNDS
    end_child(r->parent, r->issued);
    r->issued = NULL;
#endif
    __capstone_malloc_slot_release(r);
}
/* End the public lifetime of the object in record r before mallocng publishes its slot as free.
 * r is the record of mallocng's backend pointer; returned for mallocng's free, which takes it. */
static struct capstone_malloc_slot *prepare_slot(struct capstone_malloc_slot *r)
{
    if (!r || r->kind == SLOT_NESTED) __builtin_trap();
    if (r->kind == SLOT_PUBLIC) {
#if CAPSTONE_HEAP_SCRUB_FREE
        /* Remove residual payload capabilities before returning ownership.
         * The slot prefix and musl's offset-cycle state are preserved. */
        memset(backend_pointer(r), 0, r->nominal);
#endif
#if CAPSTONE_HEAP_OBJECT_BOUNDS
        end_child(r->parent, r->issued);
        r->issued = NULL;
#endif
        r->kind = SLOT_INTERNAL;
    }
    return r;
}
/* mallocng's free(p) when it frees an object of its own (realloc's moving path, exact_or_moved):
 * one lookup. The program's free reaches mallocng with the record free() found (free below). */
struct capstone_malloc_slot *__capstone_malloc_prepare_free(void *p)
{
    return prepare_slot(__capstone_malloc_find(p));
}
/* A real Linux mremap is a VM slow path. The returned grant is new authority
 * over the retained physical frames; the old mapping's descendants are dead. */
/* Option (iii): how often a remap had to fall back to a copy because the service returned a
 * mapping that was page-aligned but not granule-aligned. __capstone_vm_remap takes an address and
 * two lengths and has NO alignment parameter, unlike cap_vm_acquire, so this cannot be asked for.
 * Read by malloc-contract.c and reported from the suite runs: mremap returns page-aligned
 * addresses, so at a 128 KiB granule most huge reallocs are expected to take the copy, and that
 * cost should be a measured number rather than an inference. If the acceptance workloads hit it
 * often, an aligned remap (reserve an aligned range, then MREMAP_FIXED into it, inside the module)
 * is the follow-up, decided on this count. */
unsigned long __capstone_malloc_remap_unaligned;

/* Is a remapped mapping usable, i.e. will every bound derived inside it be representable? */
int __capstone_malloc_remap_usable(unsigned long base, size_t bytes)
{
    /* Check the length ACTUALLY mapped, not a rounded-up one: repr_round(bytes) can be
     * representable while `bytes` itself is not, which would pass a mapping whose real end the
     * encoder cannot express. */
    return bytes && capstone_repr_ok(base, bytes);
}

void *__capstone_malloc_group_remap(struct capstone_malloc_slot *slots,
                                   void *p, size_t old, size_t bytes)
{
    struct capstone_malloc_slot *r = slots;
    size_t offset = r->key - __builtin_capstone_cap_get_base(r->parent);
    sublet_cap root;
    sublet_store(&root, __capstone_vm_remap((size_t)p, old, bytes));
    unsigned long result;
    __asm__ volatile("ld %0, 0(%1)" : "=r"(result) : "r"(&root) : "memory");
    if ((long)result < 0) { errno = -(long)result; sublet_clear(&root); return MAP_FAILED; }
    /* CORRECTION (2026-10-17): this was written as a safe fallback -- "costs a copy and never a
     * wrong bound" -- and it is not one. __capstone_vm_remap() above has ALREADY moved the
     * mapping and re-registered it by the time we get here, so returning MAP_FAILED sends
     * realloc.c down a malloc-plus-copy path that reads from an address which no longer maps.
     * The launcher's remap_mapping() now guarantees a granule-aligned destination, which makes
     * this an ASSERTION rather than a fallback: a nonzero counter is a launcher bug to fix, not
     * a cost to measure. It is kept, and kept counted, precisely so that bug would be visible
     * rather than silent. */
    if (!__capstone_malloc_remap_usable(sublet_base(&root), bytes)) {
        __capstone_malloc_remap_unaligned++;
        sublet_clear(&root);
        errno = ENOMEM;
        return MAP_FAILED;
    }
    unindex(r);
    /* bytes - group_header(bytes), NOT bytes - 16, for the same reason as group_attach: the slot
     * begins after a granule-sized header, so a stride measured from a 16-byte one runs past the
     * end of the mapping. Missing it here while fixing the attach path cost a SIGSEGV that only
     * appeared once growth began completing at all -- the overrun is 496 bytes on a 401408-byte
     * mapping, and get_stride() (which realloc.c now uses) disagrees with it. */
    void *header = bind_slots(slots, mapping_group(&root, bytes),
                              bytes - group_header(bytes), 1, 0, 0);
    void *raw = (char *)r->parent + offset;
    __capstone_malloc_slot_key(r, raw, r->nominal);
    *(unsigned long *)header = (unsigned long)r->group;
    ((unsigned char *)header)[8] = 0;
    return header;
}
/* capstone_repr_round of a request, with the common case first: below 4 KiB the granule is 1 and a
 * length is its own rounding. capstone_repr_round computes the msb before it can tell, and without
 * Zbb that is ~40 instructions whose constants the compiler materialises on every call. */
static inline size_t request_round(size_t n)
{
    return n < 4096 ? (n ? n : 1) : capstone_repr_round(n);
}
/* *clean (if given): whether the object's bytes are all zero and untagged, because its slot was
 * clean or scrubbed at this take (calloc then skips its memset). Counted here, under the heap lock,
 * in __capstone_malloc_calloc_clean (malloc-contract.c calloc-clean reads it). */
unsigned long __capstone_malloc_calloc_clean;
static void *publish(void *p, size_t n, int *clean)
{
    if (!p) return NULL;
    struct capstone_malloc_slot *r = record_of_result(p);
    if (!r) __builtin_trap();
    if (clean) { *clean = r == keyed && r->taken_clean; __capstone_malloc_calloc_clean += *clean; }
    /* Successful in-place realloc replaces the public lifetime; failure
     * never reaches here. Remapping already retired the old mapping. */
#if CAPSTONE_HEAP_OBJECT_BOUNDS
    if (r->issued) end_child(r->parent, r->issued);
    r->kind = SLOT_PUBLIC; r->nominal = n;
    r->issued = issue_child(r->parent,
        r->key - __builtin_capstone_cap_get_base(r->parent), n ? n : 1);
    return r->issued;
#else
    /* The group's authority, cursor at the object (offset in integers first; see bind_slots). */
    r->kind = SLOT_PUBLIC; r->nominal = n;
    return (char *)r->issued + (r->key - __builtin_capstone_cap_get_base(r->issued));
#endif
}
static void *allocate(size_t n, int *clean)
{
    /* Option (iii): publish() derives the child over the nominal size, so that size has to be
     * representable -- a granule multiple, at a granule-aligned base. Rounding HERE rather than
     * inside mallocng keeps musl's own decision code untouched (check-mallocng-policy.py holds
     * size_to_class and the slot/class selection token-identical to upstream), and it means
     * mallocng's set_size records exactly what the capability covers, so malloc_usable_size and
     * an in-place realloc never report bytes the program cannot reach. The seven raised stride
     * entries are what make the resulting slot's base granule-aligned. */
    size_t want = request_round(n);
    if (want < n) { errno = ENOMEM; return NULL; }   /* overflow on a hostile size */
    lock_heap();
    void *raw;
    if (ensure_ids(IDS_PER_CALL)) { __capstone_malloc_ids_failed++; raw = NULL; }
    else { raw = __capstone_mallocng_malloc(want); if (!raw) __capstone_malloc_ng_null++; }
    void *p = publish(raw, want, clean);
    unlock_heap(); return p;
}
void *malloc(size_t n) { return allocate(n, NULL); }
void *__libc_malloc(size_t n) { return malloc(n); }
void *__simple_malloc(size_t n) { return malloc(n); }
__attribute__((noinline, noreturn)) static void invalid_free(void)
{
    __asm__ volatile(".global cap_malloc_invalid_free\ncap_malloc_invalid_free:\nunimp" ::: "memory");
    __builtin_unreachable();
}
__attribute__((noinline)) static struct capstone_malloc_slot *checked(void *p)
{
    __asm__ volatile(".global cap_malloc_validate\ncap_malloc_validate:\nlbu zero, 0(%0)"
                     : : "r"(p) : "memory");
    struct capstone_malloc_slot *r = __capstone_malloc_find(p);
    if (!r || r->kind != SLOT_PUBLIC) invalid_free();
    return r;
}
void free(void *p)
{
    if (!p) return;
    int e = errno;
    lock_heap();
    struct capstone_malloc_slot *r = prepare_slot(checked(p));
    __capstone_mallocng_free_slot(backend_pointer(r), r);
    unlock_heap();
    errno = e;
}
void __libc_free(void *p) { free(p); }

/* THE NESTED-ALLOCATOR TRIO: what an allocator layered on malloc needs at its own alloc and free,
 * and nothing else. An application allocator that hands out sub-objects of a malloc'd block (PHP's
 * emalloc: a 48-byte header, then the payload) must at ITS free find the block's own authority
 * from the caller's pointer and keep a live reference to the block after the caller's lifetime
 * has ended. Both are in this heap's slot record already -- the issued object with MANAGE, indexed
 * by address -- so every such allocator was rebuilding this index for itself (ports/php's seam
 * carried 886 lines of it). These calls are the index, exported.
 *
 * derive: a child of the issued object `object`, over [object + offset, + size), on this heap's
 * node budget (the same ensure_ids/take_id path malloc's own child takes, so a nested allocator
 * cannot exhaust the table the heap reserved against). NULL when the table cannot be grown, never
 * a fault: the caller decides what an unprotected object means in its arm.
 *
 * object: the issued object that holds `child`, cursor at its start, or NULL when no slot record
 * has that block. Nothing ends. The block is found by the child's BASE minus `offset`, the same
 * offset derive was given, so a child narrowed from below (its base moved) is not found.
 *
 * reclaim: object, and the caller's child ends. A missing block is refused like an interior free
 * (cap_malloc_invalid_free), after one diagnostic line. CREVOKE's direct-parent rule is the
 * identity check: a grandchild the client derived for itself is cause 29, a child already ended is
 * cause 25, both at cap_malloc_reclaim. The parent is the record's issued object, never the
 * caller's reference, so the authority that ends the child is the heap's. Lookup, validation and
 * the CREVOKE run under the heap lock, so a concurrent free or realloc of the same object cannot
 * end the parent between the two. */
#if !CAPSTONE_HEAP_OBJECT_BOUNDS
/* The trio's contract is that a slot's `issued` IS the block, cursor at its start. Under
 * OBJECT_BOUNDS=0 `issued` holds the whole group's authority (bind_slots), so object would hand a
 * nested allocator the mapping and derive would count offsets from the group's base. No Caps
 * measurement runs a nested allocator; the arm refuses the trio rather than answer wrongly. */
#define CAPS_REFUSES_TRIO() __builtin_trap()
#else
#define CAPS_REFUSES_TRIO() ((void)0)
#endif
static struct capstone_malloc_slot *slot_of_child(void *child, size_t offset)
{
    size_t key = __builtin_capstone_cap_get_base(child) - offset;
    for (struct capstone_malloc_slot *q = index_heads[bucket(key)]; q; q = q->next)
        if (q->key == key) return (q->kind == SLOT_PUBLIC && q->issued) ? q : NULL;
    return NULL;
}
void *__capstone_malloc_derive(void *object, size_t offset, size_t size)
{
    CAPS_REFUSES_TRIO();
    if (!object || !size) return NULL;
    lock_heap();
    void *child = NULL;
    if (ensure_ids(1) == 0) { take_id(); child = capstone_cap_derive(object, offset, size); }
    unlock_heap();
    return child;
}
void *__capstone_malloc_object(void *child, size_t offset)
{
    CAPS_REFUSES_TRIO();
    if (!child) return NULL;
    lock_heap();
    struct capstone_malloc_slot *r = slot_of_child(child, offset);
    void *object = r ? r->issued : NULL;
    unlock_heap();
    return object;
}
__attribute__((noinline)) static void reclaim_child(void *parent, void *child)
{
    __asm__ volatile(".global cap_malloc_reclaim\ncap_malloc_reclaim:" ::: "memory");
    capstone_cap_revoke_child(parent, child);
}
void *__capstone_malloc_reclaim(void *child, size_t offset)
{
    CAPS_REFUSES_TRIO();
    if (!child) return NULL;
    lock_heap();
    struct capstone_malloc_slot *r = slot_of_child(child, offset);
    void *object = r ? r->issued : NULL;
    if (object) reclaim_child(object, child);
    unlock_heap();
    if (!object) {
        /* One line before the refusal, so a run that dies here says which pointer: a free of a
         * pointer this heap did not issue reads otherwise exactly like a port defect. */
        char line[160];
        int n = snprintf(line, sizeof line,
                         "CAPSTONE_RECLAIM_MISS base=0x%lx cursor=0x%lx end=0x%lx offset=%lu tag=%d\n",
                         (unsigned long)__builtin_capstone_cap_get_base(child),
                         (unsigned long)__builtin_capstone_cap_get_cursor(child),
                         (unsigned long)__builtin_capstone_cap_get_end(child), (unsigned long)offset,
                         (int)__builtin_capstone_cap_get_tag(child));
        if (n > 0) write(2, line, (size_t)n < sizeof line ? (size_t)n : sizeof line - 1);
        invalid_free();
    }
    return object;
}
/* An object in a slot that was clean at this take (a fresh mapping's slot nobody took before, a
 * nested group's in such a slot, or a reused mapping's slot just scrubbed) is already zero and
 * untagged, as upstream's calloc skips the memset for fresh mmapped memory (__malloc_allzerop):
 * nothing but its first owner can have written it, since every capability the program holds
 * covers only the slot it was issued from. */
void *calloc(size_t n, size_t size)
{
    if (size && n > SIZE_MAX / size) { errno = ENOMEM; return NULL; }
    int clean = 0;
    void *p = allocate(n * size, &clean);
    if (p && !clean) memset(p, 0, n * size);
    return p;
}
/* mallocng keeps an object's offset when it grows it in place, or through the remap of a
 * single-slot mapping (the no-op one included). enframe aligned that offset only to the granule of
 * the size the object was allocated at, and the size published here can have a larger one, so the
 * child would not be exactly representable: the spatial heap's exactness check traps (cause 2)
 * and CDERIVE refuses it (cause 29). Measured on the twin, both arms: 32720 -> 32768 in place from
 * a base = 32 mod 64, and 261888 -> 262144 through the no-op remap from a base = 256 mod 512,
 * faulted on every misaligned round and passed on every aligned one; SPEC's gcc train (200.c,
 * scilab.c) and perlbench test (japh/abigail, re/pat_rt_report) stop on the same instructions.
 * Such a block is moved here, malloc-copy-free as mallocng's own moving path does, to a slot
 * enframed at the new size. Done in the runtime and not in realloc.c because
 * check-mallocng-policy.py holds mallocng's in-place decision token-identical to upstream, and
 * the decision is not wrong: only the published bounds are. The copy reads the old block through
 * its slot, which spans it whichever route kept it. If the new block cannot be had, NULL: the
 * caller keeps the old object, which a remap has already moved and whose old child is dead. */
static void *exact_or_moved(void *raw, size_t want, size_t old)
{
    if (!raw) return NULL;
    struct capstone_malloc_slot *r = record_of_result(raw);
    if (!r) __builtin_trap();
    if (capstone_repr_ok(r->key, want)) return raw;
    void *fresh = __capstone_mallocng_malloc(want);
    if (!fresh) { __capstone_malloc_ng_null++; return NULL; }
    memcpy(fresh, raw, old < want ? old : want);
    __capstone_mallocng_free(raw);
    return fresh;
}
void *realloc(void *p, size_t n)
{
    if (!p) return malloc(n);
    lock_heap();
    struct capstone_malloc_slot *r = checked(p);
    /* Round the REQUEST as malloc does, not just the published size: publish() derives the child
     * over the nominal size, so that size must be representable, and asking mallocng for the
     * rounded size is what makes the block actually that big. Publishing a rounded size over an
     * unrounded block would hand out a capability past the allocation. */
    size_t want = request_round(n);
    if (want < n) { unlock_heap(); errno = ENOMEM; return NULL; }
    size_t old = r->nominal;
    void *q = ensure_ids(IDS_PER_CALL) ? NULL
            : publish(exact_or_moved(__capstone_mallocng_realloc(backend_pointer(r), want, r), want, old),
                      want, NULL);
    unlock_heap(); return q;
}
void *__libc_realloc(void *p, size_t n) { return realloc(p, n); }
void *aligned_alloc(size_t a, size_t n)
{
    lock_heap();
    /* Same as malloc and realloc. This one is why posix_memalign(4096, 5001) faulted: 5001 has
     * granule 8 and 5001 % 8 == 1, so the child derived over the raw size is unrepresentable and
     * the CDERIVE in capstone_cap_derive raises ILLEGAL_OP_VAL. The arena itself was fine; the
     * rounding added to malloc had simply never been applied to its two siblings. */
    size_t want = request_round(n);
    if (want < n) { unlock_heap(); errno = ENOMEM; return NULL; }
    void *p = ensure_ids(IDS_PER_CALL) ? NULL
            : publish(__capstone_mallocng_aligned_alloc(a, want), want, NULL);
    unlock_heap(); return p;
}
int posix_memalign(void **out, size_t a, size_t n)
{
    if (!a || (a & (a-1)) || a < sizeof(void *)) return EINVAL;
    int e = errno;
    void *p = aligned_alloc(a, n);
    int error = p ? 0 : errno;
    errno = e;
    if (!error) *out = p;
    return error;
}
void *memalign(size_t a, size_t n) { return aligned_alloc(a, n); }
size_t malloc_usable_size(void *p)
{
    if (!p) return 0;
    lock_heap(); size_t n = checked(p)->nominal; unlock_heap(); return n;
}
/* musl's group-release decisions are its own and happen during free; what is left to trim is the
 * runtime's kept mappings ("Recycled group mappings"), which are unmapped here, so a program that
 * asks for its freed memory back gets it (the runtime gate's heap-threads case does, and reads the
 * pages released). Returns 1 when something was released, as glibc's does. */
int malloc_trim(size_t pad)
{
    (void)pad;
    lock_heap();
    int released = recycle_count != 0;
    while (recycle_count) {
        --recycle_count;
        void *root = recycle[recycle_count].root;
        size_t bytes = recycle[recycle_count].bytes;
        recycle[recycle_count].root = NULL; recycle[recycle_count].bytes = 0;
        recycle_bytes -= bytes;
        munmap(root, bytes);
    }
    for (unsigned b = 0; b < COARSE_BUCKETS; ++b) {
        struct coarse_region *r = coarse_heads[b];
        while (r) {
            struct coarse_region *next = r->hash_next;
            if (r->free == r->mask) released |= coarse_drop(r);
            r = next;
        }
    }
    unlock_heap();
    return released;
}
