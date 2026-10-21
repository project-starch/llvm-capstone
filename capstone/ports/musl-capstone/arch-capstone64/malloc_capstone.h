#ifndef CAPSTONE_MALLOC_ARCH_H
#define CAPSTONE_MALLOC_ARCH_H
#include <stddef.h>

/* Capability representation of a mallocng slot. The allocator alone chooses
 * the slot, size class, offset, retention and release. No placement policy or
 * fixed object-count limit lives in this interface. */
struct capstone_malloc_slot {
    void *parent, *issued, *group;
    struct capstone_malloc_slot *next;
    size_t key, nominal;
    unsigned index, kind;
    unsigned char prefix[4] __attribute__((aligned(4)));
};
void *__capstone_malloc_map(size_t);
void *__capstone_malloc_group_attach(struct capstone_malloc_slot *,
                                    void *, size_t);
void *__capstone_malloc_group_map(struct capstone_malloc_slot *,
                                 size_t, size_t, unsigned);
void *__capstone_malloc_group_nested(struct capstone_malloc_slot *,
                                    size_t, unsigned, struct capstone_malloc_slot *);
void *__capstone_malloc_group_remap(struct capstone_malloc_slot *,
                                   void *, size_t, size_t);
void *__capstone_malloc_slot_take(struct capstone_malloc_slot *);
void *__capstone_malloc_slot_start(struct capstone_malloc_slot *);
void __capstone_malloc_slot_key(struct capstone_malloc_slot *, void *, size_t);
void __capstone_malloc_slot_release(struct capstone_malloc_slot *);
void __capstone_malloc_nested_release(struct capstone_malloc_slot *);
void *__capstone_malloc_prepare_free(void *);
struct capstone_malloc_slot *__capstone_malloc_find(const void *);
void *__capstone_mallocng_malloc(size_t);
void *__capstone_mallocng_realloc(void *, size_t);
void *__capstone_mallocng_aligned_alloc(size_t, size_t);
void __capstone_mallocng_free(void *);

/* Option (iii): the bounds granule, inline, because enframe needs it on every allocation and a
 * cross-TU call would cost one per malloc.
 *
 * THIS DUPLICATES capstone/runtime/include/capstone/representable.h, which is the single source
 * for the runtime side. It is duplicated rather than included because the musl survey compiles
 * with -nostdinc and only musl's own include directories (survey-musl-capstone.py:91-99), and
 * widening that flag set would let musl reach arbitrary runtime headers. Two copies can drift, so
 * capstone/tests/exact-bounds-iii/granule-agree-test.c compiles BOTH and fails if they disagree on
 * any length -- that test is the reason this duplication is safe to have.
 *
 * The rule: exact below 4 KiB (the encoder's exact branch), else 2^(msb(len)-9). Both the base and
 * the length must be multiples of it, because the encoder truncates an unaligned base DOWNWARD and
 * only rounds the end up. */
static inline size_t __capstone_malloc_granule(size_t len)
{
	if (len < 4096u)
		return 1u;
	return (size_t)1 << ((size_t)(63 - __builtin_clzl((unsigned long)len)) - 9u);
}
static inline size_t __capstone_malloc_repr_round(size_t len)
{
	size_t g = __capstone_malloc_granule(len ? len : 1u);
	return ((len ? len : 1u) + g - 1u) / g * g;
}

#endif
