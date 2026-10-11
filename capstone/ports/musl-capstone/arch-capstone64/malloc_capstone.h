#ifndef CAPSTONE_MALLOC_ARCH_H
#define CAPSTONE_MALLOC_ARCH_H
#include <stddef.h>

/* Capability representation of a mallocng slot. The allocator alone chooses
 * the slot, size class, offset, retention and release. No placement policy or
 * fixed object-count limit lives in this interface. */
struct capstone_malloc_slot {
    void *authority, *raw, *group;
    struct capstone_malloc_slot *next;
    size_t key, nominal;
    unsigned index, kind;
    unsigned char prefix[4] __attribute__((aligned(4)));
};
void *__capstone_malloc_map(size_t);
void *__capstone_malloc_group_attach(struct capstone_malloc_slot *, void **,
                                    void *, size_t);
void *__capstone_malloc_group_map(struct capstone_malloc_slot *, void **,
                                 size_t, size_t, unsigned);
void *__capstone_malloc_group_nested(struct capstone_malloc_slot *, void **,
                                    size_t, unsigned, struct capstone_malloc_slot *);
void *__capstone_malloc_group_remap(struct capstone_malloc_slot *, void **,
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
#endif
