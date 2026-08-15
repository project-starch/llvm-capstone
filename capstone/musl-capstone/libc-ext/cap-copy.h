/* The two block-copy routines in this libc that may move memory holding
 * CAPABILITIES. memcpy, memmove and realloc share them; nothing else needs them.
 *
 * WHY A BYTE LOOP IS WRONG HERE, which is not obvious and cost a full session.
 * A capability is 16 bytes plus a TAG that lives beside the memory, not in it.
 * Only a capability-wide load/store carries the tag; sixteen byte moves deliver
 * exactly the right bytes with the tag CLEARED. Nothing faults at the copy --
 * the pointer arrives looking correct and faults on the first dereference,
 * arbitrarily far away, as a bare "cause = 24" in a function that did nothing
 * wrong.
 *
 * THE DEFECT THIS WAS WRITTEN FOR. mruby grows its VM stack by memcpying the old
 * stack into a larger allocation (`stack_copy`, src/vm.c). With a byte-at-a-time
 * memcpy every object pointer on the stack lost its tag, so the interpreter
 * faulted in `mrb_class` on the next method call and in `mrb_gc_mark` on the
 * next collection. It was told apart from a bounds or revocation problem by
 * being IDENTICAL under three allocators -- revoke-on-free, the same with
 * revocation disabled, and the plain libc one. An allocator-shaped symptom that
 * does not move when the allocator changes is not an allocator problem.
 *
 * SEPARATE ARCHIVE MEMBER, not a static inline. Programs that predate this libc
 * bring their own memcpy (benchmarks/beebs/adapted/beebs_freestanding_string.c),
 * and the linker then prefers theirs. If realloc reached the copy through
 * `memcpy` it would silently get the program's byte-at-a-time one and strip tags
 * again; if the helper lived in memcpy.c, pulling memmove would pull memcpy's
 * member and collide on the symbol the program defines. Its own member, with
 * names no program defines, is the only arrangement with neither problem.
 *
 * These names are also why this member is built at -O0 while the rest of
 * libc-ext is -O1: see the note in build-musl-capstone.sh. Keeping it to ONE
 * small file is the point -- malloc and memmove keep their optimisation.
 */
#ifndef CAPSTONE_LIBC_CAP_COPY_H
#define CAPSTONE_LIBC_CAP_COPY_H

#include <stddef.h>

/* Forwards. Safe when dest is below src, or the ranges do not overlap. */
void __capstone_cap_copy_fwd(void *dest, const void *src, size_t n);

/* Backwards, for an overlapping move with dest above src. Kept capability-wide
   for the same reason as the forward one: shifting an ARRAY OF POINTERS up by an
   element is exactly this case. */
void __capstone_cap_copy_bwd(void *dest, const void *src, size_t n);

#endif
