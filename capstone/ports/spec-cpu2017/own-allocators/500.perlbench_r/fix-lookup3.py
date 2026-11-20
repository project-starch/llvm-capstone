#!/usr/bin/env python3
"""fix-lookup3.py SV_C: on top of fix-lookup2.py, the table check inline and the search out of line (nested lane,
2026-11-21). With fix-lookup2.py the compiler keeps S_sv_arena_of out of line (the binary search makes it large): 53.8 M
calls on test.pl at 26 instructions each. Here S_sv_arena_of is the verified table check alone, forced inline, and the
one-entry cache and the binary search move to S_sv_arena_of_slow, which also refills the table. What is protected does
not change. Every edit is asserted to match exactly once."""
import sys
p = sys.argv[1]; s = open(p).read()
def edit(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)
edit("""/* The arena holding the head at sv's address, or NULL. sv is never read. */
static SV *
S_sv_arena_of(const SV *sv)
{
    SV *sva = sv_arena_last;
    size_t lo = 0, hi = sv_arena_count;
    if (sva && PTR2UV(sv) - PTR2UV(sva + 1) < SV_ARENA_HEADS * sizeof(SV))
        return sva;
    {
        /* The table: the entry for sv's 4 KiB block, if it is that block's arena and holds sv. */
        const UV block = PTR2UV(sv) & ~(UV)(SV_ARENA_ALIGN - 1);
        SV *const hint = sv_arena_table[SV_ARENA_SLOT(block)];
        if (hint && PTR2UV(hint) == block
            && PTR2UV(sv) - PTR2UV(hint + 1) < SV_ARENA_HEADS * sizeof(SV))
            return sv_arena_last = hint;
    }
    /* the last arena that starts at or below sv */""", """/* The arena holding the head at sv's address, or NULL, when the table does not know it: the one-entry cache, then
   the binary search, which refills the table. sv is never read. */
static SV *
S_sv_arena_of_slow(const SV *sv)
{
    SV *sva = sv_arena_last;
    size_t lo = 0, hi = sv_arena_count;
    if (sva && PTR2UV(sv) - PTR2UV(sva + 1) < SV_ARENA_HEADS * sizeof(SV))
        return sva;
    /* the last arena that starts at or below sv */""")
edit("""    sva = sv_arena_index[lo - 1];
    if (PTR2UV(sv) - PTR2UV(sva + 1) < SV_ARENA_HEADS * sizeof(SV)) {
        sv_arena_table[SV_ARENA_SLOT(PTR2UV(sva))] = sva;
        return sv_arena_last = sva;
    }
    return NULL;
}""", """    sva = sv_arena_index[lo - 1];
    if (PTR2UV(sv) - PTR2UV(sva + 1) < SV_ARENA_HEADS * sizeof(SV)) {
        sv_arena_table[SV_ARENA_SLOT(PTR2UV(sva))] = sva;
        return sv_arena_last = sva;
    }
    return NULL;
}

/* The arena holding the head at sv's address, or NULL. The table's entry for sv's 4 KiB block, verified: it must be
   that block's arena and hold sv; else the slow path. sv is never read. */
static inline __attribute__((always_inline)) SV *
S_sv_arena_of(const SV *sv)
{
    const UV block = PTR2UV(sv) & ~(UV)(SV_ARENA_ALIGN - 1);
    SV *const hint = sv_arena_table[SV_ARENA_SLOT(block)];
    if (LIKELY(hint && PTR2UV(hint) == block
               && PTR2UV(sv) - PTR2UV(hint + 1) < SV_ARENA_HEADS * sizeof(SV)))
        return hint;
    return S_sv_arena_of_slow(sv);
}""")
open(p, 'w').write(s)
print('table check inline')
