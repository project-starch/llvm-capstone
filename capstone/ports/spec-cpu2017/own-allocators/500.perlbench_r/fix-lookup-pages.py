#!/usr/bin/env python3
"""fix-lookup-pages.py SV_C: an O(1) arena lookup that keeps upstream's arena layout (nested lane, 2026-11-21), on top
of fix-lookup.py. fix-lookup2.py aligned each arena to 4 KiB, which on test.pl raised peak memory from 101.8 MB to
110.3 MB. Here arenas stay as they are (one block: the heads, then the issued children); instead a direct-mapped table
indexed by the 4 KiB page of an address holds, per page, the arenas whose heads overlap that page. An arena's heads
(PERL_ARENA_SIZE = 4,080 bytes) span at most two pages, and since arenas do not overlap a page meets the heads of at most
two arenas, so two entries per slot suffice; other pages that share the slot only make a hit fail its verification.
A lookup checks the two entries of the head's page (each verified: the head must lie in that arena's heads), else the
one-entry cache and the binary search, which stay as they were and refill the table. The check is forced inline, and
sv.c's own uses of Perl_sv_arena_head take an inline copy (as fix-lookup3.py and fix-lookup4.py). What is protected does
not change. Every edit is asserted to match exactly once."""
import sys
p = sys.argv[1]; s = open(p).read()
def edit(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)

edit("""static SV *sv_arena_last;\t/* the arena found last; most lookups repeat it */
""", """static SV *sv_arena_last;\t/* the arena found last; most lookups repeat it */

/* fix-lookup-pages.py: per 4 KiB page (address >> 12, modulo the size), the arenas whose heads overlap the page; an
   entry is a hint, verified on every hit. */
#define SV_ARENA_PAGE\t\t4096
#define SV_ARENA_TABLE\t\t2048
static SV *sv_arena_table[SV_ARENA_TABLE][2];
#define SV_ARENA_SLOT(addr)\t(((addr) / SV_ARENA_PAGE) & (SV_ARENA_TABLE - 1))
#define SV_ARENA_HOLDS(sva, sv)\t(PTR2UV(sv) - PTR2UV((sva) + 1) < SV_ARENA_HEADS * sizeof(SV))

/* Enter SVA in the slots of the pages its heads overlap, keeping each slot's other entry if it is another arena's. */
static void
S_sv_arena_table_put(SV *sva)
{
    const UV first = PTR2UV(sva + 1), last = PTR2UV(sva + 1) + SV_ARENA_HEADS * sizeof(SV) - 1;
    UV page;
    for (page = first / SV_ARENA_PAGE; page <= last / SV_ARENA_PAGE; page++) {
        SV **slot = sv_arena_table[page & (SV_ARENA_TABLE - 1)];
        if (slot[0] == sva || slot[1] == sva)
            continue;
        slot[1] = slot[0];
        slot[0] = sva;
    }
}
""")
edit("""    Move(sv_arena_index + lo, sv_arena_index + lo + 1, sv_arena_count - lo, SV *);
    sv_arena_index[lo] = sva;
    sv_arena_count++;
}""", """    Move(sv_arena_index + lo, sv_arena_index + lo + 1, sv_arena_count - lo, SV *);
    sv_arena_index[lo] = sva;
    sv_arena_count++;
    S_sv_arena_table_put(sva);
}""")
edit("""/* The arena holding the head at sv's address, or NULL. sv is never read. */
static SV *
S_sv_arena_of(const SV *sv)
{""", """/* The arena holding the head at sv's address, or NULL, when the table does not know it: the one-entry cache, then
   the binary search, which refills the table. sv is never read. */
static SV *
S_sv_arena_of_slow(const SV *sv)
{""")
edit("""    sva = sv_arena_index[lo - 1];
    if (PTR2UV(sv) - PTR2UV(sva + 1) < SV_ARENA_HEADS * sizeof(SV))
        return sv_arena_last = sva;
    return NULL;
}""", """    sva = sv_arena_index[lo - 1];
    if (PTR2UV(sv) - PTR2UV(sva + 1) < SV_ARENA_HEADS * sizeof(SV)) {
        S_sv_arena_table_put(sva);
        return sv_arena_last = sva;
    }
    return NULL;
}

/* The arena holding the head at sv's address, or NULL: the two entries of sv's page, each verified, else the slow
   path. sv is never read. */
static inline __attribute__((always_inline)) SV *
S_sv_arena_of(const SV *sv)
{
    SV *const *const slot = sv_arena_table[SV_ARENA_SLOT(PTR2UV(sv))];
    if (LIKELY(slot[0] && SV_ARENA_HOLDS(slot[0], sv)))
        return slot[0];
    if (slot[1] && SV_ARENA_HOLDS(slot[1], sv))
        return slot[1];
    return S_sv_arena_of_slow(sv);
}""")
edit("""SV *
Perl_sv_arena_head(const SV *sv)
{
    SV *const sva = S_sv_arena_of(sv);
    return sva ? sva + 1 + (PTR2UV(sv) - PTR2UV(sva + 1)) / sizeof(SV)
               : (SV *)sv;
}
""", """static inline __attribute__((always_inline)) SV *
S_sv_arena_head_inline(const SV *sv)
{
    SV *const sva = S_sv_arena_of(sv);
    return sva ? sva + 1 + (PTR2UV(sv) - PTR2UV(sva + 1)) / sizeof(SV)
               : (SV *)sv;
}

SV *
Perl_sv_arena_head(const SV *sv)
{
    return S_sv_arena_head_inline(sv);
}
/* sv.c's own uses below this point take the inline copy. */
#define Perl_sv_arena_head(sv) S_sv_arena_head_inline(sv)
""")
edit("""    Safefree(sv_arena_index);
    sv_arena_index = NULL;
    sv_arena_count = sv_arena_cap = 0;""", """    Safefree(sv_arena_index);
    sv_arena_index = NULL;
    sv_arena_count = sv_arena_cap = 0;
    memset(sv_arena_table, 0, sizeof(sv_arena_table));""")
open(p, 'w').write(s)
print('page-indexed lookup, layout unchanged')
