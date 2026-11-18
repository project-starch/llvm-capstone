#!/usr/bin/env python3
"""fix-lookup.py SV_C [more_sv-name]: replace the arena lookup of the sv-heads-as-sublet-lifetimes patch (nested lane,
2026-11-18). Dev's revision (blob 72db9236) finds a head's arena by walking PL_sv_arenaroot, one capability load
through each arena. This keeps every arena in an array sorted by address and finds the arena by binary search over
that array alone: no arena is read. Used for the 5.36.3 patch and the SPEC 5.22.1 port. Every edit is asserted."""
import sys
p = sys.argv[1]; s = open(p).read()
def edit(old, new):
    global s
    assert s.count(old) == 1, (old[:60], s.count(old))
    s = s.replace(old, new)

edit('''static SV *sv_arena_last;	/* the arena found last; most lookups repeat it */

/* The arena holding the head at sv's address, or NULL. sv is never read. */
static SV *
S_sv_arena_of(const SV *sv)
{
    SV *sva = sv_arena_last;
    if (sva && PTR2UV(sv) - PTR2UV(sva + 1) < SV_ARENA_HEADS * sizeof(SV))
        return sva;
    for (sva = PL_sv_arenaroot; sva; sva = MUTABLE_SV(SvANY(sva)))
        if (PTR2UV(sv) - PTR2UV(sva + 1) < SV_ARENA_HEADS * sizeof(SV))
            return sv_arena_last = sva;
    return NULL;
}
''', '''static SV *sv_arena_last;	/* the arena found last; most lookups repeat it */

/* Every arena, sorted by address. A lookup searches this array and reads no
   arena: walking PL_sv_arenaroot instead costs one load through each arena,
   and each arena is a heap object with a lifetime of its own. */
static SV **sv_arena_index;
static size_t sv_arena_count, sv_arena_cap;

/* Enter a new arena in the index. Perl_more_sv is the only caller, and the
   only place that adds an arena. */
static void
S_sv_arena_register(SV *sva)
{
    size_t lo = 0, hi = sv_arena_count;
    if (sv_arena_count == sv_arena_cap) {
        sv_arena_cap = sv_arena_cap ? sv_arena_cap * 2 : 64;
        Renew(sv_arena_index, sv_arena_cap, SV *);
    }
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (PTR2UV(sv_arena_index[mid]) < PTR2UV(sva))
            lo = mid + 1;
        else
            hi = mid;
    }
    Move(sv_arena_index + lo, sv_arena_index + lo + 1, sv_arena_count - lo, SV *);
    sv_arena_index[lo] = sva;
    sv_arena_count++;
}

/* The arena holding the head at sv's address, or NULL. sv is never read. */
static SV *
S_sv_arena_of(const SV *sv)
{
    SV *sva = sv_arena_last;
    size_t lo = 0, hi = sv_arena_count;
    if (sva && PTR2UV(sv) - PTR2UV(sva + 1) < SV_ARENA_HEADS * sizeof(SV))
        return sva;
    /* the last arena that starts at or below sv */
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (PTR2UV(sv_arena_index[mid]) <= PTR2UV(sv))
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo == 0)
        return NULL;
    sva = sv_arena_index[lo - 1];
    if (PTR2UV(sv) - PTR2UV(sva + 1) < SV_ARENA_HEADS * sizeof(SV))
        return sv_arena_last = sva;
    return NULL;
}
''')
edit('''    sv_add_arena(chunk, PERL_ARENA_SIZE, 0);
    ((SV *)chunk)->sv_u.svu_rv = (SV *)capstone_cap_derive(chunk,''', '''    sv_add_arena(chunk, PERL_ARENA_SIZE, 0);
    S_sv_arena_register((SV *)chunk);
    ((SV *)chunk)->sv_u.svu_rv = (SV *)capstone_cap_derive(chunk,''')
edit('''#ifdef PERL_CAPSTONE_SUBLET
    sv_arena_last = NULL;
#endif''', '''#ifdef PERL_CAPSTONE_SUBLET
    sv_arena_last = NULL;
    Safefree(sv_arena_index);
    sv_arena_index = NULL;
    sv_arena_count = sv_arena_cap = 0;
#endif''')
open(p, 'w').write(s)
print('lookup fixed')
