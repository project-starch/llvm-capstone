#!/usr/bin/env python3
"""fix-lookup2.py SV_C: an O(1) arena lookup for the sv-heads port (nested lane, 2026-11-21), on top of fix-lookup.py.
fix-lookup.py finds a head's arena by binary search over an address-sorted index: about 15 halvings on SPEC's test.pl
(26,678 arenas), and three quarters of the instructions the port adds. Here:
  - each arena is one 4 KiB block aligned to 4 KiB: PERL_ARENA_SIZE (4080) of heads, then, in the last 16 bytes, the
    arena's own pointer to its issued-child array, which moves to a block of its own;
  - a direct-mapped table of 4,096 arena pointers, indexed by the arena's address >> 12, is filled at registration
    and on every slow-path find; a hit is verified (the entry's address must be the head's address rounded down to
    4 KiB, and the head must lie in that arena's heads), so a stale or colliding entry falls through to the binary
    search, which stays as it was;
  - sv_free_arenas frees each arena's issued array and clears the table.
Nothing about what is protected changes: the same lifetimes are issued and revoked; only how the arena is found.
Every edit is asserted to match exactly once."""
import sys
p = sys.argv[1]; s = open(p).read()
def edit(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)

edit("""#define SV_ARENA_ISSUED(sva)\t((SV **)((char *)(sva) + PERL_ARENA_SIZE))
""", """/* fix-lookup2.py: an arena is a 4 KiB block aligned to 4 KiB; its last 16 bytes hold the arena's own pointer to the
   array of issued children, which is a block of its own. */
#define SV_ARENA_ALIGN\t\t4096
#define SV_ARENA_ISSUED(sva)\t(*(SV ***)((char *)(sva) + PERL_ARENA_SIZE))
typedef char sv_arena_fits_its_block[(PERL_ARENA_SIZE + sizeof(SV **) <= SV_ARENA_ALIGN) ? 1 : -1];
/* Direct-mapped: arena address >> 12, modulo the size. An entry is a hint, verified on every hit. */
#define SV_ARENA_TABLE\t\t4096
static SV *sv_arena_table[SV_ARENA_TABLE];
#define SV_ARENA_SLOT(addr)\t(((addr) / SV_ARENA_ALIGN) & (SV_ARENA_TABLE - 1))
""")
edit("""    Move(sv_arena_index + lo, sv_arena_index + lo + 1, sv_arena_count - lo, SV *);
    sv_arena_index[lo] = sva;
    sv_arena_count++;
}""", """    Move(sv_arena_index + lo, sv_arena_index + lo + 1, sv_arena_count - lo, SV *);
    sv_arena_index[lo] = sva;
    sv_arena_count++;
    sv_arena_table[SV_ARENA_SLOT(PTR2UV(sva))] = sva;
}""")
edit("""    if (sva && PTR2UV(sv) - PTR2UV(sva + 1) < SV_ARENA_HEADS * sizeof(SV))
        return sva;
    /* the last arena that starts at or below sv */""", """    if (sva && PTR2UV(sv) - PTR2UV(sva + 1) < SV_ARENA_HEADS * sizeof(SV))
        return sva;
    {
        /* The table: the entry for sv's 4 KiB block, if it is that block's arena and holds sv. */
        const UV block = PTR2UV(sv) & ~(UV)(SV_ARENA_ALIGN - 1);
        SV *const hint = sv_arena_table[SV_ARENA_SLOT(block)];
        if (hint && PTR2UV(hint) == block
            && PTR2UV(sv) - PTR2UV(hint + 1) < SV_ARENA_HEADS * sizeof(SV))
            return sv_arena_last = hint;
    }
    /* the last arena that starts at or below sv */""")
edit("""    sva = sv_arena_index[lo - 1];
    if (PTR2UV(sv) - PTR2UV(sva + 1) < SV_ARENA_HEADS * sizeof(SV))
        return sv_arena_last = sva;
    return NULL;""", """    sva = sv_arena_index[lo - 1];
    if (PTR2UV(sv) - PTR2UV(sva + 1) < SV_ARENA_HEADS * sizeof(SV)) {
        sv_arena_table[SV_ARENA_SLOT(PTR2UV(sva))] = sva;
        return sv_arena_last = sva;
    }
    return NULL;""")
edit("""    /* The heads, then the child each was handed out as (above plant_SV). */
    Newxz(chunk, PERL_ARENA_SIZE + SV_ARENA_HEADS * sizeof(SV *), char);
    sv_add_arena(chunk, PERL_ARENA_SIZE, 0);""", """    /* The heads, in a 4 KiB block aligned to 4 KiB; the children each was handed out as (above plant_SV) in a block
       of their own, whose pointer is the block's last 16 bytes. */
    chunk = (char *) aligned_alloc(SV_ARENA_ALIGN, SV_ARENA_ALIGN);
    if (!chunk)
        croak_no_mem();
    memset(chunk, 0, SV_ARENA_ALIGN);
    {
        SV **issued;
        Newxz(issued, SV_ARENA_HEADS, SV *);
        *(SV ***)(chunk + PERL_ARENA_SIZE) = issued;
    }
    sv_add_arena(chunk, PERL_ARENA_SIZE, 0);""")
edit("""	if (!SvFAKE(sva))
	    Safefree(sva);""", """	if (!SvFAKE(sva)) {
#ifdef PERL_CAPSTONE_SUBLET
	    Safefree(SV_ARENA_ISSUED(sva));
#endif
	    Safefree(sva);
	}""")
edit("""    Safefree(sv_arena_index);
    sv_arena_index = NULL;
    sv_arena_count = sv_arena_cap = 0;""", """    Safefree(sv_arena_index);
    sv_arena_index = NULL;
    sv_arena_count = sv_arena_cap = 0;
    memset(sv_arena_table, 0, sizeof(sv_arena_table));""")
open(p, 'w').write(s)
print('lookup O(1) with a verified table')
