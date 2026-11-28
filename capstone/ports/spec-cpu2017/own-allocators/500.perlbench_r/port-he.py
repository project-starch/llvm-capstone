#!/usr/bin/env python3
"""port-he.py DIR: hash entries (HE) as Sublet lifetimes in SPEC CPU2017 500.perlbench_r's Perl 5.22.1 sources (nested
lane, 2026-11-22), under PERL_CAPSTONE_SUBLET_HE. Apply after port-sv-bodies.py; compile with PERL_CAPSTONE_SUBLET and
PERL_CAPSTONE_SUBLET_HE, with or without PERL_CAPSTONE_SUBLET_BODIES. Without the new define the preprocessed sources
are those of the bodies port (hv.c's and sv.c's objects are byte-identical; checked per build). Every edit is asserted
to match exactly once.

A hash entry is popped from PL_body_roots[HE_SVSLOT] by hv.c's new_he and pushed back by del_HE; the arenas come from
sv.c's Perl_more_bodies. Under the new define those arenas are carved from the segments of port-sv-bodies.py, each HE
is handed out as a child lifetime bounded to the entry (CDERIVE), and del_HE revokes it (CREVOKE) before it links the
entry through the segment's own pointer. The segment table is shared: sv.c defines it (external under the define) and
hv.c reaches it through capstone_sv_segs.h, which this script writes into DIR and which repeats the segment lookup,
issue and retire of sv.c inline. Shared string table entries (PL_strtab) are HEs too and are covered the same way."""
import sys
from pathlib import Path
D = Path(sys.argv[1])
NR = 'PERL_CAPSTONE_SUBLET_BODIES_NOREVOKE'   # the fixtures' control: no CREVOKE in either file


def edit(name, old, new):
    p = D / name; s = p.read_text()
    assert s.count(old) == 1, (name, old[:70], s.count(old))
    p.write_text(s.replace(old, new))


# sv.c: the segments serve bodies, hash entries, or both
edit('sv.c', """#ifdef PERL_CAPSTONE_SUBLET_BODIES
#  ifndef PERL_CAPSTONE_SUBLET
#    error "PERL_CAPSTONE_SUBLET_BODIES builds on PERL_CAPSTONE_SUBLET (the heads port)"
#  endif
""", """#if defined(PERL_CAPSTONE_SUBLET_BODIES) || defined(PERL_CAPSTONE_SUBLET_HE)
#  ifndef PERL_CAPSTONE_SUBLET
#    error "PERL_CAPSTONE_SUBLET_BODIES builds on PERL_CAPSTONE_SUBLET (the heads port)"
#  endif
/* which arenas the segments carve: SV bodies, hash entries (HE_SVSLOT is SVt_NULL), or both */
#  if defined(PERL_CAPSTONE_SUBLET_BODIES) && defined(PERL_CAPSTONE_SUBLET_HE)
#    define SV_SEG_CARVES(t) 1
#  elif defined(PERL_CAPSTONE_SUBLET_BODIES)
#    define SV_SEG_CARVES(t) ((t) != SVt_NULL)
#  else
#    define SV_SEG_CARVES(t) ((t) == SVt_NULL)
#  endif
""")
edit('sv.c', """static struct sv_body_seg sv_body_segs[SV_BODY_SEGS];
static unsigned sv_body_nsegs;
""", """#ifdef PERL_CAPSTONE_SUBLET_HE	/* hv.c reaches them through capstone_sv_segs.h */
struct sv_body_seg sv_body_segs[SV_BODY_SEGS];
unsigned sv_body_nsegs;
#else
static struct sv_body_seg sv_body_segs[SV_BODY_SEGS];
static unsigned sv_body_nsegs;
#endif
""")
edit('sv.c', """#ifdef PERL_CAPSTONE_SUBLET_BODIES
    if (sv_type != SVt_NULL)	/* HE_SVSLOT: sv.h defines the name for hv.c only */
	adesc->arena = S_sv_body_carve(good_arena_size);""", """#if defined(PERL_CAPSTONE_SUBLET_BODIES) || defined(PERL_CAPSTONE_SUBLET_HE)
    if (SV_SEG_CARVES(sv_type))	/* HE_SVSLOT: sv.h defines the name for hv.c only */
	adesc->arena = S_sv_body_carve(good_arena_size);""")
edit('sv.c', """#ifdef PERL_CAPSTONE_SUBLET_BODIES
		if (S_sv_body_seg_of(""", """#if defined(PERL_CAPSTONE_SUBLET_BODIES) || defined(PERL_CAPSTONE_SUBLET_HE)
		if (S_sv_body_seg_of(""")
edit('sv.c', """#ifdef PERL_CAPSTONE_SUBLET_BODIES
    while (sv_body_nsegs)""", """#if defined(PERL_CAPSTONE_SUBLET_BODIES) || defined(PERL_CAPSTONE_SUBLET_HE)
    while (sv_body_nsegs)""")

# hv.c's view of the segments
(D / 'capstone_sv_segs.h').write_text("""/* capstone_sv_segs.h (port-he.py, nested lane, 2026-11-22): hv.c's view of sv.c's SV-body segments
   (port-sv-bodies.py), for hash entries under PERL_CAPSTONE_SUBLET_HE. The struct, the sizes and the three helpers
   repeat sv.c's; the table itself is sv.c's. */
#ifndef CAPSTONE_SV_SEGS_H
#define CAPSTONE_SV_SEGS_H
#include <capstone/capability.h>
#define SV_BODY_SEG\t\t((UV)1 << 25)
#define SV_BODY_SEGS\t\t32
struct sv_body_seg { char *chunk; char *life; UV lo, next; };
extern struct sv_body_seg sv_body_segs[SV_BODY_SEGS];
extern unsigned sv_body_nsegs;

PERL_STATIC_INLINE struct sv_body_seg *
S_sv_seg_of(const UV a)
{
    unsigned i = sv_body_nsegs;
    while (i--)
\tif (a - sv_body_segs[i].lo < SV_BODY_SEG)
\t    return &sv_body_segs[i];
    return NULL;
}

/* the free entry at own (a segment's own pointer) handed out as its child */
PERL_STATIC_INLINE void *
S_sv_seg_issue(void *own, const size_t size)
{
    const UV a = __builtin_capstone_cap_get_cursor(own);
    struct sv_body_seg *const s = S_sv_seg_of(a);
    return s ? capstone_cap_derive(s->life, a - s->lo, size) : own;
}

/* the entry's lifetime ended; the segment's own pointer */
PERL_STATIC_INLINE void *
S_sv_seg_retire(void *thing)
{
    const UV a = __builtin_capstone_cap_get_cursor(thing);
    struct sv_body_seg *const s = S_sv_seg_of(a);
    if (!s)
\treturn thing;
#ifndef """ + NR + """
    capstone_cap_revoke_child(s->life, thing);
#endif
    return s->life + (a - s->lo);
}
#endif
""")

edit('hv.c', """static const char S_strtab_error[]""", """#ifdef PERL_CAPSTONE_SUBLET_HE
#  ifndef PERL_CAPSTONE_SUBLET
#    error "PERL_CAPSTONE_SUBLET_HE builds on PERL_CAPSTONE_SUBLET (the heads port)"
#  endif
#  include "capstone_sv_segs.h"
#endif

static const char S_strtab_error[]""")
edit('hv.c', """    *root = HeNEXT(he);
    return he;
}""", """    *root = HeNEXT(he);
#ifdef PERL_CAPSTONE_SUBLET_HE
    return (HE *)S_sv_seg_issue(he, sizeof(HE));
#else
    return he;
#endif
}""")
old_del = """#define del_HE(p) \\
    STMT_START { \\
	HeNEXT(p) = (HE*)(PL_body_roots[HE_SVSLOT]);	\\
	PL_body_roots[HE_SVSLOT] = p; \\
    } STMT_END
"""
edit('hv.c', old_del, """#ifdef PERL_CAPSTONE_SUBLET_HE
#define del_HE(p) \\
    STMT_START { \\
	HE * const he_own = (HE *)S_sv_seg_retire((void *)(p)); \\
	HeNEXT(he_own) = (HE*)(PL_body_roots[HE_SVSLOT]);	\\
	PL_body_roots[HE_SVSLOT] = he_own; \\
    } STMT_END
#else
""" + old_del + "#endif\n")
print('port-he: applied to', D)
