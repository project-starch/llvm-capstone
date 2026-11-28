#!/usr/bin/env python3
"""port-sv-bodies.py DIR: SV bodies as Sublet lifetimes in SPEC CPU2017 500.perlbench_r's Perl 5.22.1 sources (nested
lane, 2026-11-22), under PERL_CAPSTONE_SUBLET_BODIES. Apply after port-sv-heads.py, fix-lookup.py and
fix-lookup-pages.py; compile with both PERL_CAPSTONE_SUBLET and PERL_CAPSTONE_SUBLET_BODIES. Without the new define the
preprocessed source is that of the heads port. Every edit is asserted to match exactly once.

Every body that an SV type keeps in an arena is a child lifetime bounded to the body (CDERIVE), revoked when the body
is freed (CREVOKE). The arenas are carved from segments of 32 MiB, each one Newx block whose aligned interior is the
segment's lifetime, so a body's parent is found by comparing its address with at most a few segments: no arena
lookup. The free lists hold the segment's own pointers. SvANY keeps upstream's offset, so for the PV types it points
32 bytes below the body's base (inside the representable window, which starts at least 2 KiB below a small
capability's base), and every field it is used to reach lies inside the body. Hash entries (HE_SVSLOT) share
Perl_more_bodies and keep upstream's arenas, so this arm prices bodies alone. Bodies without an arena (NOARENA types)
are untouched. sv_free_arenas frees the segments, which ends every body lifetime of the interpreter at once."""
import sys
from pathlib import Path
D = Path(sys.argv[1])


def edit(name, old, new):
    p = D / name; s = p.read_text()
    assert s.count(old) == 1, (name, old[:70], s.count(old))
    p.write_text(s.replace(old, new))


# del_body: a segment body is retired and linked through the segment's own pointer
old_del = """/* return a thing to the free list */

#define del_body(thing, root)				\\
    STMT_START {					\\
	void ** const thing_copy = (void **)thing;	\\
	*thing_copy = *root;				\\
	*root = (void*)thing_copy;			\\
    } STMT_END
"""
new_del = """#ifdef PERL_CAPSTONE_SUBLET_BODIES
#define del_body(thing, root)				\\
    STMT_START {					\\
	void ** const thing_copy = (void **)S_sv_body_retire((void *)(thing)); \\
	*thing_copy = *root;				\\
	*root = (void*)thing_copy;			\\
    } STMT_END
#else
""" + old_del + "#endif\n"
edit('sv.c', old_del, new_del)

# the segments, issue and retire, before sv_free_arenas (their first user)
edit('sv.c', """*/
void
Perl_sv_free_arenas(pTHX)
{
""", """*/
#ifdef PERL_CAPSTONE_SUBLET_BODIES
#  ifndef PERL_CAPSTONE_SUBLET
#    error "PERL_CAPSTONE_SUBLET_BODIES builds on PERL_CAPSTONE_SUBLET (the heads port)"
#  endif
/* Capstone Sublet, SV bodies (port-sv-bodies.py): every arena body is a child
   lifetime of the segment its arena was carved from, bounded to the body and
   revoked when the body is freed. */
#define SV_BODY_SEG\t\t((UV)1 << 25)\t/* 32 MiB of arenas a segment */
#define SV_BODY_SEG_ALIGN\t((UV)1 << 20)\t/* exact Sublet bounds for the segment */
#define SV_BODY_SEGS\t\t32
struct sv_body_seg { char *chunk; char *life; UV lo, next; };
static struct sv_body_seg sv_body_segs[SV_BODY_SEGS];
static unsigned sv_body_nsegs;

/* bytes of arena from the newest segment, as the segment's own pointer */
static char *
S_sv_body_carve(const size_t bytes)
{
    struct sv_body_seg *s = sv_body_nsegs ? &sv_body_segs[sv_body_nsegs - 1] : NULL;
    if (!s || s->lo + SV_BODY_SEG - s->next < bytes) {
\tchar *chunk;
\tUV at;
\tif (sv_body_nsegs == SV_BODY_SEGS)
\t    Perl_croak_nocontext("PERL_CAPSTONE_SUBLET_BODIES: out of body segments");
\tNewx(chunk, SV_BODY_SEG + SV_BODY_SEG_ALIGN, char);
\ts = &sv_body_segs[sv_body_nsegs++];
\ts->chunk = chunk;
\tat = (__builtin_capstone_cap_get_cursor(chunk) + SV_BODY_SEG_ALIGN - 1)
\t     & ~(SV_BODY_SEG_ALIGN - 1);
\ts->life = (char *)capstone_cap_derive(chunk,
\t    at - __builtin_capstone_cap_get_base(chunk), SV_BODY_SEG);
\ts->lo = s->next = at;
    }
    s->next += bytes;
    return s->life + (s->next - bytes - s->lo);
}

/* the segment holding address a, or NULL (a body outside every segment) */
PERL_STATIC_INLINE struct sv_body_seg *
S_sv_body_seg_of(const UV a)
{
    unsigned i = sv_body_nsegs;
    while (i--)
\tif (a - sv_body_segs[i].lo < SV_BODY_SEG)
\t    return &sv_body_segs[i];
    return NULL;
}

/* hand out the free body at own (a segment's own pointer) as its child */
PERL_STATIC_INLINE void *
S_sv_body_issue(void *own, const size_t size)
{
    const UV a = __builtin_capstone_cap_get_cursor(own);
    struct sv_body_seg *const s = S_sv_body_seg_of(a);
    return s ? capstone_cap_derive(s->life, a - s->lo, size) : own;
}

/* end the lifetime of the body at body's address; the segment's own pointer */
PERL_STATIC_INLINE void *
S_sv_body_retire(void *body)
{
    const UV a = __builtin_capstone_cap_get_cursor(body);
    struct sv_body_seg *const s = S_sv_body_seg_of(a);
    if (!s)
\treturn body;
#ifndef PERL_CAPSTONE_SUBLET_BODIES_NOREVOKE	/* the fixture's control build */
    capstone_cap_revoke_child(s->life, body);
#endif
    return s->life + (a - s->lo);
}

#endif

void
Perl_sv_free_arenas(pTHX)
{
""")

# Perl_more_bodies: SV body arenas from the segments, hash entries as upstream
edit('sv.c', """    Newx(adesc->arena, good_arena_size, char);
    adesc->size = good_arena_size;""", """#ifdef PERL_CAPSTONE_SUBLET_BODIES
    if (sv_type != SVt_NULL)	/* HE_SVSLOT: sv.h defines the name for hv.c only */
\tadesc->arena = S_sv_body_carve(good_arena_size);
    else
#endif
    Newx(adesc->arena, good_arena_size, char);
    adesc->size = good_arena_size;""")

# new_body_inline: the popped body is handed out as its child
edit('sv.c', """	*(r3wt) = *(void**)(xpv); \\
    } STMT_END
""", """	*(r3wt) = *(void**)(xpv); \\
	SV_BODY_ISSUE(xpv, sv_type); \\
    } STMT_END
#ifdef PERL_CAPSTONE_SUBLET_BODIES
#  define SV_BODY_ISSUE(xpv, sv_type) \\
	((xpv) = (PTR_TBL_ENT_t *)S_sv_body_issue((void *)(xpv), bodies_by_type[sv_type].body_size))
#else
#  define SV_BODY_ISSUE(xpv, sv_type) NOOP
#endif
""")

# sv_free_arenas: segment arenas are not Newx blocks; the segments go at the end, ending every body lifetime
edit('sv.c', """		assert(aroot->set[i].arena);
		Safefree(aroot->set[i].arena);""", """		assert(aroot->set[i].arena);
#ifdef PERL_CAPSTONE_SUBLET_BODIES
		if (S_sv_body_seg_of(__builtin_capstone_cap_get_cursor(aroot->set[i].arena)))
		    continue;
#endif
		Safefree(aroot->set[i].arena);""")
edit('sv.c', """    PL_body_arenas = 0;

    i = PERL_ARENA_ROOTS_SIZE;""", """    PL_body_arenas = 0;
#ifdef PERL_CAPSTONE_SUBLET_BODIES
    while (sv_body_nsegs)
\tSafefree(sv_body_segs[--sv_body_nsegs].chunk);
#endif

    i = PERL_ARENA_ROOTS_SIZE;""")
print('port-sv-bodies: applied to', D)
