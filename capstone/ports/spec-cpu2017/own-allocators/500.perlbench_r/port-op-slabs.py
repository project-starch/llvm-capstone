#!/usr/bin/env python3
"""port-op-slabs.py DIR: ops as Sublet lifetimes of their op slabs in SPEC CPU2017 500.perlbench_r's Perl 5.22.1 sources
(nested lane, 2026-11-22), under PERL_CAPSTONE_SUBLET_OPS. Apply after port-sv-heads.py, fix-lookup.py and
fix-lookup-pages.py, alone or with the bodies and hash-entry ports; compile every object with PERL_CAPSTONE_SUBLET and
PERL_CAPSTONE_SUBLET_OPS. Without the new define the preprocessed sources are those of the heads port (checked per build:
every object byte-identical). Every edit is asserted to match exactly once.

A compiling CV's ops come from a chain of slabs (op.c's Perl_Slab_Alloc); every slot is a header of two pointers
(opslot_next, opslot_slab) followed by the op, and opslot_slab names the chain's FIRST slab, where the bookkeeping is.
Under the new define:
- each slab carries a lifetime of its own memory (opslab_life, a new field after opslab_refcnt, so no other field
  moves and no slot grows), derived when the slab is allocated;
- an op is handed out as a child of the slab that holds it (CDERIVE), bounded to its whole slot, header included,
  up to opslot_next: Perl reads the header through the op (OpSLOT, OpSLAB) and grows a LISTOP into a LOOP in place when
  the slot has room;
- Perl_Slab_Free revokes the child (CREVOKE) and links the slot through the slab's own pointer, so the free chain and
  the slot chain hold only own pointers, and a reused slot is handed out again as a new child;
- the holding slab is found by walking the chain from its first slab and comparing the op's address with each slab's
  lifetime (no slot header grows);
- opslab_force_free frees the live ops of an unfinished CV through the slab's own pointers. Those are recognised by
  their bounds (the whole slab) and not revoked: their lifetimes end when the slab is freed, as every op's does.
- op_free reads the type and the sibling of ops it may already have freed: its own comments say ops may be freed
  twice and kids before their parents. Those two reads find the op's slab in an index of every slab, by address
  (entered in S_new_slab, removed in opslab_free before the free; a lookup reads no slab and not the op), and
  go through the slab's own pointer. Every other read of an op goes through its child, so a stale op whose slot was
  reused passes the type test and faults at op_free's next read. A slab already freed is not in the index, and the
  read faults through the child.
  The index has an entry per 4 KiB page each slab touches, in an open-addressing table (revision 4). Revision 3's
  sorted array was correct (fixture `census/out/perlfxo3b-c81dca75`, test.pl `census/out/zperl-ops3-aba18d33`
  equal to stock) but cost +4.775 % on test.pl, most of it in memmove, most likely the array's shifts.
  Revision 1 read those two fields through the revoked child and faulted in three of test.pl's tests
  (comp/package_block, uni/opcroak, uni/package: `census/out/zperl-ops-b296436b`). Revision 2 took the slab's pointer
  only while opslab_force_free ran, and still faulted at op_free's type test in the fixture's mode 5
  (`census/out/perlfxo2-5a1b1473`): ops are freed after a failed compilation outside that window too.
Ops allocated outside a slab (no CV being compiled) are untouched."""
import sys
from pathlib import Path
D = Path(sys.argv[1])
NR = 'PERL_CAPSTONE_SUBLET_OPS_NOREVOKE'   # the fixture's control: no CREVOKE


def edit(name, old, new):
    p = D / name; s = p.read_text()
    assert s.count(old) == 1, (name, old[:70], s.count(old))
    p.write_text(s.replace(old, new))


# op.h: the slab's lifetime, after the refcount (pad.c reads opslab_refcnt; its offset does not move)
edit('op.h', """    size_t	opslab_refcnt;		/* number of ops */
""", """    size_t	opslab_refcnt;		/* number of ops */
# ifdef PERL_CAPSTONE_SUBLET_OPS
    char *	opslab_life;		/* the slab's memory as a lifetime; its ops are children */
# endif
""")

# op.c: the helpers, after DIFF (their first users are below it)
edit('op.c', """#define DIFF(o,p)		((size_t)((I32 **)(p) - (I32**)(o)))
""", """#define DIFF(o,p)		((size_t)((I32 **)(p) - (I32**)(o)))

#ifdef PERL_CAPSTONE_SUBLET_OPS
#  ifndef PERL_CAPSTONE_SUBLET
#    error "PERL_CAPSTONE_SUBLET_OPS builds on PERL_CAPSTONE_SUBLET (the heads port)"
#  endif
#  include <capstone/capability.h>
/* Capstone Sublet, ops (port-op-slabs.py): every slab op is a child lifetime
   of the slab that holds it, bounded to its slot, revoked when it is freed. */

/* the slab of slab's chain whose memory holds p, or NULL */
static OPSLAB *
S_op_home(OPSLAB *slab, const void *p)
{
    const UV a = __builtin_capstone_cap_get_cursor((void *)p);
    for (; slab; slab = slab->opslab_next) {
	const UV lo = __builtin_capstone_cap_get_base(slab->opslab_life);
	if (a - lo < __builtin_capstone_cap_get_end(slab->opslab_life) - lo)
	    return slab;
    }
    return NULL;
}

/* hand out the op at o (the slab's own pointer) as a child of home, bounded
   to its slot: the header below the op and the op up to the next slot */
static OP *
S_op_issue(OP *o, OPSLAB *home)
{
    OPSLOT *const slot = OpSLOT(o);
    char *life, *child;
    if (!home)
	return o;
    life = home->opslab_life;
    child = (char *)capstone_cap_derive(life,
	__builtin_capstone_cap_get_cursor(slot)
	    - __builtin_capstone_cap_get_base(life),
	(char *)slot->opslot_next - (char *)slot);
    return (OP *)(child + OPSLOT_HEADER);
}

/* end the op's lifetime; the slab's own pointer to it. An own pointer
   (opslab_force_free's) has the slab's bounds and is not a child. */
static OP *
S_op_retire(OP *o, OPSLAB *head)
{
    OPSLAB *const home = S_op_home(head, o);
    if (!home
     || (__builtin_capstone_cap_get_base(o) == __builtin_capstone_cap_get_base(home)
	 && __builtin_capstone_cap_get_end(o) == __builtin_capstone_cap_get_end(home)))
	return o;
#  ifndef """ + NR + """	/* the fixture's control build */
    capstone_cap_revoke_child(home->opslab_life, o);
#  endif
    return (OP *)((char *)home + (__builtin_capstone_cap_get_cursor(o)
				  - __builtin_capstone_cap_get_cursor(home)));
}

/* Every op slab, by page: an entry for each 4 KiB page a slab's memory
   touches, in a table with open addressing (linear probing, deletion by
   backward shift, so no tombstones). op_free reads the type and the sibling of
   an op it may already have freed (its own comments: ops may be freed twice,
   and kids may be freed before their parents). Those two reads find the op's
   slab here, reading no slab and not the op, and go through the slab's own
   pointer; every other read goes through the op's own. Revision 3 kept a sorted
   array, and its shifts cost most of its price on test.pl. */
typedef struct { OPSLAB *slab; UV page, hi; } capstone_opslab_ent;
static capstone_opslab_ent *S_op_tab;
static size_t S_op_tabmask, S_op_tabn;	/* table size - 1, entries in use */
#  define CAPSTONE_OP_PAGE 12

static size_t
S_op_slot0(UV page)
{
    return (size_t)((page * (UV)0x9E3779B97F4A7C15ULL) >> 24) & S_op_tabmask;
}

static void
S_op_tab_put(OPSLAB *slab, UV page, UV hi)
{
    size_t i = S_op_slot0(page);
    while (S_op_tab[i].slab)
	i = (i + 1) & S_op_tabmask;
    S_op_tab[i].slab = slab;
    S_op_tab[i].page = page;
    S_op_tab[i].hi = hi;
    S_op_tabn++;
}

/* S_new_slab, the only place a slab is made, sz pointers long. The bounds are
   the slab's addresses, not its lifetime's, which may be rounded outward. */
static void
S_op_slab_register(OPSLAB *slab, size_t sz)
{
    const UV lo = __builtin_capstone_cap_get_cursor(slab);
    const UV hi = lo + sz * sizeof(I32 *);
    const UV p0 = lo >> CAPSTONE_OP_PAGE, p1 = (hi - 1) >> CAPSTONE_OP_PAGE;
    UV p;
    if (!S_op_tab || 2 * (S_op_tabn + (p1 - p0 + 1)) > S_op_tabmask + 1) {
	capstone_opslab_ent *const old = S_op_tab;
	const size_t oldn = old ? S_op_tabmask + 1 : 0;
	size_t n = oldn ? oldn : 1024, i;
	while (2 * (S_op_tabn + (p1 - p0 + 1)) > n)
	    n *= 2;
	Newxz(S_op_tab, n, capstone_opslab_ent);
	S_op_tabmask = n - 1;
	S_op_tabn = 0;
	for (i = 0; i < oldn; i++)
	    if (old[i].slab)
		S_op_tab_put(old[i].slab, old[i].page, old[i].hi);
	Safefree(old);
    }
    for (p = p0; p <= p1; p++)
	S_op_tab_put(slab, p, hi);
}

/* opslab_free, the only place a slab is freed, before the free */
static void
S_op_slab_unregister(OPSLAB *slab)
{
    const UV lo = __builtin_capstone_cap_get_cursor(slab);
    UV p;
    if (!S_op_tab)
	return;
    for (p = lo >> CAPSTONE_OP_PAGE; ; p++) {
	size_t i = S_op_slot0(p), j, k;
	UV hi;
	while (S_op_tab[i].slab
	    && !(S_op_tab[i].page == p
		 && __builtin_capstone_cap_get_cursor(S_op_tab[i].slab) == lo))
	    i = (i + 1) & S_op_tabmask;
	if (!S_op_tab[i].slab)
	    return;		/* not entered: nothing more to remove */
	hi = S_op_tab[i].hi;
	/* backward shift: move up every later entry of the cluster whose home
	   slot does not lie cyclically in (i, j] */
	for (j = i; ; ) {
	    j = (j + 1) & S_op_tabmask;
	    if (!S_op_tab[j].slab)
		break;
	    k = S_op_slot0(S_op_tab[j].page);
	    if (i <= j ? (i < k && k <= j) : (i < k || k <= j))
		continue;
	    S_op_tab[i] = S_op_tab[j];
	    i = j;
	}
	S_op_tab[i].slab = NULL;
	S_op_tab[i].page = S_op_tab[i].hi = 0;
	S_op_tabn--;
	if (p == (hi - 1) >> CAPSTONE_OP_PAGE)
	    return;
    }
}

/* the op at o's address through its slab's own pointer, or o if no slab holds it */
static OP *
S_op_view(OP *o)
{
    const UV a = __builtin_capstone_cap_get_cursor(o);
    const UV p = a >> CAPSTONE_OP_PAGE;
    size_t i;
    if (!S_op_tab)
	return o;
    for (i = S_op_slot0(p); S_op_tab[i].slab; i = (i + 1) & S_op_tabmask) {
	OPSLAB *const slab = S_op_tab[i].slab;
	const UV lo = __builtin_capstone_cap_get_cursor(slab);
	if (S_op_tab[i].page == p && a - lo < S_op_tab[i].hi - lo)
	    return (OP *)((char *)slab + (a - lo));
    }
    return o;
}
#endif
""")

# S_new_slab: the slab's lifetime
edit('op.c', """    slab->opslab_first = (OPSLOT *)((I32 **)slab + sz - 1);
    return slab;
}""", """    slab->opslab_first = (OPSLOT *)((I32 **)slab + sz - 1);
#ifdef PERL_CAPSTONE_SUBLET_OPS
    slab->opslab_life = (char *)capstone_cap_derive(slab,
	__builtin_capstone_cap_get_cursor(slab)
	    - __builtin_capstone_cap_get_base(slab), sz * sizeof(I32 *));
    S_op_slab_register(slab, sz);
#endif
    return slab;
}""")

# opslab_free: the slab leaves the index before it is freed
edit('op.c', """#else
	PerlMemShared_free(slab);
#endif
        slab = slab2;
    } while (slab);""", """#else
#  ifdef PERL_CAPSTONE_SUBLET_OPS
	S_op_slab_unregister(slab);
#  endif
	PerlMemShared_free(slab);
#endif
        slab = slab2;
    } while (slab);""")

# Perl_Slab_Alloc: a reused slot and a new slot are handed out as children
edit('op.c', """	    Zero(o, opsz, I32 *);
	    o->op_slabbed = 1;
	    goto gotit;""", """	    Zero(o, opsz, I32 *);
	    o->op_slabbed = 1;
#ifdef PERL_CAPSTONE_SUBLET_OPS
	    o = S_op_issue(o, S_op_home(slab, o));
#endif
	    goto gotit;""")
edit('op.c', """    INIT_OPSLOT;
    DEBUG_S_warn((aTHX_ "allocating op at %p, slab %p", (void*)o, (void*)slab));
""", """    INIT_OPSLOT;
#ifdef PERL_CAPSTONE_SUBLET_OPS
    o = S_op_issue(o, slab2);
#endif
    DEBUG_S_warn((aTHX_ "allocating op at %p, slab %p", (void*)o, (void*)slab));
""")

# Perl_Slab_Free: the child revoked, the slot linked through the slab's own pointer
edit('op.c', """Perl_Slab_Free(pTHX_ void *op)
{
    OP * const o = (OP *)op;""", """Perl_Slab_Free(pTHX_ void *op)
{
#ifdef PERL_CAPSTONE_SUBLET_OPS
    OP * o = (OP *)op;
#else
    OP * const o = (OP *)op;
#endif""")
edit('op.c', """    slab = OpSLAB(o);
    /* If this op is already freed, our refcount will get screwy. */""", """    slab = OpSLAB(o);
#ifdef PERL_CAPSTONE_SUBLET_OPS
    o = S_op_retire(o, slab);
#endif
    /* If this op is already freed, our refcount will get screwy. */""")

# op_free: the two reads of a possibly freed op (the op itself, and a kid with its sibling)
edit('op.c', """        if (!o || o->op_type == OP_FREED)
            continue;
""", """#ifdef PERL_CAPSTONE_SUBLET_OPS
        if (!o || S_op_view(o)->op_type == OP_FREED)
#else
        if (!o || o->op_type == OP_FREED)
#endif
            continue;
""")
edit('op.c', """                nextkid = OpSIBLING(kid); /* Get before next freeing kid */
                if (!kid || kid->op_type == OP_FREED)
""", """#ifdef PERL_CAPSTONE_SUBLET_OPS
                nextkid = OpSIBLING(S_op_view(kid)); /* Get before next freeing kid */
                if (!kid || S_op_view(kid)->op_type == OP_FREED)
#else
                nextkid = OpSIBLING(kid); /* Get before next freeing kid */
                if (!kid || kid->op_type == OP_FREED)
#endif
""")
print('port-op-slabs: applied to', D)
