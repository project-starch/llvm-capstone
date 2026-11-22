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
  twice and kids before their parents. Those two reads find the op's slab in an index of every slab, sorted by
  address (entered in S_new_slab, removed in opslab_free before the free; a lookup reads no slab and not the op), and
  go through the slab's own pointer. Every other read of an op goes through its child, so a stale op whose slot was
  reused passes the type test and faults at op_free's next read. A slab already freed is not in the index, and the
  read faults through the child.
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

/* Every op slab, sorted by address: [lo, hi) is the slab's memory and slab its
   own pointer. op_free reads the type and the sibling of an op it may already
   have freed (its own comments: ops may be freed twice, and kids may be freed
   before their parents). Those two reads find the op's slab here, reading no
   slab and not the op, and go through the slab's own pointer; every other read
   goes through the op's own. */
typedef struct { UV lo, hi; OPSLAB *slab; } capstone_opslab_ent;
static capstone_opslab_ent *S_op_slabs;
static size_t S_op_nslabs, S_op_capslabs;

/* the number of slabs that start at or below a */
static size_t
S_op_slab_pos(UV a)
{
    size_t lo = 0, hi = S_op_nslabs;
    while (lo < hi) {
	const size_t mid = lo + (hi - lo) / 2;
	if (S_op_slabs[mid].lo <= a)
	    lo = mid + 1;
	else
	    hi = mid;
    }
    return lo;
}

/* S_new_slab, the only place a slab is made, sz pointers long. The bounds are
   the slab's addresses, not its lifetime's, which may be rounded outward. */
static void
S_op_slab_register(OPSLAB *slab, size_t sz)
{
    const UV lo = __builtin_capstone_cap_get_cursor(slab);
    size_t i;
    if (S_op_nslabs == S_op_capslabs) {
	S_op_capslabs = S_op_capslabs ? S_op_capslabs * 2 : 64;
	Renew(S_op_slabs, S_op_capslabs, capstone_opslab_ent);
    }
    i = S_op_slab_pos(lo);
    Move(S_op_slabs + i, S_op_slabs + i + 1, S_op_nslabs - i, capstone_opslab_ent);
    S_op_slabs[i].lo = lo;
    S_op_slabs[i].hi = lo + sz * sizeof(I32 *);
    S_op_slabs[i].slab = slab;
    S_op_nslabs++;
}

/* opslab_free, the only place a slab is freed, before the free */
static void
S_op_slab_unregister(OPSLAB *slab)
{
    const UV lo = __builtin_capstone_cap_get_cursor(slab);
    const size_t i = S_op_slab_pos(lo);
    if (i == 0 || S_op_slabs[i - 1].lo != lo)
	return;
    Move(S_op_slabs + i, S_op_slabs + i - 1, S_op_nslabs - i, capstone_opslab_ent);
    S_op_nslabs--;
}

/* the op at o's address through its slab's own pointer, or o if no slab holds it */
static OP *
S_op_view(OP *o)
{
    const UV a = __builtin_capstone_cap_get_cursor(o);
    const size_t i = S_op_slab_pos(a);
    OPSLAB *slab;
    if (i == 0 || a >= S_op_slabs[i - 1].hi)
	return o;
    slab = S_op_slabs[i - 1].slab;
    return (OP *)((char *)slab + (a - __builtin_capstone_cap_get_cursor(slab)));
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
