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
#endif
    return slab;
}""")

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
print('port-op-slabs: applied to', D)
