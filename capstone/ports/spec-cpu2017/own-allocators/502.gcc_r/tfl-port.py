#!/usr/bin/env python3
"""tfl-port.py DIR: five more of gcc's private free lists (502.gcc_r) as lifetimes (nested lane, 2026-11-20). For a tree
with the collaborator's port (GC objects of less than 4 KiB are lifetimes). Each object put on one of these lists is
renewed (CRENEW) after the last read through the client's reference, so every reference the client kept is dead and
the list holds the renewed one:
  c-decl.c     free_binding_and_advance (c_binding), pop_scope (c_scope)
  emit-rtl.c   end_sequence (sequence_stack)
  gimple.c     gimple_seq_free (gimple_seq_d)
  tree-phinodes.c release_phi_node (phi nodes; renewed only below 4 KiB, where the GC heap gave them a lifetime of
               their own: renewing a larger one would renew its page's)
SSA names (tree-ssanames.c) are left out: gcc reads released names through old references by design
(SSA_NAME_IN_FREE_LIST), so a renewed name would fault on correct code. Every edit is asserted to match exactly once."""
import sys
from pathlib import Path
D = Path(sys.argv[1])
def edit(name, old, new):
    p = D / name; s = p.read_text()
    assert s.count(old) == 1, (name, old[:70], s.count(old))
    p.write_text(s.replace(old, new))
def include(name, after):
    edit(name, after, after + '#include "sublet-port.h"\n')

include('c-decl.c', '#include "ggc.h"\n')
edit('c-decl.c', """  struct c_binding *prev = b->prev;

  memset (b, 0, sizeof (struct c_binding));""", """  struct c_binding *prev = b->prev;

  b = (struct c_binding *) sublet_renew (b);	/* Sublet: the client's references end.  */
  memset (b, 0, sizeof (struct c_binding));""")
edit('c-decl.c', """  memset (scope, 0, sizeof (struct c_scope));
  scope->outer = scope_freelist;""", """  scope = (struct c_scope *) sublet_renew (scope);	/* Sublet: the client's references end.  */
  memset (scope, 0, sizeof (struct c_scope));
  scope->outer = scope_freelist;""")
include('emit-rtl.c', '#include "ggc.h"\n')
edit('emit-rtl.c', """  seq_stack = tem->next;

  memset (tem, 0, sizeof (*tem));""", """  seq_stack = tem->next;

  tem = (struct sequence_stack *) sublet_renew (tem);	/* Sublet: the client's references end.  */
  memset (tem, 0, sizeof (*tem));""")
include('gimple.c', '#include "ggc.h"\n')
edit('gimple.c', """  /* Add SEQ to the pool of free sequences.  */
  seq->next_free = gimple_seq_cache;""", """  /* Add SEQ to the pool of free sequences.  */
  seq = (gimple_seq) sublet_renew (seq);	/* Sublet: the client's references end.  */
  seq->next_free = gimple_seq_cache;""")
include('tree-phinodes.c', '#include "ggc.h"\n')
edit('tree-phinodes.c', """  bucket = len > NUM_BUCKETS - 1 ? NUM_BUCKETS - 1 : len;
  bucket -= 2;
  VEC_safe_push (gimple, gc, free_phinodes[bucket], phi);""", """  bucket = len > NUM_BUCKETS - 1 ? NUM_BUCKETS - 1 : len;
  bucket -= 2;
  /* Sublet: the client's references end; only a phi node the GC heap gave a
     lifetime of its own (below 4 KiB) is renewed.  */
  if (ggc_get_size (phi) < SUBLET_EXACT_LIMIT)
    phi = (gimple) sublet_renew (phi);
  VEC_safe_push (gimple, gc, free_phinodes[bucket], phi);""")
print('five free lists ported')
