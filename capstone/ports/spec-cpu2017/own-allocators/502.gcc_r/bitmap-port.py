#!/usr/bin/env python3
"""bitmap-port.py DIR: gcc's bitmap free lists (502.gcc_r bitmap.c) as lifetimes (nested lane, 2026-11-20). For a tree
that already has the collaborator's port (obstack and GC objects are lifetimes): bitmap elements and bitmap heads come
from an obstack or the GC heap, and gcc keeps freed ones on free lists of its own for reuse. Each element or head put on
a free list is renewed (CRENEW), so every reference the client kept is dead, and the list holds the renewed one:
  - bitmap_elem_to_freelist (one element);
  - bitmap_elt_clear_from, which upstream moves a whole chain onto the free list in one operation: each element of the
    chain is renewed and the chain relinked through the renewed references (the walk is the cost of the protection);
  - bitmap_obstack_free (a bitmap head onto the heads list).
Every edit is asserted to match exactly once."""
import sys
from pathlib import Path
D = Path(sys.argv[1])
def edit(old, new):
    p = D / 'bitmap.c'; s = p.read_text()
    assert s.count(old) == 1, (old[:70], s.count(old))
    p.write_text(s.replace(old, new))

edit('#include "bitmap.h"\n', '#include "bitmap.h"\n#include "sublet-port.h"\n')
edit("""  bitmap_obstack *bit_obstack = head->obstack;

  elt->next = NULL;
  if (bit_obstack)""", """  bitmap_obstack *bit_obstack = head->obstack;

  /* Sublet: the client's references to ELT end; the free list keeps the renewed one.  */
  elt = (bitmap_element *) sublet_renew (elt);
  elt->next = NULL;
  if (bit_obstack)""")
edit("""  /* Put the entire list onto the free list in one operation. */
  if (bit_obstack)""", """  /* Sublet: end every element's lifetime, relinking the chain through the renewed
     references; then the whole chain goes onto the free list as upstream does.  */
  {
    bitmap_element *first = NULL, *last = NULL, *e = elt;
    while (e)
      {
	bitmap_element *following = e->next;
	bitmap_element *renewed = (bitmap_element *) sublet_renew (e);
	if (last)
	  last->next = renewed;
	else
	  first = renewed;
	last = renewed;
	e = following;
      }
    elt = first;
  }

  /* Put the entire list onto the free list in one operation. */
  if (bit_obstack)""")
edit("""      bitmap_clear (map);
      map->first = (bitmap_element *) map->obstack->heads;""", """      bitmap_clear (map);
      /* Sublet: the client's references to the head end; the heads list keeps the renewed one.  */
      map = (bitmap) sublet_renew (map);
      map->first = (bitmap_element *) map->obstack->heads;""")
print('bitmap free lists ported')
