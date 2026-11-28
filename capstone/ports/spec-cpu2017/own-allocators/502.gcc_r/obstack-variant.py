#!/usr/bin/env python3
"""obstack-variant.py DIR VARIANT: refine the collaborator's obstack port in DIR (a ported 502.gcc_r tree: obstack.c,
include/obstack.h) without changing what it protects (nested lane, 2026-11-20). Every object keeps a lifetime of its
own; a free still ends every object it cuts off. VARIANT is a, b, ab or abc:
  a  the issue fast path inline: obstack_finish derives the object and stores its reference in the reference stack
     itself while the stack has room; only growth of the stack (and objects of 4 KiB or more) call out.
  b  a freed chunk's references are forgotten in O(1): each chunk keeps the reference-stack depth at its creation, and a
     free that drops chunks sets the stack back to the oldest dropped chunk's depth, instead of popping one by one.
  c  (with a and b) the reference stack in segments of 1,024 references that are never moved: growth adds a segment,
     where the flat stack's realloc copied every reference at each doubling.
Every edit is asserted to match exactly once."""
import sys
from pathlib import Path
D = Path(sys.argv[1]); V = sys.argv[2]
assert V in ('a', 'b', 'ab', 'abc'), V
def edit(name, old, new):
    p = D / name; s = p.read_text()
    assert s.count(old) == 1, (name, old[:70], s.count(old))
    p.write_text(s.replace(old, new))

if 'a' in V:
    edit('include/obstack.h', """extern void *_obstack_sublet_issue (struct obstack *, void *, unsigned long);
""", """extern void *_obstack_sublet_issue (struct obstack *, void *, unsigned long);
#include "sublet-port.h"
/* The issue fast path (obstack-variant.py a): derive the finished object and push its reference while the reference
   stack has room; grow the stack, or leave an object of 4 KiB or more with the chunk's lifetime, out of line.  */
static __inline__ void *
_obstack_sublet_issue_fast (struct obstack *h, void *object, unsigned long size)
{
  if (__builtin_expect (size < SUBLET_EXACT_LIMIT && h->sublet_live < h->sublet_room, 1))
    {
      void *child = sublet_derive (object, size);
      h->sublet_issued[h->sublet_live++] = child;
      return child;
    }
  return _obstack_sublet_issue (h, object, size);
}
""")
    edit('include/obstack.h', """      = (char *) _obstack_sublet_issue ((h), (h)->object_base,		\\""",
         """      = (char *) _obstack_sublet_issue_fast ((h), (h)->object_base,	\\""")

if 'b' in V:
    edit('include/obstack.h', """  struct _obstack_chunk *prev;	/* address of prior chunk or NULL */
  char	contents[4];		/* objects begin here */""", """  struct _obstack_chunk *prev;	/* address of prior chunk or NULL */
  unsigned long sublet_base;	/* Sublet: the reference stack's depth when this chunk began */
  unsigned long sublet_pad;	/* keeps contents at a 16-byte offset: objects start there unaligned otherwise */
  char	contents[4];		/* objects begin here */""")
    # the two _obstack_begin variants and _obstack_newchunk record the depth
    p = D / 'obstack.c'; s = p.read_text()
    old = "  chunk->prev = 0;\n"
    assert s.count(old) == 2, s.count(old)
    s = s.replace(old, "  chunk->prev = 0;\n  chunk->sublet_base = 0;\n")
    old = "  new_chunk->prev = old_chunk;\n"
    assert s.count(old) == 1
    s = s.replace(old, old + "  new_chunk->sublet_base = h->sublet_live;\n")
    # sublet_forget_chunk: O(1)
    old = """static void
sublet_forget_chunk (struct obstack *h, struct _obstack_chunk *chunk)
{
  unsigned long start = (unsigned long) chunk;
  unsigned long limit = (unsigned long) chunk->limit;

  while (h->sublet_live > 0)
    {
      unsigned long top = (unsigned long) h->sublet_issued[h->sublet_live - 1];

      if (top <= start || top >= limit)
	break;
      h->sublet_live--;
    }
}"""
    assert s.count(old) == 1
    s = s.replace(old, """static void
sublet_forget_chunk (struct obstack *h, struct _obstack_chunk *chunk)
{
  /* Every reference issued while CHUNK or a later chunk was current sits above its depth (obstack-variant.py b).  */
  if (chunk->sublet_base < h->sublet_live)
    h->sublet_live = chunk->sublet_base;
}""")
    p.write_text(s)
if 'c' in V:
    edit('include/obstack.h', "  void\t**sublet_issued;\t/* Sublet: the reference of each live object, oldest first */\n",
         "  void\t***sublet_issued;\t/* Sublet: the reference of each live object, oldest first, in segments */\n")
    edit('include/obstack.h', """#include "sublet-port.h"
""", """#include "sublet-port.h"
/* The reference stack in segments of 1 << SUBLET_SEG_SHIFT references (obstack-variant.py c). */
#define SUBLET_SEG_SHIFT 10
#define SUBLET_REF(h, i) ((h)->sublet_issued[(i) >> SUBLET_SEG_SHIFT][(i) & ((1u << SUBLET_SEG_SHIFT) - 1)])
""")
    edit('include/obstack.h', """      h->sublet_issued[h->sublet_live++] = child;
""", """      unsigned i = h->sublet_live;
      SUBLET_REF (h, i) = child;
      h->sublet_live = i + 1;
""")
    edit('obstack.c', """  if (h->sublet_live == h->sublet_room)
    {
      h->sublet_room = h->sublet_room ? 2 * h->sublet_room : 32;
      h->sublet_issued = (POINTER *) realloc (h->sublet_issued,
\t\t\t\t\t      h->sublet_room * sizeof (POINTER));
      if (!h->sublet_issued)
\t(*obstack_alloc_failed_handler) ();
    }
  h->sublet_issued[h->sublet_live++] = child;
""", """  if (h->sublet_live == h->sublet_room)
    {
      /* A new segment. The segment table holds pointers only, and doubles when the count is 0 or a power of two. */
      unsigned nseg = h->sublet_room >> SUBLET_SEG_SHIFT;
      if ((nseg & (nseg - 1)) == 0)
\t{
\t  h->sublet_issued = (void ***) realloc (h->sublet_issued, (nseg ? 2 * nseg : 1) * sizeof (void **));
\t  if (!h->sublet_issued)
\t    (*obstack_alloc_failed_handler) ();
\t}
      h->sublet_issued[nseg] = (void **) malloc ((1u << SUBLET_SEG_SHIFT) * sizeof (POINTER));
      if (!h->sublet_issued[nseg])
\t(*obstack_alloc_failed_handler) ();
      h->sublet_room += 1u << SUBLET_SEG_SHIFT;
    }
  SUBLET_REF (h, h->sublet_live) = child;
  h->sublet_live++;
""")
    edit('obstack.c', """      unsigned long top = (unsigned long) h->sublet_issued[h->sublet_live - 1];

      if (top < address || top >= limit)
\tbreak;
      sublet_release (h->sublet_issued[--h->sublet_live]);""", """      unsigned long top = (unsigned long) SUBLET_REF (h, h->sublet_live - 1);

      if (top < address || top >= limit)
\tbreak;
      h->sublet_live--;
      sublet_release (SUBLET_REF (h, h->sublet_live));""")
    edit('obstack.c', """  free (h->sublet_issued);
  h->sublet_issued = 0;""", """  for (unsigned i = 0; i < (h->sublet_room >> SUBLET_SEG_SHIFT); i++)
    free (h->sublet_issued[i]);
  free (h->sublet_issued);
  h->sublet_issued = 0;""")
print('variant', V, 'applied')
