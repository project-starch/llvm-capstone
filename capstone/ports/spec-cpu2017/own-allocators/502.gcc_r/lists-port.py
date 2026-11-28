#!/usr/bin/env python3
"""lists-port.py DIR: gcc's EXPR_LIST and INSN_LIST free lists (502.gcc_r lists.c) as lifetimes (nested lane,
2026-11-20). For a tree with the collaborator's port (GC objects are lifetimes): list nodes are rtx from the GC heap,
and lists.c keeps freed ones on unused_expr_list and unused_insn_list for reuse. Each node put on a free list is renewed
(CRENEW), so every reference the client kept is dead, and the list holds the renewed one; free_list, which upstream
splices a whole list in one step after a walk that only checks codes, renews and relinks each node in that walk.
Every edit is asserted to match exactly once."""
import sys
from pathlib import Path
D = Path(sys.argv[1])
def edit(old, new):
    p = D / 'lists.c'; s = p.read_text()
    assert s.count(old) == 1, (old[:70], s.count(old))
    p.write_text(s.replace(old, new))

s = (D / 'lists.c').read_text()
inc = '#include "ggc.h"\n'
assert s.count(inc) == 1, s.count(inc)
edit(inc, inc + '#include "sublet-port.h"\n')
edit("""  rtx link, prev_link;

  prev_link = *listp;
  link = XEXP (prev_link, 1);

  gcc_assert (unused_listp != &unused_insn_list
	      || GET_CODE (prev_link) == INSN_LIST);

  while (link)
    {
      gcc_assert (unused_listp != &unused_insn_list
		  || GET_CODE (prev_link) == INSN_LIST);

      prev_link = link;
      link = XEXP (link, 1);
    }

  XEXP (prev_link, 1) = *unused_listp;
  *unused_listp = *listp;
  *listp = 0;""", """  rtx link, first = 0, last = 0;

  /* Sublet: end every node's lifetime and relink the list through the
     renewed references, in the walk upstream makes to check the codes.  */
  for (link = *listp; link; )
    {
      rtx following, renewed;

      gcc_assert (unused_listp != &unused_insn_list
		  || GET_CODE (link) == INSN_LIST);
      following = XEXP (link, 1);
      renewed = (rtx) sublet_renew (link);
      if (last)
	XEXP (last, 1) = renewed;
      else
	first = renewed;
      last = renewed;
      link = following;
    }

  XEXP (last, 1) = *unused_listp;
  *unused_listp = first;
  *listp = 0;""")
edit("""free_EXPR_LIST_node (rtx ptr)
{
  XEXP (ptr, 1) = unused_expr_list;""", """free_EXPR_LIST_node (rtx ptr)
{
  ptr = (rtx) sublet_renew (ptr);	/* Sublet: the client's references end.  */
  XEXP (ptr, 1) = unused_expr_list;""")
edit("""  gcc_assert (GET_CODE (ptr) == INSN_LIST);
  XEXP (ptr, 1) = unused_insn_list;""", """  gcc_assert (GET_CODE (ptr) == INSN_LIST);
  ptr = (rtx) sublet_renew (ptr);	/* Sublet: the client's references end.  */
  XEXP (ptr, 1) = unused_insn_list;""")
print('lists.c free lists ported')
