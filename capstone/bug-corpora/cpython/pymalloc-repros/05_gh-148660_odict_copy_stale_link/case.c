/* Case 5: gh-148660 -- use-after-free in OrderedDict.copy() on reentrant mutation
 *
 * Shape: pointer load out of a freed block, then followed
 * Consumer: Objects/odictobject.c
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

PYC_CASE(5) {
  /* gh-148660 -- OrderedDict.copy() under re-entrant mutation. The copy loop
   * is _odict_FOREACH, which advances by reading node->next OUT OF the node
   * it just processed. Re-entrant mutation frees that node, so the loop
   * reads its link field from freed memory and then follows it.
   *
   * This is the one case whose stale access is a POINTER load rather than a
   * byte read: the block is reused by another node, so in the spatial arm the
   * link reads back as a valid pointer and the walk continues -- silently
   * visiting the wrong node.
   *
   * ORDERING, AND WHAT MAY NOT BE CLAIMED FROM IT. The labelled byte read
   * below runs FIRST and is what faults in a protected arm; the pointer load
   * after it never executes. So a protected run of case 5 shows the stale
   * ACCESS refused, not the stale LINK being followed and refused -- the
   * wrong-node walk is only ever observed in the unprotected arm. Do not
   * reorder these two to make the fault land on the pointer load: the byte
   * probe is the labelled instruction the oracle checks. */
  struct node *first = pym_malloc(sizeof *first);
  struct node *second = pym_malloc(sizeof *second);
  CHECK(first && second && first != second, 711);
  second->next = NULL;
  second->payload[0] = 61;
  first->next = second;
  first->payload[0] = 67;
  held = (volatile unsigned char *)first; /* the loop's cursor */
  pym_free(first);                        /* the re-entrant mutation */
  struct node *fresh = pym_malloc(sizeof *fresh);
  CHECK(fresh == first, 712);
  fresh->next = second;      /* the block is now a different node */
  fresh->payload[0] = 71;
  mark(5);
  (void)read_probe(held);    /* _odict_FOREACH reads node->next */
  struct node *next = ((struct node *volatile *)held)[0];
  CHECK(next == second && next->payload[0] == 61, 713);
}
