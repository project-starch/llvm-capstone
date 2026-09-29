// Phase 0, THE OPEN QUESTION: does sublet_handle() twice on ONE node give two
// INDEPENDENTLY revocable parents, or does the second stand above the first?
//
// hier_sibling_conn_survives_ok already shows that siblings scope correctly when they
// are independent SPLITs off one arena -- disjoint ranges. What no probe asks is
// whether two parents of the SAME node are independent, and that is the question
// mruby's fourth nesting level turns on: two RStrings share one buffer, and
// mrb_str_modify's un-share has to revoke ONE of them while the other keeps the bytes.
// A split cannot express it, because a split partitions. A second handle might.
//
//   sublet_store(region, grant)     -- the granted region, LINEAR
//   sublet_handle(region, h1)       -- first parent of the node
//   sublet_handle(region, h2)       -- second parent of the SAME node
//   alias = sublet_take(region)     -- the bytes; region keeps a handle
//   write/read alias                -- live
//   sublet_give(h2)                 -- revoke the SECOND parent only
//   read alias                      -- survives?
//
// Expected: UNKNOWN, which is why the probe exists. If the read survives, siblings
// over one range are expressible and mruby's level 4 is buildable as written. If it
// faults, h2 is senior to the alias's derivation and level 4 needs either
// copy-on-share or an instruction the set does not have.
//
// Read the result with hier_two_handles_no_give_ok as the control: it takes the same
// two handles and revokes NEITHER, so a fault there would mean the two handles broke
// the node rather than that the revoke reached the alias.
#include "hier_probe_domain.h"
#include "../../../sublet/sublet.h"

#define HIER_RET_TWO_HANDLES_SURVIVED 0x08740000u

void domain_main(void *arg, unsigned func) {
  if (hier_probe_receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  sublet_cap region, h1, h2;
  sublet_store(&region, hier_probe_grant);
  if (sublet_type(&region) != SUBLET_TYPE_LIN) {
    *res = HIER_RET_TWO_HANDLES_SURVIVED | 0xFFu; /* no linear grant: nothing measured */
    return;
  }

  sublet_handle(&region, &h1); /* first parent */
  sublet_handle(&region, &h2); /* second parent of the same node */

  volatile char *p = (volatile char *)sublet_take(&region);
  p[0] = (char)HIER_PROBE_SENTINEL_A;
  volatile char live = p[0];
  (void)live;

  sublet_give(&h2); /* revoke ONLY the second parent */

  volatile char after = p[0]; /* does the alias survive h2's revoke? */
  *res = HIER_RET_TWO_HANDLES_SURVIVED | (unsigned char)after;
}
