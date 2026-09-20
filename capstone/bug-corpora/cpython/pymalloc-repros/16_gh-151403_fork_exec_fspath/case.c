/* Case 16: gh-151403 -- use-after-free when an argv item's __fspath__ mutates args
 *
 * Shape: free / reuse / stale read
 * Consumer: Modules/_posixsubprocess.c
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

PYC_CASE(16) {
  /* gh-151403 -- subprocess fork_exec. borrowed_arg comes from fast_args and
   * is handed to PyUnicode_FSConverter, whose __fspath__ can mutate args and
   * drop the sequence's last reference to it. */
  void **fast_args = pym_malloc(OBJ);
  unsigned char *arg = pym_malloc(OBJ);
  CHECK(fast_args && arg, 737);
  arg[0] = 197;
  fast_args[0] = arg;        /* PySequence_Fast_GET_ITEM, borrowed */
  pym_free(arg);             /* __fspath__ mutated args */
  unsigned char *fresh = pym_malloc(OBJ);
  CHECK(fresh == arg, 738);
  fresh[0] = 199;
  mark(16);
  (void)read_probe(fast_args[0]);
}
