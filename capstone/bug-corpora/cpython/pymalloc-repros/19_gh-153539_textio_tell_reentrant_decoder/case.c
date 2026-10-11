/* Case 19: gh-153539 -- use-after-free in TextIOWrapper.tell() with a reentrant decoder
 *
 * Shape: free / reuse / stale read
 * Consumer: Modules/_io/textio.c
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

PYC_CASE(19) {
  /* gh-153539 -- TextIOWrapper.tell() with a re-entrant decoder. next_input
   * is the snapshot bytes object, borrowed; the decoder's getstate can run
   * Python that seeks the file and replaces the snapshot, dropping the last
   * reference while tell() is still measuring against it.
   *
   * SIZE. The snapshot is a bytes object holding buffered input, so a large
   * buffer puts it above pymalloc's 512-byte threshold and back within a
   * malloc-level tool's reach -- the same caveat as case 6 and case 10. The
   * driver pins the small size. */
  unsigned char *next_input = pym_malloc(OBJ);
  CHECK(next_input, 743);
  memset(next_input, 233, OBJ);
  held = next_input;         /* the borrowed snapshot */
  pym_free(next_input);      /* the re-entrant decoder seeked */
  unsigned char *fresh = pym_malloc(OBJ);
  CHECK(fresh == next_input, 744);
  memset(fresh, 239, OBJ);
  mark(19);
  (void)read_probe(held);    /* cookie.start_pos -= PyBytes_GET_SIZE(...) */
}
