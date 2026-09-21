/* Case 18: gh-151695 -- use-after-free of the curses screen encoding
 *
 * Shape: dangling pointer parked in a global
 * Consumer: Modules/_cursesmodule.c
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

PYC_CASE(18) {
  /* gh-151695 -- the curses screen encoding. A MODULE-LEVEL static pointed
   * into the encoding string owned by the window object initscr() returned.
   * The window is an ordinary object and can be deallocated while
   * module-level functions -- unctrl(), ungetch() -- keep reading through
   * the static.
   *
   * The dangling pointer outlives every frame here. Case 14's lived in
   * another object; this one lives in a global, so nothing in the program's
   * structure bounds when it is next used. */
  unsigned char *window = pym_malloc(OBJ);
  CHECK(window, 741);
  memset(window, 227, OBJ);
  held = window + 32;        /* curses_screen_encoding, into ->encoding */
  pym_free(window);          /* the window object was deallocated */
  unsigned char *fresh = pym_malloc(OBJ);
  CHECK(fresh == window, 742);
  memset(fresh, 229, OBJ);
  mark(18);
  (void)read_probe(held);    /* unctrl(), through the module-level static */
}
