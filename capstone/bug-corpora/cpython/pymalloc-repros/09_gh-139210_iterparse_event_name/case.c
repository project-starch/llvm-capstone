/* Case 9: gh-139210 -- use-after-free in xml.etree.ElementTree.iterparse()
 *
 * Shape: payload buffer, error path
 * Consumer: Modules/_elementtree.c
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

PYC_CASE(9) {
  /* gh-139210 -- xml.etree.ElementTree.iterparse(). event_name is a C string
   * pointing INTO an item of events_seq. The pre-fix order drops the
   * sequence first and formats the message second, so PyErr_Format reads the
   * string out of memory the sequence took with it.
   *
   * The distinguishing feature is that nothing here is a PyObject* the
   * checker could have followed -- it is a char* into a payload, consumed by
   * a formatter on the error path. */
  unsigned char *item = pym_malloc(OBJ);   /* the event name string object */
  CHECK(item, 721);
  memset(item, 107, OBJ);
  held = item + 24;          /* event_name, into the string's payload */
  pym_free(item);            /* Py_DECREF(events_seq), before the format */
  unsigned char *fresh = pym_malloc(OBJ);
  CHECK(fresh == item, 722);
  memset(fresh, 109, OBJ);
  mark(9);
  (void)read_probe(held);    /* PyErr_Format("unknown event '%s'", ...) */
}
