/* Case 7: gh-148395 -- possible UAF in {LZMA,BZ2,_Zlib}Decompressor
 *
 * Shape: cursor surviving in a struct field across two API calls
 * Consumer: Modules/_bz2module.c, _lzmamodule.c, zlibmodule.c
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

PYC_CASE(7) {
  /* gh-148395 -- {LZMA,BZ2,_Zlib}Decompressor. On the error path the stream
   * struct keeps next_in pointing into the caller's input buffer, which the
   * caller then releases. The stale pointer therefore survives BETWEEN two
   * API calls, in a long-lived struct field, rather than inside one
   * operation -- and the next decompress() resumes from it. */
  struct stream {
    unsigned char *next_in;
    unsigned long avail_in;
  };
  struct stream *d = pym_malloc(sizeof *d);
  unsigned char *input = pym_malloc(OBJ);
  CHECK(d && input, 716);
  memset(input, 83, OBJ);
  d->next_in = input;        /* set by the failed decompress() */
  d->avail_in = OBJ;
  held = d->next_in;
  pym_free(input);           /* the caller released the input buffer */
  unsigned char *fresh = pym_malloc(OBJ);
  CHECK(fresh == input, 717);
  memset(fresh, 89, OBJ);
  mark(7);
  (void)read_probe(held);    /* the NEXT decompress() resumes from next_in */
}
