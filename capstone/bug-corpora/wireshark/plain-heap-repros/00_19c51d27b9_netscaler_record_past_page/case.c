/* Case 0: 19c51d27b9 -- "Don't go past the end of a page in a NetScaler file."
 * PACKET_DESCRIBE copies a record out of the page buffer using a length taken
 * straight from the file, with nothing relating that length to how much of the
 * page is left.
 *
 * Shape: a record size read from the file makes the copy run past the end of a
 *        g_malloc'd page buffer
 * Consumer: wiretap/netscaler.c, the PACKET_DESCRIBE macro
 * SPATIAL, and NOT NESTED -- the page buffer is one g_malloc from the system
 * allocator. wiretap does not use wmem, so no inner layer is involved; that is
 * what makes this the inventory's not-nested spatial row for tshark.
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

/* wiretap/netscaler.c:52 at 19c51d27b9^ and unchanged at the v4.6.8 pin. */
#define NSPR_PAGESIZE 8192

WSH_CASE(0) {
  o->defect_text = "a record size read from the file made the copy start inside "
                   "the page buffer and run past its end";
  o->fixed_text = "the added guard refuses a record that does not fit in what is "
                  "left of the page";

  /* The page buffer. g_malloc is malloc plus abort-on-failure; PROVENANCE.md
   * records that substitution. No slack, no capacity field, no container. */
  const unsigned long cap = NSPR_PAGESIZE;
  unsigned char *nstrace_buf = malloc(cap);
  CHECK(nstrace_buf, 850);
  memset(nstrace_buf, 0x5a, cap);
  o->cap = cap;

  /* Mid-page, where the reader is when it reaches the last record of a page. */
  const unsigned long nstrace_buf_offset = cap - 64;

  /* (phdr)->caplen = pletoh16(&pp->nsprRecordSize) -- netscaler.c:957 of the
   * parent. A 16-bit field straight out of the file, so 0..65535, and the
   * pre-fix macro relates it to nothing. Taken here at its maximum, which is
   * what a crafted file supplies. */
  const unsigned long caplen = 65535;

  /* THE DEFECT AND THE FIX. The fix adds three guards; this case is reduced to
   * the one that bounds the copy:
   *   if ((nstrace_buflen - nstrace_buf_offset) < (phdr)->caplen) { ... return FALSE; }
   * Everything else about the arms is identical. */
  const unsigned long room = cap - nstrace_buf_offset;
  const int refused = fixed && room < caplen;

  o->touched = nstrace_buf_offset + caplen;
  o->extent = (long)(o->touched - cap);
  CHECK(o->extent > 0, 851);

  if (refused) {
    o->crossed = 0;
    o->damage = 0;
    free(nstrace_buf);
    return;
  }

  /* memcpy(ws_buffer_start_ptr(wth->frame_buffer), type, (phdr)->caplen) with
   * type = &nstrace_buf[nstrace_buf_offset]. REDUCED TO THE FIRST CROSSING: the
   * unreduced copy spans o->extent bytes past the allocation -- about 57 KB --
   * and reading all of it would be an unbounded walk through whatever follows.
   * The first byte past is what makes the crossing, and it is what a sanitiser
   * reports; PROVENANCE.md states the full magnitude. */
  CHECK(nstrace_buf_offset + caplen > cap, 852);
  o->crossed = 1;
  unsigned char first_past = (unsigned char)read_probe(nstrace_buf + cap);
  o->damage = 1;                 /* the copy took a byte that is not the page's */
  (void)first_past;

  free(nstrace_buf);
}
