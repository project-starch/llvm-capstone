#include "corpus.h"

/* wiretap/nettrace_3gpp_32_423.c, create_temp_pcapng_file, fix 140aad08e081. The crossed buffer is ONE direct g_malloc-family
 * allocation; no wmem allocator is in the path. */

WSH_CASE(5) {
  /* Case 5 -- nettrace's packet buffer, fix 140aad08e081. PLAIN HEAP: the
   * buffer is sized packet_size + 12 and all of it is written, then searched
   * with strstr.
   *
   * At the fix's parent:
   *
   *     packet_buf = (guint8 *)g_malloc(packet_size + 12);
   *     ...
   *     curr_pos = packet_buf + 12;
   *     curr_pos = strstr(curr_pos, "<fileHeader");
   *
   * and the fix adds a byte and terminates it:
   *
   *     packet_buf = (guint8 *)g_malloc(packet_size + 12+1);
   *     packet_buf[packet_size + 12] = 0;
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long packet_size = 4;
  const unsigned long header = 12;
  const unsigned long filled = packet_size + header;    /* 16 B: a size class */
  const unsigned long cap = fixed ? filled + 1 : filled;
  unsigned char *packet_buf = calloc((size_t)cap, 1);
  CHECK(packet_buf, 1211);
  for (unsigned long i = 0; i < filled; i++)
    packet_buf[i] = (unsigned char)('A' + (i % 26));     /* no zero byte, no match */
  if (fixed)
    packet_buf[filled] = 0;

  /* strstr's scan for "<fileHeader", which the content never matches. */
  unsigned long touched = 0;
  unsigned acc = 0;
  for (unsigned long i = header; ; i++) {
    touched = i;
    if (i >= cap) {
      acc += read_probe(packet_buf + i);          /* the labelled crossing */
      break;
    }
    if (packet_buf[i] == 0)
      break;
    acc += packet_buf[i];
  }
  (void)acc;

  o->cap = cap;
  o->touched = touched;
  o->crossed = touched >= cap;
  o->extent = 1;
  o->damage = o->crossed;
  o->defect_text = "the packet buffer was sized for its twelve header bytes plus the file's bytes "
                   "and written in full, so strstr's search runs past it";
  o->fixed_text = "the fix allocates one more byte and stores a terminator there";
  free(packet_buf);
}
