#include "corpus.h"

WM_CASE(14) {
/* Row 14 -- SolarEdge, fix 1d8acb21ab ("SolarEdge: Fix buffer overflow").
 * SPATIAL, and LIVE at our v4.6.8 pin.
 *
 * solaredge_decrypt carves TWO consecutive chunks of payload_length from the
 * packet scope and then, for i < payload_length, reads
 * intermediate_decrypted_payload[i + 6]. The last iteration therefore reads
 * index payload_length + 5: six bytes past the chunk, into the storage the
 * same block handed out next. The fix adds the guard
 * "length < SOLAREDGE_ENCRYPTION_KEY_LENGTH + 6" and then "payload_length -= 6",
 * exactly compensating the +6; the pin has neither.
 *
 * Quoted from v4.6.8:epan/dissectors/packet-solaredge.c:1005-1029. `scratch` is
 * pinfo->pool at the :1292 call site. */
  enum { PAYLOAD = 16 }; /* upstream's payload_length, any value works */
  /* :1007 -- the first of the two chunks. The overread lands in a successor,
   * so this one is allocated first to match upstream's order, and the one the
   * read crosses into is allocated after. */
  unsigned char *payload = wmem_alloc(wm_packet, PAYLOAD);
  CHECK(payload, 1);
  memset(payload, 0x11, PAYLOAD);
  /* :1008 -- the chunk the cursor leaves. */
  unsigned char *inter = wmem_alloc(wm_packet, PAYLOAD);
  CHECK(inter, 2);
  memset(inter, 0x22, PAYLOAD);
  /* The storage the +6 read crosses into: the next chunk of the same block,
   * filled with a pattern so the read has something to find. Its position is
   * asserted BEFORE the marker, so the marker's presence is evidence that the
   * crossing was created and not merely written down. */
  unsigned char *successor = wmem_alloc(wm_packet, 64);
  CHECK(successor, 3);
  CHECK((uintptr_t)successor > (uintptr_t)inter, 4);
  memset(successor, 0x5a, 64);
  /* :1029 -- out[i] = intermediate_decrypted_payload[i + 6] ^ ... for
   * i < payload_length. The last read is at index PAYLOAD - 1 + 6. */
  CHECK(PAYLOAD + 5 >= PAYLOAD, 5);                      /* past the chunk */
  CHECK((uintptr_t)inter + PAYLOAD + 5
        < (uintptr_t)successor + 64, 6);                 /* inside the block */
  wm_held = inter + (PAYLOAD - 1) + 6;
  wm_mark();
  (void)wm_probe(wm_held);
}
