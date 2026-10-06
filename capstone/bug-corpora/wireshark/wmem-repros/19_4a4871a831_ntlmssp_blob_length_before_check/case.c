#include "corpus.h"

WM_CASE(19) {
/* Row 19 -- NTLMSSP blob, fix 4a4871a831 ("ntlmssp: swap bounds check and
 * length for memcpy"). SPATIAL, and a different error from every other row
 * here: the metadata is written before the allocation it describes is known to
 * have happened.
 *
 * dissect_ntlmssp_blob at the pin (:983-989) does:
 *
 *     result->length = blob_length;
 *     if (blob_length < MAX_BLOB_SIZE)
 *     {
 *       result->contents = (guint8 *)wmem_alloc(wmem_file_scope(), blob_length);
 *       tvb_memcpy(tvb, result->contents, blob_offset, blob_length);
 *
 * `length` is assigned UNCONDITIONALLY, before the size check. A blob at or
 * above MAX_BLOB_SIZE therefore leaves the struct claiming a large length while
 * `contents` was never allocated for it. A later consumer trusts the length:
 *
 *     if (conv_ntlmssp_info->ntlm_response.length > 24)
 *       memcpy(..., conv_ntlmssp_info->ntlm_response.contents + 32, 8);
 *
 * -- and reads 8 bytes at offset 32 of a buffer that is either stale from an
 * earlier, smaller blob or was never sized for it. The fix moves
 * `result->length = blob_length` INSIDE the check, so the two can no longer
 * disagree.
 *
 * THE SHAPE IS "METADATA OVERSTATES THE ALLOCATION", and it is the only
 * instance of it in this corpus. The crossing is not an index arithmetic error;
 * it is a consumer correctly trusting a field that was written too early. A
 * bound on the allocation catches it only because the consumer reaches past
 * what was actually allocated -- which is what makes it spatial rather than a
 * plain logic bug.
 *
 * NESTED: the blob is wmem FILE scope, a chunk the BLOCK allocator carved from
 * a block g_malloc handed out, so the read leaves its chunk and stays inside
 * the block. */
#define MAX_BLOB_SIZE 256
  /* An earlier, SMALLER blob is what `contents` is left pointing at -- the
   * reachable case, since the struct is per-conversation and reused. */
  const unsigned small = 24;
  unsigned char *contents = wmem_alloc(wm_file_scope(), small);
  CHECK(contents, 1);
  memset(contents, 0x11, small);
  /* The next chunk of the same block: where a read at offset 32 lands. */
  unsigned char *successor = wmem_alloc(wm_file_scope(), 64);
  CHECK(successor, 2);
  CHECK((uintptr_t)successor > (uintptr_t)contents, 3);
  memset(successor, 0x5a, 64);

  /* The oversized blob arrives. At the pin `length` is written first. */
  const unsigned blob_length = MAX_BLOB_SIZE + 16; /* at or above the limit */
  unsigned recorded_length = blob_length;          /* :983, unconditional */
  if (blob_length < MAX_BLOB_SIZE) {
    /* not taken: contents is NOT reallocated, and keeps the small buffer */
    CHECK(0, 4);
  }
  /* Both halves of the claim, asserted rather than assumed: the recorded length
   * exceeds what was allocated, and the consumer's threshold admits it. */
  CHECK(recorded_length > small, 5);
  CHECK(recorded_length > 24, 6);
  /* And the read at offset 32 is past the 24-byte chunk, yet inside the block. */
  CHECK(32 >= small, 7);
  CHECK((uintptr_t)contents + 32 + 8 <= (uintptr_t)successor + 64, 8);

  /* The consumer at :1649, which trusts the length. */
  wm_held = contents + 32;
  wm_mark();
  (void)wm_probe(wm_held);
}
