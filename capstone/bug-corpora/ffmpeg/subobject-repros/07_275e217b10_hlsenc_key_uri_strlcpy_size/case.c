#include "corpus.h"
#include <stdint.h>

/* VariantStream's four string members in declaration order, as hlsenc.c:174-178
 * has them at the pin. LINE_BUFFER_SIZE is MAX_URL_SIZE = 4096 (hlsenc.c:72,
 * internal.h:30) and KEYSIZE is 16 (hlsenc.c:71). The struct is ONE allocation,
 * so the bound between key_uri and key_string is not a bound any allocator
 * knows about.
 *
 * The buffers are REDUCED from 4097 to 64 bytes. The defect is that a
 * destination-size argument is given a source LENGTH, so what matters is that
 * the length can exceed the destination -- not the destination's absolute size.
 * The reduction keeps the overrun inside the allocation, so what is measured is
 * the sub-object crossing. It is stated here rather than left silent. */
#define LINE_BUF 64 /* reduced from LINE_BUFFER_SIZE + 1 == 4097 */
#define KEY_STR 33  /* KEYSIZE*2 + 1 */

struct variant_stream {
  char key_file[LINE_BUF];
  char key_uri[LINE_BUF];
  char key_string[KEY_STR];
  char iv_string[KEY_STR];
};

/* av_strlcpy, reduced to exactly what C requires of it and nothing more: at
 * most size-1 bytes of src, then a terminating NUL. This is the function whose
 * CONTRACT the defect misuses, so reducing it would hide the bug -- it is
 * reproduced faithfully instead. */
static size_t ff2_strlcpy(char *dst, const char *src, size_t size) {
  size_t len = strlen(src);
  if (size) {
    size_t n = len < size - 1 ? len : size - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
  }
  return len;
}

FF2_CASE(7) {
  /* Case 7 -- HLS playlist parser, fix 275e217b10. SUB-OBJECT: a WRITE of
   * attacker-chosen length from one struct member into the two after it, inside
   * a single allocation.
   *
   * parse_playlist at the pin (hlsenc.c:1203-1207) reads an #EXT-X-KEY line:
   *
   *     ptr += strlen("URI=\"");
   *     end = av_stristr(ptr, ",");
   *     if (end) {
   *         av_strlcpy(vs->key_uri, ptr, end - ptr);
   *
   * av_strlcpy's third parameter is the size of the DESTINATION. What is passed
   * is `end - ptr`, the length of the SOURCE. For a URI longer than key_uri the
   * two are not interchangeable: the call is licensed to write end-ptr bytes
   * into a 4097-byte member, and the playlist chooses end-ptr. The fix passes
   * FFMIN(end - ptr + 1, sizeof(vs->key_uri)), which restores the contract, and
   * it also switches the delimiter search from "," to '"' so that a URI
   * containing a comma is no longer truncated -- a separate correctness fix in
   * the same hunk, not part of this crossing.
   *
   * The two arms differ by exactly the size argument. The URI, its length, and
   * the destination are identical between them.
   *
   * NOTHING WE HAVE CAN CATCH THIS: the crossing is inside one allocation. */
  struct variant_stream *vs = av_refstruct_allocz(sizeof *vs);
  CHECK(vs, 651);
  /* The claim: the member after key_uri is key_string, directly adjacent. */
  CHECK((char *)&vs->key_uri[LINE_BUF] == (char *)&vs->key_string[0], 652);

  /* A URI that overruns key_uri by 16 bytes and so reaches into key_string.
   * This is a playlist's content, which is to say an input. */
  const size_t uri_len = LINE_BUF + 16;
  char *uri = av_refstruct_allocz(uri_len + 2);
  CHECK(uri, 653);
  memset(uri, 'A', uri_len);
  uri[uri_len] = ',';  /* the delimiter av_stristr finds */
  uri[uri_len + 1] = '\0';

  memset(vs->key_string, 0x5A, sizeof vs->key_string - 1);
  memset(vs->iv_string, 0x5A, sizeof vs->iv_string - 1);

  const char *ptr = uri;
  const char *end = strchr(uri, ',');
  CHECK(end, 654);
  size_t span = (size_t)(end - ptr); /* == uri_len, the SOURCE length */

  size_t size_arg = fixed ? (span + 1 < sizeof vs->key_uri ? span + 1
                                                           : sizeof vs->key_uri)
                          : span;
  ff2_strlcpy(vs->key_uri, ptr, size_arg);

  /* How far past key_uri the copy reached. */
  unsigned past = 0;
  for (unsigned i = 0; i < sizeof vs->key_string - 1; i++)
    if (vs->key_string[i] == 'A')
      past++;
  int terminated_inside = vs->key_uri[LINE_BUF - 1] == '\0';

  printf("cap=%zu source_len=%zu size_arg=%zu past=%u\n", sizeof vs->key_uri,
         span, size_arg, past);

  /* `past` counts the 'A' bytes landing in key_string. The copy writes
   * size_arg-1 source bytes plus a terminating NUL, so it reaches 16 bytes past
   * key_uri: 15 of them 'A', the 16th the NUL. The counter and the text are
   * stated in the same terms so neither can drift from the other. */
  FF2_VERDICT(!fixed && past == 15 && vs->key_string[15] == '\0',
              fixed && past == 0 && terminated_inside,
              "the source length was passed as av_strlcpy's destination size, so the "
              "copy ran 16 bytes past key_uri into key_string (15 'A' plus the "
              "terminator), inside one allocation",
              "the fix's FFMIN(end - ptr + 1, sizeof(key_uri)) keeps the copy inside its member");
  av_refstruct_unref(&uri);
  av_refstruct_unref(&vs);
  return !fixed ? !past : !!past;
}
