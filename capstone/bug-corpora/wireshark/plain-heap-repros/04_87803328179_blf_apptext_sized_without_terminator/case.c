#include "corpus.h"

/* wiretap/blf.c, blf_read_apptextmessage, fix 87803328179. The crossed buffer is ONE direct g_malloc-family
 * allocation; no wmem allocator is in the path. */

WSH_CASE(4) {
  /* Case 4 -- blf's APP_TEXT payload, fix 87803328179. PLAIN HEAP: the buffer is
   * sized to the file's declared textLength and then filled entirely from the
   * file, so nothing terminates it.
   *
   * At the fix's parent:
   *
   *     gchar *text = g_try_malloc0((gsize)apptextheader.textLength);
   *     if (!blf_read_bytes(params, ..., text, apptextheader.textLength, ...)) {
   *     ...
   *     gchar **tokens = g_strsplit_set(text, ";", -1);
   *
   * and the fix adds the byte and writes the terminator:
   *
   *     gchar *text = g_try_malloc((gsize)apptextheader.textLength + 1);
   *     ...
   *     text[apptextheader.textLength] = '\0';
   *
   * g_try_malloc0 zero-fills, so the buffer looks terminated until blf_read_bytes
   * overwrites every byte with file content.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long textLength = 16;            /* a size class */
  const unsigned long cap = fixed ? textLength + 1 : textLength;
  unsigned char *text = calloc((size_t)cap, 1);   /* g_try_malloc0 */
  CHECK(text, 1201);
  /* blf_read_bytes fills ALL textLength bytes from the file, with no zero byte. */
  for (unsigned long i = 0; i < textLength; i++)
    text[i] = (unsigned char)('a' + (i % 26));
  CHECK(text[textLength - 1] != 0, 1202);         /* the premise, asserted */
  if (fixed)
    text[textLength] = 0;

  /* g_strsplit_set's scan. */
  unsigned long touched = 0;
  unsigned acc = 0;
  for (unsigned long i = 0; ; i++) {
    touched = i;
    if (i >= cap) {
      acc += read_probe(text + i);                /* the labelled crossing */
      break;
    }
    if (text[i] == 0)
      break;
    acc += text[i];
  }
  (void)acc;

  o->cap = cap;
  o->touched = touched;
  o->crossed = touched >= cap;
  o->extent = 1;
  o->damage = o->crossed;
  o->defect_text = "the APP_TEXT buffer was sized to the file's textLength and then filled "
                   "completely from the file, so g_strsplit_set's scan runs past it";
  o->fixed_text = "the fix allocates textLength + 1 and writes the terminator";
  free(text);
}
