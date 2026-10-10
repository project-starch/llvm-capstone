#include "corpus.h"

/* Row 22 -- USB HID report descriptors, #16818, fix c702b44a01 (a fix reversal: the fix is in the
 * 4.6.8 pin, as cases 16 and 17's are).
 *
 * parse_report_descriptor (packet-usb-hid.c, c702b44a01^ :3294) builds the descriptor's fields in
 * the FILE scope -- wmem_file_scope(), a BLOCK allocator. Each main item appends the current
 * `field` to fields_in or fields_out by value, and its `usages` member is a wmem_array_t*. After
 * an INPUT item the parser starts a fresh usages array (:3334); after an OUTPUT item, before the
 * fix, it did not (:3341-3350), so every OUTPUT item appended the SAME array. The error path
 * (:3464-3470) frees each field's usages:
 *
 *     for (j = 0; j < count(fields_out); j++)
 *         wmem_free(scope, ((hid_field_t*) wmem_array_index(rdesc->fields_out, j))->usages);
 *
 * so a descriptor with an INPUT item, two OUTPUT items and any malformed item after them frees one
 * chunk twice into the block allocator. BLOCK keeps its recycler list IN the freed chunks
 * (wmem_allocator_block.c, a wmem_block_free_t written over the data), and re-adding a chunk that
 * is already on that circular list relinks it to itself: the other chunk freed before it -- the
 * INPUT field's array -- drops out of the list and is never handed out again. An S2 double free
 * that corrupts the allocator's in-band free list (S4), with nothing reaching g_free.
 *
 * The consumer is reduced to the allocator calls the defect makes, in its order: each
 * wmem_array_new is the array header's allocation and its one-element buffer's. The defective
 * access is the second wmem_free itself, so wm_mark() is the last thing before it; a protected arm
 * that revokes the chunk at its first free must stop at the handback. Under wm_observe (the native
 * fix differential) the next descriptor's arrays are allocated: with the fix the INPUT field's
 * freed array is reissued at once; with the defect it never is -- the corrupted list made visible.
 * (The driver's verdict line words every defect as "reached storage outside its object"; for this
 * case DEFECT-REPRODUCED means that observation, the lost chunk, as case.json says.) */

struct usb_usages { /* wmem_array_t's members, so the header chunk has its size on every target */
  void *allocator;
  unsigned char *buf;
  unsigned long elem_size;
  unsigned elem_count, alloc_count;
  int null_terminated;
};

static struct usb_usages *usages_new(wmem_allocator_t *scope) { /* wmem_array_new(scope, 4) */
  struct usb_usages *a = wmem_alloc0(scope, sizeof *a);
  CHECK(a, 1);
  a->allocator = scope;
  a->elem_size = 4;
  a->alloc_count = 1;
  a->buf = wmem_alloc(scope, 4);
  CHECK(a->buf, 2);
  return a;
}

WM_CASE(22) {
  wmem_allocator_t *scope = wm_file_scope();         /* :3301 */
  struct usb_usages *fields_in[1], *fields_out[2];
  struct usb_usages *usages = usages_new(scope);     /* :3305, field.usages */
  /* INPUT item (:3325): appended to fields_in, and a fresh usages array started (:3334) */
  fields_in[0] = usages;
  usages = usages_new(scope);
  /* OUTPUT item 1 (:3341): wmem_array_append_one(rdesc->fields_out, field) */
  fields_out[0] = usages;
  if (wm_fixed) /* THE FIX, c702b44a01: field.usages = wmem_array_new(scope, ...) after OUTPUT */
    usages = usages_new(scope);
  /* OUTPUT item 2 */
  fields_out[1] = usages;
  if (wm_fixed)
    usages = usages_new(scope);
  /* A malformed item (`goto err`, :3345 and the like): the error path frees each field's usages. */
  wmem_free(scope, fields_in[0]);                    /* :3466 */
  wmem_free(scope, fields_out[0]);                   /* :3469, j = 0 */
  wm_mark();
  wmem_free(scope, fields_out[1]);                   /* :3469, j = 1: the same chunk again before the fix */
  if (wm_observe) {
    /* The next descriptor's parse takes its arrays from the same scope (:3305-3307). The recycler
     * hands its freed chunks back first, so the INPUT field's array comes back at once -- unless
     * the second free cut it out of the list. */
    void *next[4];
    int reissued = 0;
    for (int i = 0; i < 4; i++) {
      next[i] = wmem_alloc0(scope, sizeof(struct usb_usages));
      CHECK(next[i], 3);
      reissued |= next[i] == (void *)fields_in[0];
    }
    wm_defect = !reissued; /* a freed chunk lost from the free list */
  }
}
