#include "corpus.h"

/* extcap.c, interfaces_cb, fix 07ffcf90426b.
 * The object is ONE direct g_malloc-family allocation -- no wmem -- which is
 * what puts this row in the plain-temporal corpus rather than ../wmem-repros. */

WST_CASE(4) {
  /* Case 4 -- interfaces_cb's help string, fix 07ffcf90426b. One allocation was
   * stored into EVERY interface the loop produced, and each is later freed by
   * extcap_free_interface.
   *
   * At the fix's parent:
   *
   *     help = g_strdup(int_iter->help);
   *     ...
   *     int_iter->help = help;
   *
   * and the fix duplicates at the assignment instead:
   *
   *     help = int_iter->help;
   *     ...
   *     int_iter->help = g_strdup(help);
   */
  const unsigned long n = 48;
  unsigned char *source = malloc((size_t)n);      /* int_iter->help, the tool's text */
  CHECK(source, 841);
  memset(source, 0x11, (size_t)n);

  /* Two interfaces. The buggy arm gives both the SAME allocation. */
  unsigned char *help = fixed ? NULL : malloc((size_t)n);
  if (!fixed) {
    CHECK(help, 842);
    memcpy(help, source, (size_t)n);
  }
  unsigned char *owner_a, *owner_b;
  if (fixed) {
    owner_a = malloc((size_t)n); CHECK(owner_a, 843); memcpy(owner_a, source, (size_t)n);
    owner_b = malloc((size_t)n); CHECK(owner_b, 844); memcpy(owner_b, source, (size_t)n);
  } else {
    owner_a = help;
    owner_b = help;                               /* the same pointer in both structs */
  }

  free(owner_a);                                  /* extcap_free_interface, first interface */
  o->freed = 1;

  unsigned char *fresh = malloc((size_t)n);
  CHECK(fresh, 845);
  memset(fresh, 0xAA, (size_t)n);

  o->bytes = n;
  o->marker = 0xAA;
  o->observed = read_probe(owner_b);              /* the second interface's release */
  o->aliased = o->observed == o->marker;
  o->damage = o->aliased;
  o->defect_text = "one g_strdup'd help string was stored into every interface the loop produced, "
                   "so the second interface's release reaches storage that now belongs to another "
                   "object";
  o->fixed_text = "the fix duplicates the string per interface, so each owner has its own";
  free(fresh);
  if (fixed)
    free(owner_b);
  free(source);
}
