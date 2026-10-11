#include "corpus.h"

/* capchild/capture_ifinfo.c, capture_interface_list, fix cfc15838bdec.
 * The object is ONE direct g_malloc-family allocation -- no wmem -- which is
 * what puts this row in the plain-temporal corpus rather than ../wmem-repros. */

WST_CASE(9) {
  /* Case 9 -- capture_interface_list's out-parameter, fix cfc15838bdec. The
   * freed object is the error message; what is left holding its address is the
   * GLOBAL global_capture_opts.ifaces_err_info, because one return path leaves
   * *err_str unassigned.
   *
   * The fix initialises the out-parameter unconditionally:
   *
   *     *err = 0;
   *     *err_str = NULL;
   */
  const unsigned long n = 48;
  unsigned char *primary_msg = malloc((size_t)n);
  CHECK(primary_msg, 891);
  memset(primary_msg, 0x11, (size_t)n);

  /* The global, and the caller's free of it just before the call. */
  volatile unsigned char *ifaces_err_info = primary_msg;
  free(primary_msg);
  o->freed = 1;

  /* The callee. The fix assigns at entry; the buggy arm's early return leaves
   * the global untouched on the path where extcap interfaces WERE found. */
  if (fixed)
    ifaces_err_info = NULL;                       /* *err_str = NULL at entry */

  unsigned char *fresh = malloc((size_t)n);
  CHECK(fresh, 892);
  memset(fresh, 0xAA, (size_t)n);

  o->bytes = n;
  o->marker = 0xAA;
  /* The NEXT refresh's free of the global. */
  o->observed = ifaces_err_info ? read_probe(ifaces_err_info) : 0u;
  o->aliased = ifaces_err_info && o->observed == o->marker;
  o->damage = o->aliased;
  o->defect_text = "the early return left *err_str unassigned, so the caller's global still named "
                   "the message it had just freed and the next refresh frees it again";
  o->fixed_text = "the fix assigns *err_str = NULL at entry, making the contract unconditional";
  free(fresh);
}
