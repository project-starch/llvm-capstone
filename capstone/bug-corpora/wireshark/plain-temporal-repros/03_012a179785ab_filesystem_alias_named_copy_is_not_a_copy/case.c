#include "corpus.h"

/* wsutil/filesystem.c, create_persconffile_profile, fix 012a179785ab.
 * The object is ONE direct g_malloc-family allocation -- no wmem -- which is
 * what puts this row in the plain-temporal corpus rather than ../wmem-repros. */

WST_CASE(3) {
  /* Case 3 -- create_persconffile_profile's alias, fix 012a179785ab. The freed
   * object is the path string; the pointer left holding its address is
   * `pf_dir_path`, because `pf_dir_path_copy` was an ALIAS and not a copy.
   *
   * At the fix's parent:
   *
   *     pf_dir_path_copy = pf_dir_path;
   *     pf_dir_parent_path = get_dirname(pf_dir_path_copy);
   *     ...
   *     g_free(pf_dir_path_copy);
   *     ret = ws_mkdir(pf_dir_path, 0755);
   *
   * and the fix makes the copy real:
   *
   *     pf_dir_path_copy = g_strdup(pf_dir_path);
   */
  const unsigned long n = 48;
  unsigned char *pf_dir_path = malloc((size_t)n);
  CHECK(pf_dir_path, 831);
  memset(pf_dir_path, 0x11, (size_t)n);

  /* The "copy": an alias in the buggy arm, a real duplicate in the fixed one. */
  unsigned char *pf_dir_path_copy;
  if (fixed) {
    pf_dir_path_copy = malloc((size_t)n);         /* g_strdup */
    CHECK(pf_dir_path_copy, 832);
    memcpy(pf_dir_path_copy, pf_dir_path, (size_t)n);
  } else {
    pf_dir_path_copy = pf_dir_path;               /* the plain assignment */
  }

  free(pf_dir_path_copy);                         /* g_free(pf_dir_path_copy) */
  o->freed = 1;

  unsigned char *fresh = malloc((size_t)n);
  CHECK(fresh, 833);
  memset(fresh, 0xAA, (size_t)n);

  o->bytes = n;
  o->marker = 0xAA;
  o->observed = read_probe(pf_dir_path);          /* ws_mkdir's use of the path */
  o->aliased = !fixed && o->observed == o->marker;
  o->damage = o->aliased;
  o->defect_text = "the variable named _copy was a plain assignment, so freeing it destroyed the "
                   "buffer pf_dir_path still points at, and ws_mkdir reads it";
  o->fixed_text = "the fix makes the copy a real g_strdup, so freeing it leaves the original live";
  free(fresh);
  if (fixed)
    free(pf_dir_path);
}
