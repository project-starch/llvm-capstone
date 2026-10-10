/* Case 0: ddee3e2 -- "Fix minor severity heap buffer overflow reading
 * `--auth-file`". authfile_load sizes its buffer to the file's exact length and
 * then lets fgets write a terminating NUL into it, one byte past the end.
 *
 * Shape: a line that exactly fills the allocation makes fgets's NUL land one
 *        byte past calloc(1, sb.st_size)
 * Consumer: authfile.c, authfile_load()
 * SPATIAL, and NOT NESTED -- the object is one calloc from the system
 * allocator, so the crossing leaves the malloc bound itself. That is what makes
 * it the inventory's not-nested spatial row rather than another slab case.
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

/* authfile.c:16 at ddee3e2^ and unchanged at the 1.6.45 pin. */
#define MAX_ENTRY_LEN 256

MCH_CASE(0) {
  o->defect_text = "fgets wrote its terminating NUL at offset sb.st_size of a "
                   "calloc(1, sb.st_size), one byte past the allocation";
  o->fixed_text = "calloc(1, sb.st_size + 1) leaves room for the terminator";

  /* One credential line, the shape authfile_load parses. No trailing newline:
   * this is a file whose last line runs to EOF, which is the reachable case --
   * the loop's own "// EOF" break at authfile.c:83 exists for it. */
  static const char line[] = "user:pass";
  const unsigned long st_size = (unsigned long)(sizeof line - 1); /* 9, no NUL */
  CHECK(st_size > 0 && st_size < MAX_ENTRY_LEN, 800);

  /* THE DEFECT AND THE FIX, and nothing else differs between the arms.
   *   ddee3e2^  auth_data = calloc(1, sb.st_size);
   *   ddee3e2   auth_data = calloc(1, sb.st_size + 1);
   * The fix's other two hunks -- an auth_end clamp on the fgets length and a
   * '\0' break in the scan -- are what keep LATER lines inside; this case is
   * reduced to the first crossing, which the +1 alone removes. */
  const unsigned long cap = fixed ? st_size + 1 : st_size;
  unsigned char *auth_data = calloc(1, cap);
  CHECK(auth_data, 801);
  o->cap = cap;

  /* fgets(auth_cur, MAX_ENTRY_LEN, pwfile) at authfile.c:50, reduced to the
   * write C requires of it: for a line shorter than the size argument and ending
   * at EOF, fgets stores the line's bytes and then a terminating NUL. The call
   * is reduced rather than made so the crossing sits on this corpus's labelled
   * probe instead of inside libc; PROVENANCE.md states the equivalence. Nothing
   * about WHERE the NUL lands is reduced -- that is the defect. */
  unsigned char *auth_cur = auth_data;
  memcpy(auth_cur, line, (size_t)st_size);
  const unsigned long nul_at = st_size;    /* fgets's terminator offset */
  o->touched = nul_at;

  /* The whole claim, asserted rather than assumed: in the buggy arm the
   * terminator's address IS the allocation's end, and in the fixed arm it is the
   * last byte inside. If this ever stops holding, the case is wrong, not the
   * allocator. */
  CHECK(nul_at == (fixed ? cap - 1 : cap), 802);
  o->crossed = !fixed;

  write_probe(auth_cur + nul_at, 0x00);

  /* The consequence the upstream report describes is the overflow itself: one
   * byte of heap outside the allocation is set to zero. There is no neighbour
   * sentinel here on purpose -- two callocs are not adjacent, and asserting that
   * they were would be an assumption the allocator never makes. The oracle that
   * SEES the byte is the sanitiser arm; this arm proves the address. */
  o->damage = o->crossed;

  free(auth_data);
}
