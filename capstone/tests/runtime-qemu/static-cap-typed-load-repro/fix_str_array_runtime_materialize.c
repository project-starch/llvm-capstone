/*
 * Positive case proving the constructor-codegen architecture for the array
 * shape: instead of relying on a statically-initialized capability array
 * surviving in the image (it does not -- see fail_str_array_load.c), initialize
 * each element at runtime with an ordinary C store.  The compiler materializes
 * each string-literal address as a properly bounded, tagged capability and
 * `scc`-stores it into the array slot in place, so subsequent normal loads of
 * `gTable[i]` and dereferences succeed.
 *
 * This is exactly the code a per-module "initialize my capability globals"
 * constructor would emit automatically; here it is written by hand to prove the
 * runtime semantics (in-place tagged store + later tagged load) end-to-end.
 *
 * The array itself is non-const (writable .data), matching dtoa's `char *nums[]`
 * (elements are `const char *` only to avoid a writable-string-literal warning).
 *
 * Expected result: 'o'(111) + 'h'(104) + 'y'(121) = 336.
 */
static const char *gTable[3];

static void materialize_table(void) {
  gTable[0] = "ok";
  gTable[1] = "hi";
  gTable[2] = "yo";
}

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  materialize_table();
  /* Read back through the array at runtime (the previously-faulting pattern). */
  const char *const volatile *p = gTable;
  *res = (unsigned)p[0][0] + (unsigned)p[1][0] + (unsigned)p[2][0];
}
