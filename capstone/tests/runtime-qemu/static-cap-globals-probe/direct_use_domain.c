// Control variant for the static capability globals diagnostic.
// This uses the same logical values as the failing case, but keeps them in
// direct code use rather than first storing them in a file-scope static object.

static int helper(void) { return 0x12340000u; }

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  *res = (unsigned)(helper() + (unsigned)"ok"[0]);
}

