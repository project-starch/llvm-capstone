/*
 * Reduced failing case: an *array* of capability string pointers (the shape of
 * BEEBS dtoa's `char *nums[]`).  The file-scope statically-initialized array
 * stores capability values that carry no tag in the static image, so a runtime
 * load of an element followed by a dereference faults.
 *
 * Expected failure: `[CAPSTONE] Cap mem access requires capability`.
 */
static const char *kTable[3] = {"ok", "hi", "yo"};

void domain_main(unsigned *res, unsigned func) {
  const char *const volatile *p = kTable; /* force a runtime load through the array */
  (void)func;
  /* p[0] loads an (untagged) capability from the array slot; dereferencing it
     to read the first character traps. */
  *res = (unsigned)p[0][0];
}
