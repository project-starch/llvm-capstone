/*
 * Gap-2 reproducer: a LOCAL aggregate initialized from a constant template with
 * capability (function + string) pointers. >64 bytes so at -O2 clang would lower
 * it to a memcpy from a private untagged constant template (the SQLite
 * sqlite3RegisterBuiltinFunctions shape). The fix in CGDecl.cpp emits per-leaf
 * tagged stores instead. Expected (no fault): local[0].function() + 'a' = 104.
 */
typedef unsigned (*probe_fn)(void);
struct probe_entry { probe_fn function; const char *name; };
static unsigned probe_value(void) { return 7; }

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  volatile unsigned sel = 0;
  struct probe_entry local[] = {
      {probe_value, "ab"}, {probe_value, "cd"}, {probe_value, "ef"}};
  *res = local[sel].function() + (unsigned)local[sel].name[0];
}
