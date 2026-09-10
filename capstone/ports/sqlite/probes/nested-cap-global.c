/*
 * Minimal SQLite blocker shape: CapstoneCapGlobalInit does not recurse into an
 * array of structs, so both capability fields load without tags at runtime.
 * Expected today: a capability fault before the result value can be returned.
 */
typedef unsigned (*probe_fn)(void);

struct probe_entry {
  probe_fn function;
  const char *name;
};

static unsigned probe_value(void) { return 7; }

static const struct probe_entry entries[] = {
    {probe_value, "ok"},
};

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  *res = entries[0].function() + (unsigned)entries[0].name[0];
}
