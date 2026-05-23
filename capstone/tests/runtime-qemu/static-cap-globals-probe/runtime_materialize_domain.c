// Positive POC for runtime-side materialization into writable global storage.
// The capability-bearing object is not relied on as a ready-to-use static const
// image object. Instead, the domain populates a writable global object at
// runtime and then uses it.

struct pair {
  int (*fn)(void);
  const char *name;
};

static struct pair g_pair;

static int helper(void) { return 0x12340000u; }

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  g_pair.fn = helper;
  g_pair.name = "ok";
  *res = (unsigned)(g_pair.fn() + (unsigned)g_pair.name[0]);
}

