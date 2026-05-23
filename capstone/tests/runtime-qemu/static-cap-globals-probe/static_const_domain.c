// Reduced reproducer for the current static/global capability issue.
// The same logical values as the control case are first stored in a file-scope
// static const object, then loaded back at runtime and used.

struct pair {
  int (*fn)(void);
  const char *name;
};

static int helper(void) { return 0x12340000u; }

static const struct pair kPair = {
  helper,
  "ok",
};

void domain_main(unsigned *res, unsigned func) {
  const volatile struct pair *p = &kPair;
  (void)func;
  *res = (unsigned)(p->fn() + (unsigned)p->name[0]);
}

