struct holder {
  int (*fn)(void);
};

static int helper(void) { return 0x12345678u; }

static const struct holder kHolder = {
  helper,
};

void domain_main(unsigned *res, unsigned func) {
  const volatile struct holder *p = &kHolder;
  (void)func;
  *res = (unsigned)p->fn();
}

