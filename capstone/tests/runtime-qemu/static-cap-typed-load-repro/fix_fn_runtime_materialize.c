struct holder {
  int (*fn)(void);
};

static int helper(void) { return 0x12345678u; }

static struct holder gHolder;

static void materialize_holder(void) { gHolder.fn = helper; }

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  materialize_holder();
  *res = (unsigned)gHolder.fn();
}

