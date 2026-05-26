struct holder {
  const char *name;
};

static struct holder gHolder;

static void materialize_holder(void) { gHolder.name = "ok"; }

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  materialize_holder();
  *res = (unsigned)gHolder.name[0];
}

