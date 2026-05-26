struct holder {
  const char *name;
};

static const struct holder kHolder = {
  "ok",
};

void domain_main(unsigned *res, unsigned func) {
  const volatile struct holder *p = &kHolder;
  (void)func;
  *res = (unsigned)p->name[0];
}

