extern int coremark_main(void);

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  *res = (unsigned)coremark_main();
}

