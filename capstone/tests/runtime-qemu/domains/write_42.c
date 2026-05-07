// The domain entry/exit ABI is provided by capstone/my_first_domain/start.S.
void domain_main(unsigned *res, unsigned func) {
  (void)func;
  *res = 42;
}

