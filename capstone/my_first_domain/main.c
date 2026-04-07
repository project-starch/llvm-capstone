// The domain entry/exit ABI is handled in start.S.
// This file is intentionally ordinary C compiled by our clang.
void domain_main(unsigned *res, unsigned func) {
  (void)func;
  *res = 42;
}
