// CONTROL for capstone-tighten-range-diagnostic.c and for the recorded-
// unreachable entries in capstone/tests/lit-coverage-unreachable.txt: the
// selector's "TIGHTEN immediate non-constant" and "CCSRRW CSR non-constant"
// fatal routes cannot be reached from C, because both builtins declare the
// operand `_Constant` and Sema rejects a non-constant argument before any
// IR exists.  (The IR-level route is closed by the intrinsic's immarg -- see
// unreachable-fatal-routes.ll.)
//
// MUTATION: replace `n` with `7` -> the expected diagnostic is no longer
// produced and -verify fails the RUN line (performed 2026-09-04).  (Prose in a
// -verify test must not spell the directive token; clang reads every line.)
//
// RUN: %clang_cc1 -triple capstone64-unknown-elf -ffreestanding -fsyntax-only -verify %s

void *tighten_nonconst(void *p, unsigned long n) {
  return __builtin_capstone_cap_tighten(p, n); // expected-error {{argument to '__builtin_capstone_cap_tighten' must be a constant integer}}
}

void *ccsrrw_nonconst(void *p, unsigned long csr) {
  return __builtin_capstone_cap_ccsrrw(p, csr); // expected-error {{argument to '__builtin_capstone_cap_ccsrrw' must be a constant integer}}
}
