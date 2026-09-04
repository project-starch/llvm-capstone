// The __builtin_capstone_* immediates that Sema checks (SemaCapstone.cpp, since
// 2026-09-05).  Before it existed, an out-of-range TIGHTEN immediate reached the
// backend and the compiler died with a fatal error and a stack dump; a wrong
// CCSR id reached QEMU, which asserts (an emulator abort) on any id it does not
// know.  Each accepted value below is a control for the rejected one beside it.
//
// The ranges come from the implementations, not the encoding fields: TIGHTEN's
// immediate is a 3-bit permission mask (the RTL raises for imm > 7); the CCSR
// ids are the set QEMU switches on (ctvec 0, cih 1, cepc 2, cscratch 4, cpmp
// 16..31; 3 is reserved); SHRINK's constant base must be below its end.
//
// MUTATION: change `8` in tighten_8 to `7` -> its expected diagnostic is no
// longer produced and -verify fails the RUN line (performed 2026-09-05).  (Prose
// in a -verify test must not spell the directive token.)
//
// RUN: %clang_cc1 -triple capstone64-unknown-elf -ffreestanding -fsyntax-only -verify %s

void *tighten_0(void *p) { return __builtin_capstone_cap_tighten(p, 0); }
void *tighten_7(void *p) { return __builtin_capstone_cap_tighten(p, 7); }
void *tighten_8(void *p) { return __builtin_capstone_cap_tighten(p, 8); } // expected-error {{argument value 8 is outside the valid range [0, 7]}}
void *tighten_999(void *p) { return __builtin_capstone_cap_tighten(p, 999); } // expected-error {{argument value 999 is outside the valid range [0, 7]}}

void *ccsr_ctvec(void *p) { return __builtin_capstone_cap_ccsrrw(p, 0); }
void *ccsr_cih(void *p) { return __builtin_capstone_cap_ccsrrw(p, 1); }
void *ccsr_cepc(void *p) { return __builtin_capstone_cap_ccsrrw(p, 2); }
void *ccsr_cscratch(void *p) { return __builtin_capstone_cap_ccsrrw(p, 4); }
void *ccsr_cpmp0(void *p) { return __builtin_capstone_cap_ccsrrw(p, 16); }
void *ccsr_cpmp15(void *p) { return __builtin_capstone_cap_ccsrrw(p, 31); }
void *ccsr_reserved(void *p) { return __builtin_capstone_cap_ccsrrw(p, 3); } // expected-error {{capability CSR id 3 is not one of ctvec (0), cih (1), cepc (2), cscratch (4) or a cpmp entry (16-31)}}
void *ccsr_gap(void *p) { return __builtin_capstone_cap_ccsrrw(p, 5); } // expected-error {{capability CSR id 5 is not one of}}
void *ccsr_past(void *p) { return __builtin_capstone_cap_ccsrrw(p, 32); } // expected-error {{capability CSR id 32 is not one of}}

void *shrink_ok(void *p) { return __builtin_capstone_cap_shrink(p, 8, 16); }
void *shrink_var(void *p, unsigned long a, unsigned long b) { return __builtin_capstone_cap_shrink(p, a, b); }
void *shrink_empty(void *p) { return __builtin_capstone_cap_shrink(p, 16, 16); } // expected-error {{'__builtin_capstone_cap_shrink' base 16 must be below its end 16}}
void *shrink_inverted(void *p) { return __builtin_capstone_cap_shrink(p, 32, 8); } // expected-error {{'__builtin_capstone_cap_shrink' base 32 must be below its end 8}}
