// The scalar-crypto builtins the Capstone target inherited from the RISCV copy
// carry the same immediate ranges SemaRISCV enforces (aes32*/sm4* byte select
// 0..3, aes64ks1i round 0..10).  Until SemaCapstone existed (2026-09-05) they
// were unchecked on capstone64 and a bad immediate reached the backend.
//
// MUTATION: change `4` in aes32dsi_4 to `3` -> its expected diagnostic is no
// longer produced and -verify fails the RUN line (performed 2026-09-05).
//
// RUN: %clang_cc1 -triple capstone64-unknown-elf -target-feature +zknd -target-feature +zkne -target-feature +zksed -ffreestanding -fsyntax-only -verify %s

unsigned aes32dsi_3(unsigned a, unsigned b) { return __builtin_capstone_aes32dsi(a, b, 3); }
unsigned aes32dsi_4(unsigned a, unsigned b) { return __builtin_capstone_aes32dsi(a, b, 4); } // expected-error {{argument value 4 is outside the valid range [0, 3]}}
unsigned sm4ed_4(unsigned a, unsigned b) { return __builtin_capstone_sm4ed(a, b, 4); } // expected-error {{argument value 4 is outside the valid range [0, 3]}}
unsigned long aes64ks1i_10(unsigned long a) { return __builtin_capstone_aes64ks1i(a, 10); }
unsigned long aes64ks1i_11(unsigned long a) { return __builtin_capstone_aes64ks1i(a, 11); } // expected-error {{argument value 11 is outside the valid range [0, 10]}}
