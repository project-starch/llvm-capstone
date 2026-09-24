// -Wcapstone-pointer-roundtrip (since 2026-09-05, default on for Capstone):
// an integer cannot carry a capability's tag. The backend rebuilds a pointer
// computed from ONE pointer in the same function (CapstoneRecoverProvenance),
// so `(T *)(uintptr_t)p` in one expression is not diagnosed any more. What it
// cannot rebuild is a uintptr_t that came from memory or from a caller, and a
// cast from a value of that type is what the warning names -- saying which
// case is the safe one, since the front end cannot see where the value came
// from. A pointer made from a plain integer is the programmer's business and
// stays silent.
//
// MUTATION: change `uintptr_t x` in @via_uintptr to `unsigned long x` -> the
// first expected diagnostic is no longer produced and -verify fails the RUN
// line (the typedef spelling is the whole trigger).
//
// The size warnings (-Wpointer-to-int-cast, -Wint-to-pointer-cast: a 128-bit
// pointer does not fit a 64-bit integer) fire on the same lines and are about
// width, not provenance; they are silenced here so the test is about this one.
//
// RUN: %clang_cc1 -triple capstone64-unknown-elf -ffreestanding -fsyntax-only -Wno-pointer-to-int-cast -Wno-int-to-pointer-cast -verify %s
// RUN: %clang_cc1 -triple capstone64-unknown-elf -ffreestanding -fsyntax-only -Wno-pointer-to-int-cast -Wno-int-to-pointer-cast -Wno-capstone-pointer-roundtrip -verify=off %s
// RUN: %clang_cc1 -triple riscv64-unknown-elf -ffreestanding -fsyntax-only -verify=off %s
// off-no-diagnostics

typedef unsigned long uintptr_t;
typedef long intptr_t;
typedef uintptr_t addr_t;

char *via_uintptr(uintptr_t x) { return (char *)x; } // expected-warning {{casting a value of type 'uintptr_t' (aka 'unsigned long') to 'char *' restores capability provenance on Capstone only if the value was computed from a single pointer in this function; one read from memory or passed in is untagged}}
char *via_intptr(intptr_t x) { return (char *)x; } // expected-warning {{casting a value of type 'intptr_t' (aka 'long') to 'char *'}}
char *via_nested_typedef(addr_t x) { return (char *)x; } // expected-warning {{casting a value of type 'addr_t' (aka 'unsigned long') to 'char *'}}

char *same_expr(char *p) { return (char *)(unsigned long)p; }
char *same_expr_uintptr(char *p) { return (char *)(uintptr_t)p; }
char *same_expr_arith(char *p) { return (char *)(((uintptr_t)p + 15) & ~(uintptr_t)15); }
char *two_pointers(char *p, char *q) { return (char *)((uintptr_t)p ^ (uintptr_t)q); } // expected-warning {{casting a value of type 'uintptr_t' (aka 'unsigned long') to 'char *'}}
struct node { uintptr_t link; };
char *from_memory(struct node *n) { return (char *)(n->link & ~(uintptr_t)1); } // expected-warning {{casting a value of type 'uintptr_t' (aka 'unsigned long') to 'char *'}}
char *plain_integer(unsigned long n) { return (char *)n; }
char *null_constant(void) { return (char *)0; }
char *pointer_to_pointer(char *p) { return (char *)(void *)p; }
unsigned long to_integer(char *p) { return (unsigned long)p; }
