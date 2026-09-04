// -Wcapstone-pointer-roundtrip (since 2026-09-05, default on for Capstone):
// an integer cannot carry a capability's tag, so a pointer made from an
// integer that came from a pointer is UNTAGGED and faults when dereferenced.
// The C standard only promises that uintptr_t, if provided, round-trips a
// pointer's value; on this target the value survives and the provenance does
// not, at every -O level once every ptrtoint is an integer write.  The warning
// names the two shapes that spell the round trip; a pointer made from a plain
// integer is the programmer's business and stays silent.
//
// MUTATION: change `(unsigned long)p` in @same_expr to `(unsigned long)0` ->
// the first expected diagnostic is no longer produced and -verify fails the RUN
// line (performed 2026-09-05, after the cycle-2 build).
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

char *same_expr(char *p) { return (char *)(unsigned long)p; } // expected-warning {{casting an integer that was itself converted from a pointer to 'char *' does not restore capability provenance on Capstone}}
char *via_uintptr(uintptr_t x) { return (char *)x; } // expected-warning {{casting a value of type 'uintptr_t' (aka 'unsigned long') to 'char *' does not restore capability provenance}}
char *via_intptr(intptr_t x) { return (char *)x; } // expected-warning {{casting a value of type 'intptr_t' (aka 'long') to 'char *'}}
char *via_nested_typedef(addr_t x) { return (char *)x; } // expected-warning {{casting a value of type 'addr_t' (aka 'unsigned long') to 'char *'}}

char *plain_integer(unsigned long n) { return (char *)n; }
char *null_constant(void) { return (char *)0; }
char *pointer_to_pointer(char *p) { return (char *)(void *)p; }
unsigned long to_integer(char *p) { return (unsigned long)p; }
