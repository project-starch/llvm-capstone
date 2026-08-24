// RUN: %clang_cc1 -triple capstone64-unknown-elf -target-feature +m -std=c23 \
// RUN:     -O2 -S -o - %s 2>&1 | FileCheck %s

// THE 128-BIT LIMITATION IS GONE. This file used to require a diagnostic here;
// it now requires that there is nothing to diagnose.
//
// The limitation was never about 128-bit arithmetic as such. i128 was the
// CAPABILITY CARRIER, so the backend had to lower i128 operations itself, and
// the ones it could not lower -- a right shift by >= XLen among them -- had to be
// reported. A capability is c128 now; i128 is an ordinary illegal type that the
// generic legalizer expands exactly as it does on any other RV64 target.
//
// The file's own warning is why this is a test and not a deletion: three separate
// source routes to the old limitation were found over time, and each time the
// assumption "nothing in source C reaches this any more" was recorded, it turned
// out to be wrong. So the routes are pinned here, compiling.
//
// __int128 and _BitInt(65..128) are still rejected in the front end -- that is a
// separate policy with its own test, Sema/capstone-no-int128.c -- so the routes
// that remain are the ones that reach a 16-byte machine type without asking for
// a 128-bit integer by name.

#include <stdatomic.h>

typedef struct { unsigned long a, b; } P;
_Atomic P g;

// A 16-byte _Atomic. This was the route the diagnostic fired on.
// CHECK-LABEL: load_it:
// CHECK: __atomic_load_16
P load_it(void) { return atomic_load(&g); }

// CHECK-LABEL: xchg_it:
// CHECK: __atomic_exchange_16
P xchg_it(P n) { return atomic_exchange(&g, n); }

// A 16-byte struct by value, which is two registers and no libcall.
// CHECK-LABEL: by_value:
// CHECK: xor
typedef struct { unsigned long a, b; } S16;
S16 by_value(S16 s) { s.a ^= s.b; return s; }

// It must not look like a compiler crash, and it must not be a diagnostic.
// CHECK-NOT: error:
// CHECK-NOT: PLEASE submit a bug report
// CHECK-NOT: Stack dump
