// RUN: not %clang_cc1 -triple capstone64-unknown-elf -target-feature +m -std=c23 \
// RUN:     -O2 -S -o /dev/null %s 2>&1 | FileCheck %s

// THE BACKEND'S 128-BIT LIMITATION MUST READ AS A COMPILER ERROR, NOT A CRASH.
//
// This is a CLANG test on purpose. Every other test of this limitation runs llc on
// hand-written IR, and llc was always clean -- one "LLVM ERROR:" line, exit 1. Through
// clang it was not: report_fatal_error(GenCrashDiag=false) still let the driver print
// "PLEASE submit a bug report" plus a full stack dump, so the suite could not observe
// what users actually saw. It is now a DiagnosticInfoUnsupported, reported at the source
// location, and codegen completes.
//
// THE ROUTE HERE IS A 128-BIT ATOMIC, and that choice matters. __int128 and
// _BitInt(65..128) are rejected in the front end (see Sema/capstone-no-int128.c), but a
// 16-byte _Atomic reaches the same i128 machine type with no front-end guard. Three
// separate source routes to this point have now been found and each time the assumption
// "nothing in source C reaches this any more" was recorded it turned out to be wrong.
// So the diagnostic is the thing under test, not the reachability.

#include <stdatomic.h>

typedef struct { unsigned long a, b; } P;
_Atomic P g;

P load_it(void) { return atomic_load(&g); }

// It must be a diagnostic at a source location...
// CHECK: error: Capstone PureCap: cannot lower a 128-bit right shift by >= XLen
// ...that tells the user what to do instead...
// CHECK-SAME: use two 64-bit halves instead
// ...and it must NOT look like a compiler crash.
// CHECK-NOT: PLEASE submit a bug report
// CHECK-NOT: Stack dump
