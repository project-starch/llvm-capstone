// 128-bit atomics at every optimisation level.  An _Atomic __int128 object's
// load, store and += become the size-generic __atomic_load/__atomic_store and
// __atomic_fetch_add_16; the __atomic_* builtins on a plain __int128 become
// the _16 libcalls; nothing becomes an lr/sc or amo sequence, and nothing is a
// backend diagnostic or a crash.  The 16-byte _Atomic STRUCT and the by-value
// 16-byte struct are inherited from the retired
// capstone-i128-backend-diagnostic.c: they were the source routes to the old
// i128-carrier limitation and stay pinned compiling.  clang warns once per
// 16-byte atomic that it is not lock-free; that is pinned as well.
// Measured 2026-09-04 on the branch tools.
//
// MUTATION: add `#error probe` -> "error: probe" is printed and the NOERR-NOT
// on `error:` fires (performed 2026-09-04).
//
// RUN: %clang_cc1 -triple capstone64-unknown-elf -target-feature +a -target-feature +m -std=c23 -O0 -S -o - %s 2>/dev/null | FileCheck %s --implicit-check-not=lr.d --implicit-check-not=sc.d --implicit-check-not=amoadd --implicit-check-not=amoswap --implicit-check-not=amocas
// RUN: %clang_cc1 -triple capstone64-unknown-elf -target-feature +a -target-feature +m -std=c23 -O1 -S -o - %s 2>/dev/null | FileCheck %s --implicit-check-not=lr.d --implicit-check-not=sc.d --implicit-check-not=amoadd --implicit-check-not=amoswap --implicit-check-not=amocas
// RUN: %clang_cc1 -triple capstone64-unknown-elf -target-feature +a -target-feature +m -std=c23 -O2 -S -o - %s 2>/dev/null | FileCheck %s --implicit-check-not=lr.d --implicit-check-not=sc.d --implicit-check-not=amoadd --implicit-check-not=amoswap --implicit-check-not=amocas
// RUN: %clang_cc1 -triple capstone64-unknown-elf -target-feature +a -target-feature +m -std=c23 -O2 -S -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=NOERR

#include <stdatomic.h>

_Atomic __int128 g;
__int128 h;

// CHECK-LABEL: ld:
// CHECK: %pcrel_hi(__atomic_load)
__int128 ld(void) { return g; }

// CHECK-LABEL: st:
// CHECK: %pcrel_hi(__atomic_store)
void st(__int128 v) { g = v; }

// CHECK-LABEL: adda:
// CHECK: %pcrel_hi(__atomic_fetch_add_16)
__int128 adda(__int128 v) { return g += v; }

// CHECK-LABEL: xchg:
// CHECK: %pcrel_hi(__atomic_exchange_16)
__int128 xchg(__int128 v) { return __atomic_exchange_n(&h, v, __ATOMIC_SEQ_CST); }

// CHECK-LABEL: add:
// CHECK: %pcrel_hi(__atomic_fetch_add_16)
__int128 add(__int128 v) { return __atomic_fetch_add(&h, v, __ATOMIC_SEQ_CST); }

// CHECK-LABEL: cas:
// CHECK: %pcrel_hi(__atomic_compare_exchange_16)
int cas(__int128 *e, __int128 d) { return __atomic_compare_exchange_n(&h, e, d, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); }

typedef struct { unsigned long a, b; } P;
_Atomic P gp;

// CHECK-LABEL: load_it:
// CHECK: __atomic_load_16
P load_it(void) { return atomic_load(&gp); }

// CHECK-LABEL: xchg_it:
// CHECK: __atomic_exchange_16
P xchg_it(P n) { return atomic_exchange(&gp, n); }

// A 16-byte struct by value: two registers and no libcall.
// CHECK-LABEL: by_value:
// CHECK: xor
typedef struct { unsigned long a, b; } S16;
S16 by_value(S16 s) { s.a ^= s.b; return s; }

// NOERR: warning: large atomic operation may incur significant performance penalty
// NOERR-NOT: error:
// NOERR-NOT: PLEASE submit a bug report
// NOERR-NOT: Stack dump
