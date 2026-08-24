// RUN: %clang_cc1 -triple capstone64-unknown-elf -std=c23 -fsyntax-only -verify %s

// __int128 IS AVAILABLE ON capstone64. The file keeps its "no-int128" name for the
// history that references it; what it pins is the opposite of what it used to.
//
// It was refused, deliberately, because MVT::i128 was the machine type this backend
// used for CAPABILITIES. There was no capability MVT to hold the two apart, so a
// genuine 128-bit integer and a capability were told apart by heuristics that
// DEFAULTED TO CAPABILITY. A source-level __int128 matched none of them and was
// compiled AS a capability, silently:
//
//     unsigned __int128 a + b  ->  cincoffset a0, a1, a0   (a cursor increment)
//     unsigned __int128 a & b  ->  lcc a0,a0,2 / lcc a1,a1,2 / and
//
// `lcc rd, rs, 2` reads field 2, the CURSOR, so the high 64 bits were never in the
// computation. The note on hasInt128Type said lifting the restriction meant giving
// capabilities their own MVT, as CHERI-LLVM does. That happened on 2026-08-24: a
// capability is MVT::c128 in register class GPCR, i128 is an ordinary illegal type,
// and the generic legalizer expands it into exactly the code upstream riscv64 emits.
//
// THIS TEST STAYS AT THE CLANG LEVEL ON PURPOSE. Every other i128 test on this target
// runs llc on hand-written IR, which is not the path a user takes, so the whole suite
// was blind to the report that prompted the original restriction. The arithmetic is
// checked in CodeGen/capstone-int128.c; this file checks only that the front end
// accepts the types.


__int128 a;
unsigned __int128 b;
__int128_t c;
__uint128_t d;

// _BitInt was capped at XLen for the same reason and is uncapped for the same reason:
// it is a different type that reached the same machine type, so rejecting __int128
// alone never closed it.
unsigned _BitInt(64) ok64;
unsigned _BitInt(65) ok65;
unsigned _BitInt(128) ok128;
_BitInt(128) ok128s;

// The two-uint64_t workaround must keep compiling -- real code uses it, and it is
// still the portable way to write this.
struct u128 { unsigned long lo, hi; };
struct u128 add128(struct u128 x, struct u128 y) {
  struct u128 r;
  r.lo = x.lo + y.lo;
  r.hi = x.hi + y.hi + (r.lo < x.lo);
  return r;
}

// AND THE THING THIS MUST NOT BREAK: uintptr_t round-trips. These reach the backend as
// ptrtoint/inttoptr on ptr addrspace(200), not as a source-level __int128, so
// capability arithmetic is untouched.
//
// The two warnings are expected and are NOT about this: on capstone64 a pointer is a
// 128-bit capability while __UINTPTR_TYPE__ is the 64-bit ADDRESS, so the round-trip
// narrows to the address -- which is what the backend emits it as, and it costs one
// `andi`. Pinned here so a change in that shape shows up.
void *align_down(void *p) {
  // expected-warning@+2 {{cast to smaller integer type 'unsigned long' from 'void *'}}
  // expected-warning@+1 {{cast to 'void *' from smaller integer type 'unsigned long'}}
  return (void *)((__UINTPTR_TYPE__)p & ~(__UINTPTR_TYPE__)31);
}
long ptr_diff(char *x, char *y) { return x - y; }
