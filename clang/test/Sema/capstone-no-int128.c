// RUN: %clang_cc1 -triple capstone64-unknown-elf -std=c23 -fsyntax-only -verify %s

// __int128 is REJECTED on capstone64, deliberately. MVT::i128 is the machine type this
// backend uses for CAPABILITIES, and there is no capability MVT to hold the two apart, so
// the backend separates a genuine 128-bit integer from a capability with heuristics that
// DEFAULT TO CAPABILITY. A source-level __int128 matches none of them and was therefore
// compiled AS A CAPABILITY -- silently:
//
//     unsigned __int128 a + b  ->  cincoffset a0, a1, a0      (cursor increment)
//     unsigned __int128 a & b  ->  lcc a0,a0,2 / lcc a1,a1,2 / and
//
// `lcc rd, rs, 2` reads field 2, the CURSOR, so the high 64 bits were never in the
// computation. See CapstoneTargetInfo::hasInt128Type in clang/lib/Basic/Targets/Capstone.h.
//
// THIS TEST EXISTS AT THE CLANG LEVEL ON PURPOSE. Every other i128 test on this target runs
// llc on hand-written IR, which is not the path a user takes -- so the whole suite was blind
// to the report that prompted this. A test that cannot observe the reported symptom does not
// cover it.

__int128 a;                    // expected-error {{__int128 is not supported on this target}}
unsigned __int128 b;           // expected-error {{__int128 is not supported on this target}}
// The __int128_t / __uint128_t typedefs are not predeclared at all once the type is off,
// so these are "unknown type name" rather than the __int128 diagnostic.
__int128_t c;                  // expected-error {{unknown type name '__int128_t'}}
__uint128_t d;                 // expected-error {{unknown type name '__uint128_t'}}

// _BitInt IS A SEPARATE HOLE AND MUST BE CAPPED TOO. Rejecting __int128 alone does NOT close
// it: _BitInt is a different type that reaches the same i128 machine type. Measured with
// __int128 already rejected, _BitInt(65) and _BitInt(128) still reached the backend and died
// there. The boundary is exactly XLen -- at or below it the value lives in an XLen register,
// above it it is widened into the 128-bit capability class. See getMaxBitIntWidth().
unsigned _BitInt(64) ok64;     // no diagnostic: fits in XLen, lowers to a plain `add`
unsigned _BitInt(65) bad65;    // expected-error {{unsigned _BitInt of bit sizes greater than 64 not supported}}
unsigned _BitInt(128) bad128;  // expected-error {{unsigned _BitInt of bit sizes greater than 64 not supported}}
_BitInt(128) bad128s;          // expected-error {{signed _BitInt of bit sizes greater than 64 not supported}}

// The workaround, and it must keep compiling: two 64-bit halves never enter a capability
// register, so no heuristic can misread them.
struct u128 { unsigned long lo, hi; };
struct u128 add128(struct u128 x, struct u128 y) {
  struct u128 r;
  r.lo = x.lo + y.lo;
  r.hi = x.hi + y.hi + (r.lo < x.lo);
  return r;
}

// AND THE THING THIS MUST NOT BREAK: uintptr_t round-trips. These reach the backend as
// ptrtoint/inttoptr on ptr addrspace(200), not as a source-level __int128, so capability
// arithmetic is untouched. If this ever starts ERRORING, the restriction has been applied
// too widely and real code (MicroPython's gc_init, pairheap.c) breaks with it.
//
// The two warnings are expected and are NOT caused by this change: on capstone64 a pointer
// is a 128-bit capability while __UINTPTR_TYPE__ is 64-bit, so the round-trip narrows to the
// cursor -- which is exactly what the backend emits it as (`lcc rd, rs, 2`). Pinned here so
// that if the restriction is ever widened, the failure shows up as an ERROR appearing rather
// than as these warnings quietly changing shape.
void *align_down(void *p) {
  // expected-warning@+2 {{cast to smaller integer type 'unsigned long' from 'void *'}}
  // expected-warning@+1 {{cast to 'void *' from smaller integer type 'unsigned long'}}
  return (void *)((__UINTPTR_TYPE__)p & ~(__UINTPTR_TYPE__)31);
}
long ptr_diff(char *x, char *y) { return x - y; }
