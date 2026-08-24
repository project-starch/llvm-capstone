// `long double` is binary128 here, and it works.
//
// It did not: musl's survey (capstone/musl-capstone/survey-musl-capstone.py)
// carried src/math/fmodl.c as its CONTROL_MUST_FAIL, with the reason recorded as
// "long double lowers to a 128-bit shift, which this target cannot do while
// MVT::i128 is the capability carrier. It will keep failing until the
// capability-MVT work lands." It landed on 2026-08-24.
//
// The shape that failed is the one every musl long-double routine starts with:
// take the bit pattern apart through a union. That is a 128-bit shift, which had
// no lowering because a capability has no high half to shift.
//
// Arithmetic goes to the soft-float libcalls, which is what any RV64 target
// without quad-precision hardware does; it is not a Capstone property.
//
// The checks are per function rather than an implicit-check-not over the whole
// output: addl calls a libcall, so it legitimately has a frame and a global
// address, both of which are cincoffset.
//
// RUN: %clang_cc1 -triple capstone64-unknown-elf -target-feature +m -ffreestanding \
// RUN:   -O2 -mframe-pointer=none -S -o - %s | FileCheck %s

_Static_assert(sizeof(long double) == 16, "long double is binary128 here");

// musl's ldshape union, and the exponent field every fmodl/scalbnl/frexpl reads.
union ldshape { long double f; struct { unsigned long lo, hi; } i; };

// The 128-bit shift, reduced to two shifts of the high word.
// CHECK-LABEL: expo:
// CHECK:      %bb.0:
// CHECK-NEXT: slli a0, a1, 1
// CHECK-NEXT: srli a0, a0, 49
// CHECK-NEXT: cjalr zero, 0(ra)
int expo(long double x) { union ldshape u = {x}; return (u.i.hi >> 48) & 0x7fff; }

// The mantissa's low half is already in the first register. CHECK-NEXT off the
// basic-block marker is what pins that: the function is the return and nothing
// else, so there is no room for a shift, an lcc or a cincoffset.
// CHECK-LABEL: mant_lo:
// CHECK:      %bb.0:
// CHECK-NEXT: cjalr zero, 0(ra)
unsigned long mant_lo(long double x) { union ldshape u = {x}; return u.i.lo; }

// Round-trip: build a long double out of two words and read one back. This is
// scalbnl's shape.
// CHECK-LABEL: rebuild:
// CHECK:      %bb.0:
// CHECK-NEXT: cjalr zero, 0(ra)
unsigned long rebuild(unsigned long hi, unsigned long lo) {
  union ldshape u;
  u.i.hi = hi;
  u.i.lo = lo;
  union ldshape v = {u.f};
  return v.i.hi;
}

// Arithmetic: the soft-float libcall, not an error.
// CHECK-LABEL: addl:
// CHECK: __addtf3
long double addl(long double a, long double b) { return a + b; }
