// C1 subobject-bounds narrowing (-fcapstone-subobject-bounds, default off).
//
// With the flag on, a *non-last* array field's lvalue is narrowed to the field's
// own bounds via the capstone_cap_shrink intrinsic, so an over-read that leaves
// the field but stays inside the enclosing object faults at runtime. v1 refuses
// to narrow: union members, flexible/incomplete array members, any last-member
// (trailing) array, and scalar fields. With the flag off (default) nothing is
// narrowed in the frontend at all.
//
// RUN: %clang_cc1 -triple capstone64-unknown-elf -ffreestanding -emit-llvm -O0 \
// RUN:   -fcapstone-subobject-bounds -o - %s | FileCheck %s --check-prefix=ON
// RUN: %clang_cc1 -triple capstone64-unknown-elf -ffreestanding -emit-llvm -O0 \
// RUN:   -o - %s | FileCheck %s --check-prefix=OFF

typedef unsigned long size_t;

struct two_arrays {
  unsigned char first[8];   // non-last array field  -> narrowed
  unsigned char second[8];  // last-member array     -> refused (trailing-array)
};

struct scalar_then_array {
  int tag;                  // scalar field          -> refused (v1: arrays only)
  unsigned char buf[16];    // last-member array     -> refused (trailing-array)
};

union u_arr {
  unsigned char a[8];       // union member          -> refused (overlap)
  long i;
};

struct flex {
  int n;
  unsigned char data[];     // flexible array member -> refused (incomplete)
};

// A non-last array field IS narrowed: expect a cursor read + shrink.
// ON-LABEL: @read_first(
// ON: call{{.*}}@llvm.capstone.cap.get.cursor.p200
// ON: call{{.*}}@llvm.capstone.cap.shrink.p200
// OFF-LABEL: @read_first(
// OFF-NOT: @llvm.capstone.cap.shrink
unsigned char read_first(struct two_arrays *s, unsigned i) {
  volatile unsigned idx = i;
  return s->first[idx];
}

// The last-member array is a trailing-array idiom: refused.
// ON-LABEL: @read_second(
// ON-NOT: @llvm.capstone.cap.shrink
unsigned char read_second(struct two_arrays *s, unsigned i) {
  volatile unsigned idx = i;
  return s->second[idx];
}

// Scalar field: v1 narrows only arrays -> refused.
// ON-LABEL: @read_tag(
// ON-NOT: @llvm.capstone.cap.shrink
int read_tag(struct scalar_then_array *s) {
  return s->tag;
}

// Union member array: overlapping members -> refused.
// ON-LABEL: @read_union(
// ON-NOT: @llvm.capstone.cap.shrink
unsigned char read_union(union u_arr *u, unsigned i) {
  volatile unsigned idx = i;
  return u->a[idx];
}

// Flexible array member (incomplete) -> refused.
// ON-LABEL: @read_flex(
// ON-NOT: @llvm.capstone.cap.shrink
unsigned char read_flex(struct flex *f, unsigned i) {
  volatile unsigned idx = i;
  return f->data[idx];
}
