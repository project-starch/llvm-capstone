// RUN: %clang_cc1 -triple capstone64-unknown-elf -ffreestanding -emit-llvm \
// RUN:   -O0 -o - %s | FileCheck %s

// A negative subscript sign-extends to the GEP index width. That width is the
// ADDRESS width, not the pointer width: it used to be i128, because the pointer
// integer type was the capability width, and CodeGen's alignment refinement then
// had to cope with a constant index that does not fit in uint64_t. It does fit
// now, and the routine that assumed it can no longer be reached this way from C.
//
// A GEP with a wider index is still expressible in IR and still narrowed
// correctly -- see llvm/test/CodeGen/Capstone/cap-stack-addressing.ll.

// CHECK-LABEL: define{{.*}} i8 @load_before
// CHECK: getelementptr inbounds i8, ptr addrspace(200) %{{.*}}, i64 -1
// CHECK: load i8
char load_before(char *pointer) {
  return pointer[-1];
}
