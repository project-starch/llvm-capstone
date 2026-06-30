// A local aggregate whose constant initializer embeds capability pointers
// (addrspace(200) pointers to a function and to string globals) must be emitted
// as element-wise stores, each lowered to a tagged capability store, rather than
// a memcpy from a private untagged constant template (a bytewise copy cannot
// carry the out-of-band tag). See
// capstone/agent-handoff/design/cap-local-aggregate-init-plan.md.

// RUN: %clang_cc1 -triple capstone64-unknown-elf -ffreestanding -O0 -emit-llvm \
// RUN:   -o - %s | FileCheck %s

typedef unsigned (*fn)(void);
struct e { fn function; const char *name; };

unsigned f0(void) { return 7; }

// CHECK-LABEL: define {{.*}} @use
// CHECK-NOT: @llvm.memcpy{{.*}}@__const.use
// CHECK-DAG: store ptr addrspace(200) @f0,
// CHECK-DAG: store ptr addrspace(200) @.str,
// CHECK-DAG: store ptr addrspace(200) @.str.1,
unsigned use(unsigned i) {
  struct e local[] = {{f0, "ab"}, {f0, "cd"}};
  return local[i].function() + (unsigned)local[i].name[0];
}
