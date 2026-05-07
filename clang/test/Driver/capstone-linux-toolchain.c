// RUN: %clang -target capstone64-unknown-linux-gnu -fuse-ld=lld -nostdlib -### %s 2>&1 | FileCheck %s

// CHECK: ld.lld
// CHECK-SAME: "-m" "elf64lcapstone"
// CHECK-SAME: "-dynamic-linker" "/lib/ld-linux-capstone64-lp64d.so.1"

int main(void) { return 0; }


