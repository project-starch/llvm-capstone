// Capstone keeps the frame pointer in functions that call and omits it in
// leaves ("non-leaf"), at every optimization level; -fomit-frame-pointer and
// -fno-omit-frame-pointer still decide as everywhere else.
// RUN: %clang --target=capstone64-unknown-elf -### -S -O0 %s 2>&1 | FileCheck -check-prefix=NONLEAF %s
// RUN: %clang --target=capstone64-unknown-elf -### -S -O2 %s 2>&1 | FileCheck -check-prefix=NONLEAF %s
// RUN: %clang --target=capstone64-unknown-elf -### -S -O2 -fomit-frame-pointer %s 2>&1 | FileCheck -check-prefix=NONE %s
// RUN: %clang --target=capstone64-unknown-elf -### -S -O2 -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer %s 2>&1 | FileCheck -check-prefix=ALL %s
// NONLEAF: "-mframe-pointer=non-leaf"
// NONE-NOT: "-mframe-pointer=all"
// NONE-NOT: "-mframe-pointer=non-leaf"
// ALL: "-mframe-pointer=all"
void f(void) {}
