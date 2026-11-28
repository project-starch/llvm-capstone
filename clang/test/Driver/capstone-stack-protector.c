// C-60: the IR stack protector cannot be run on Capstone: its intrinsics take
// address-space-0 pointers and the guard slot is an address-space-200 alloca, and
// the pass asserted. The driver refuses every -fstack-protector form as an error,
// so a configure or CMake flag probe fails and drops the flag instead of building
// without the protection it asked for. -fno-stack-protector, the default and other
// targets are unaffected.
// RUN: not %clang --target=capstone64-unknown-elf -fstack-protector -### -c %s 2>&1 | FileCheck %s --check-prefix=ERR -DOPT=-fstack-protector
// RUN: not %clang --target=capstone64-unknown-elf -fstack-protector-strong -### -c %s 2>&1 | FileCheck %s --check-prefix=ERR -DOPT=-fstack-protector-strong
// RUN: not %clang --target=capstone64-unknown-elf -fstack-protector-all -### -c %s 2>&1 | FileCheck %s --check-prefix=ERR -DOPT=-fstack-protector-all
// RUN: %clang --target=capstone64-unknown-elf -fno-stack-protector -### -c %s 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: %clang --target=capstone64-unknown-elf -fstack-protector -fno-stack-protector -### -c %s 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: %clang --target=capstone64-unknown-elf -### -c %s 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: %clang --target=riscv64-unknown-elf -fstack-protector-strong -### -c %s 2>&1 | FileCheck %s --check-prefix=RISCV

// ERR: error: unsupported option '[[OPT]]' for target 'capstone64
// ERR-NOT: "-stack-protector"
// OFF-NOT: error:
// OFF-NOT: "-stack-protector"
// RISCV-NOT: error:
// RISCV: "-stack-protector" "2"
