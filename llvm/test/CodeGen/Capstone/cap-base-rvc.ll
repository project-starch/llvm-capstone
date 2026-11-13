; +cap-base-rvc: with C, a load or store whose base is a capability in c8-c15,
; or the stack capability, is emitted as the standard 16-bit form (c.ld, c.sd,
; c.lw, c.sw, c.fld, c.fsd and their SP-relative forms). Every access on this
; target has a capability base, so without the feature none of them compresses.
; The 16-bit form expands to the same 32-bit encoding, so the disassembly of the
; object is the only thing that changes. Offsets out of the 16-bit range, and
; capability loads and stores (ldc/stc), stay 32-bit.
; RUN: llc -mtriple=capstone64 -mattr=+m,+f,+d,+c,+cap-base-rvc -capstone-gp-free \
; RUN:   -filetype=obj < %s | llvm-objdump -d -M no-aliases --mattr=+c,+d - \
; RUN:   | FileCheck %s --check-prefixes=CHECK,CB
; RUN: llc -mtriple=capstone64 -mattr=+m,+f,+d,+c -capstone-gp-free \
; RUN:   -filetype=obj < %s | llvm-objdump -d -M no-aliases --mattr=+c,+d - \
; RUN:   | FileCheck %s --check-prefixes=CHECK,NOCB
; Without C the feature changes nothing.
; RUN: llc -mtriple=capstone64 -mattr=+m,+f,+d,+cap-base-rvc -capstone-gp-free \
; RUN:   -filetype=obj < %s | llvm-objdump -d -M no-aliases --mattr=+d - \
; RUN:   | FileCheck %s --check-prefixes=CHECK,NOC

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

; Integer loads and stores off a capability argument (a0 = c10, a1 = c11).
; CHECK-LABEL: <ld_sd>:
; CB:   c.ld a0, 0x8(a0)
; CB:   c.sd a0, 0x10(a1)
; NOCB: ld a0, 0x8(a0)
; NOCB: sd a0, 0x10(a1)
; NOC:  ld a0, 0x8(a0)
; NOC:  sd a0, 0x10(a1)
define void @ld_sd(ptr addrspace(200) %p, ptr addrspace(200) %q) addrspace(200) {
  %a = getelementptr i64, ptr addrspace(200) %p, i64 1
  %v = load i64, ptr addrspace(200) %a
  %b = getelementptr i64, ptr addrspace(200) %q, i64 2
  store i64 %v, ptr addrspace(200) %b
  ret void
}

; CHECK-LABEL: <lw_sw>:
; CB:   c.lw a0, 0x4(a0)
; CB:   c.sw a0, 0x7c(a1)
; NOCB: lw a0, 0x4(a0)
; NOCB: sw a0, 0x7c(a1)
define void @lw_sw(ptr addrspace(200) %p, ptr addrspace(200) %q) addrspace(200) {
  %a = getelementptr i32, ptr addrspace(200) %p, i64 1
  %v = load i32, ptr addrspace(200) %a
  %b = getelementptr i32, ptr addrspace(200) %q, i64 31
  store i32 %v, ptr addrspace(200) %b
  ret void
}

; Out of the 16-bit ranges: offset 256 for a doubleword (c.ld reaches 248), a
; negative offset, and offset 128 for a word (c.lw reaches 124).
; CHECK-LABEL: <out_of_range>:
; CHECK-DAG: {{^ +[0-9a-f]+: [0-9a-f]{2} [0-9a-f]{2} [0-9a-f]{2} [0-9a-f]{2} +}}ld a{{[0-9]}}, -0x8(a0)
; CHECK-DAG: {{^ +[0-9a-f]+: [0-9a-f]{2} [0-9a-f]{2} [0-9a-f]{2} [0-9a-f]{2} +}}lw a{{[0-9]}}, 0x80(a0)
; CHECK-DAG: {{^ +[0-9a-f]+: [0-9a-f]{2} [0-9a-f]{2} [0-9a-f]{2} [0-9a-f]{2} +}}ld a{{[0-9]}}, 0x100(a0)
define i64 @out_of_range(ptr addrspace(200) %p) addrspace(200) {
  %a = getelementptr i8, ptr addrspace(200) %p, i64 256
  %x = load i64, ptr addrspace(200) %a
  %b = getelementptr i8, ptr addrspace(200) %p, i64 -8
  %y = load i64, ptr addrspace(200) %b
  %c = getelementptr i8, ptr addrspace(200) %p, i64 128
  %w = load i32, ptr addrspace(200) %c
  %z = sext i32 %w to i64
  %s = add i64 %x, %y
  %t = add i64 %s, %z
  ret i64 %t
}

; Double-precision loads and stores off a capability.
; CHECK-LABEL: <fld_fsd>:
; CB:   c.fld fa5, 0x8(a0)
; CB:   c.fsd fa5, 0x0(a1)
; NOCB: fld fa5, 0x8(a0)
; NOCB: fsd fa5, 0x0(a1)
define void @fld_fsd(ptr addrspace(200) %p, ptr addrspace(200) %q) addrspace(200) {
  %a = getelementptr double, ptr addrspace(200) %p, i64 1
  %v = load double, ptr addrspace(200) %a
  %w = fadd double %v, %v
  store double %w, ptr addrspace(200) %q
  ret void
}

; The stack capability: the return address is saved and restored with an
; integer store and load off sp; capability spills (stc/ldc) stay 32-bit.
; CHECK-LABEL: <stack>:
; CB:   c.sdsp ra, {{.*}}(sp)
; CB:   c.ldsp ra, {{.*}}(sp)
; NOCB: sd ra, {{.*}}(sp)
; NOCB: ld ra, {{.*}}(sp)
declare void @g(ptr addrspace(200)) addrspace(200)
define void @stack() addrspace(200) {
  %x = alloca i64, align 8, addrspace(200)
  call void @g(ptr addrspace(200) %x)
  ret void
}
