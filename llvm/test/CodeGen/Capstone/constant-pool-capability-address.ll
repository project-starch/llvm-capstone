; RUN: llc -mtriple=capstone64-unknown-elf -mattr=+m,+a,+f,+d -target-abi lp64 \
; RUN:   -verify-machineinstrs < %s | FileCheck %s
;
; C-80: every generic producer of a constant-pool address types it with
; TargetLowering::getConstantPoolPtrTy, the pointer type of the address space
; the target gives constant-pool memory (getAddressSpaceForPseudoSourceKind).
; On Capstone that is a capability (AS 200), which lowerConstantPool returns.
; Before the fix each producer below built ConstantPool:i64 (the AS0 pointer
; type), and on an assertions build llc aborted with "Type mismatch for custom
; legalized operation"; the inline-asm memory operand of a capability pointer
; aborted on "Memory operands expect pointer values". No select is needed:
; generic FP-constant expansion is enough.

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

; ExpandConstantFP, f64: loaded through the gp-derived pool capability.
; CHECK-LABEL: addpi:
; CHECK:       %pcrel_hi(.LCPI0_0)
; CHECK:       cincoffset [[P:a[0-9]+]], gp,
; CHECK:       fld {{f[a-z0-9]+}}, 0([[P]])
; CHECK:       fadd.d
define double @addpi(double %x) addrspace(200) {
  %r = fadd double %x, 0x400921FB54442D18
  ret double %r
}

; ExpandConstantFP, f32.
; CHECK-LABEL: addpif:
; CHECK:       %pcrel_hi(.LCPI1_0)
; CHECK:       cincoffset [[P:a[0-9]+]], gp,
; CHECK:       flw {{f[a-z0-9]+}}, 0([[P]])
; CHECK:       fadd.s
define float @addpif(float %x) addrspace(200) {
  %r = fadd float %x, 0x400921FB60000000
  ret float %r
}

; Inline asm "m" of an integer constant: spilled to the pool, address operand
; is the pool capability.
; CHECK-LABEL: asm_i64:
; CHECK:       %pcrel_hi(.LCPI2_0)
; CHECK:       cincoffset [[P:a[0-9]+]], gp,
; CHECK:       ld zero, 0([[P]])
define void @asm_i64() addrspace(200) {
  call addrspace(200) void asm sideeffect "ld zero, $0", "m"(i64 1234567890123456789)
  ret void
}

; Inline asm "m" of a double constant.
; CHECK-LABEL: asm_double:
; CHECK:       %pcrel_hi(.LCPI3_0)
; CHECK:       cincoffset [[P:a[0-9]+]], gp,
; CHECK:       fld ft0, 0([[P]])
define void @asm_double() addrspace(200) {
  call addrspace(200) void asm sideeffect "fld ft0, $0", "m"(double 0x400921FB54442D18)
  ret void
}

; Inline asm "*m" of a capability pointer: the operand is a c128, which the
; memory-operand assertion now accepts (it is the pointer type of the
; operand's own address space).
; CHECK-LABEL: asm_ptr:
; CHECK:       ld zero, 0(a0)
define void @asm_ptr(ptr addrspace(200) %p) addrspace(200) {
  call addrspace(200) void asm sideeffect "ld zero, $0", "*m"(ptr addrspace(200) elementtype(i64) %p)
  ret void
}
