; An `"r"` inline-asm operand holding a CAPABILITY must get a capability
; register, and must reach the asm still TAGGED.
;
; This is the half of the register-class split that only inline asm exercises.
; While one class held integers and capabilities, `"r"` needed no decision. Once
; GPR became integer-only, `"r"` kept returning it and a capability operand
; reached getCopyFromParts with nothing to reconcile -- "Unknown mismatch in
; getCopyFromParts!", which is how CoreMark failed to build. lit did not catch
; it; the QEMU tier did, which is why it is pinned here.
;
; The register prints by its X name. C and X are the same hardware register and
; encode identically, and `.insn` and hand-written asm only know the X names --
; so `a0`, not `c10`, even though the value is a capability.
;
; RUN: llc -mtriple=capstone64 -verify-machineinstrs < %s | FileCheck %s

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

; `delin` faults on an untagged register, so nothing may clear the tag on the way
; in. A `mv` here is the exact bug this guards: it is ADDI, which writes the
; address half and drops the tag.
; CHECK-LABEL: delin_ptr:
; CHECK-NOT: mv
; CHECK: .insn r 91, 1, 3, a0, zero, zero
; CHECK-NOT: mv
define ptr addrspace(200) @delin_ptr(ptr addrspace(200) %p) {
  %1 = call ptr addrspace(200) asm sideeffect ".insn r 0x5b, 0x1, 0x3, $0, x0, x0", "=r,0"(ptr addrspace(200) %p)
  ret ptr addrspace(200) %1
}

; The control: an INTEGER through the same constraint still gets an integer
; register. Without it "capabilities go to GPCR" would be satisfied by a backend
; that sent everything there.
; CHECK-LABEL: asm_int:
; CHECK: addi a0, a0, 1
define i64 @asm_int(i64 %x) {
  %1 = call i64 asm sideeffect "addi $0, $0, 1", "=r,0"(i64 %x)
  ret i64 %1
}

; A capability that is used AFTER the asm as a capability -- the shape where a
; dropped tag would fault at run time rather than merely look wrong.
; CHECK-LABEL: delin_then_load:
; CHECK: .insn r 91, 1, 3, {{a[0-9]+}}, zero, zero
; CHECK: ld a0, 0({{a[0-9]+}})
define i64 @delin_then_load(ptr addrspace(200) %p) {
  %1 = call ptr addrspace(200) asm sideeffect ".insn r 0x5b, 0x1, 0x3, $0, x0, x0", "=r,0"(ptr addrspace(200) %p)
  %v = load i64, ptr addrspace(200) %1
  ret i64 %v
}
