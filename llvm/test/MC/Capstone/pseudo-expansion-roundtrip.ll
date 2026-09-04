; PseudoTRUNC_CAP (capability -> integer address) and PseudoCapGlobalBase
; (gp-relative global base) are expanded after ISel.  This pins the expansion
; through the OBJECT path -- llc -filetype=obj, then llvm-objdump -- so the
; bytes a linker sees carry exactly the instructions the text path prints.
; @cursor_a0 is the control for @cursor_a1: same shape, but the address is
; already in a0's integer half, so the pseudo expands to NOTHING and the body
; is the return alone.  When the source is another register the pseudo
; expands to a FULL capability move (`movc a0, a1`), not an integer `mv`: the
; integer half of the destination is the cursor, and the metadata half rides
; along.  (Whether that is the right expansion on silicon -- RTL nulls an
; UNTAGGED movc source -- is a Tier 4 semantics row, not this test's claim.)
; Measured 2026-09-04 on the branch tools; objdump prints immediates in hex.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -filetype=obj -o %t.o %s
; RUN: llvm-objdump -d %t.o | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -o - %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-captable -filetype=obj -o %t.gpct.o %s
; RUN: llvm-objdump -d %t.gpct.o | FileCheck %s --check-prefix=GPCT

@g = addrspace(200) global i64 0

define i64 @cursor_a0(ptr addrspace(200) %p) {
; CHECK-LABEL: <cursor_a0>:
; CHECK-NEXT: cjalr zero, 0x0(ra)
; GPCT-LABEL: <cursor_a0>:
; GPCT-NEXT: ret
  %v = ptrtoint ptr addrspace(200) %p to i64
  ret i64 %v
}

define i64 @cursor_a1(ptr addrspace(200) %p, ptr addrspace(200) %q) {
; CHECK-LABEL: <cursor_a1>:
; CHECK-NEXT: movc a0, a1
; CHECK-NEXT: cjalr zero, 0x0(ra)
; ASM-LABEL: cursor_a1:
; ASM: movc a0, a1
; ASM: cjalr zero, 0(ra)
; GPCT-LABEL: <cursor_a1>:
; GPCT-NEXT: movc a0, a1
; GPCT-NEXT: ret
  %v = ptrtoint ptr addrspace(200) %q to i64
  ret i64 %v
}

define i64 @load_global() {
; CHECK-LABEL: <load_global>:
; CHECK-NEXT: auipc a0, 0x0
; CHECK-NEXT: mv a0, a0
; CHECK-NEXT: cincoffset a0, gp, a0
; CHECK-NEXT: delin a0
; CHECK-NEXT: lcc a1, a0, 0x2
; CHECK-NEXT: li a2, 0x8
; CHECK-NEXT: add a2, a1, a2
; CHECK-NEXT: shrink a0, a1, a2
; CHECK-NEXT: ld a0, 0x0(a0)
; CHECK-NEXT: cjalr zero, 0x0(ra)
; ASM-LABEL: load_global:
; ASM: auipc a0, %pcrel_hi(g)
; ASM-NEXT: addi a0, a0, %pcrel_lo(.Lpcrel_hi0)
; ASM-NEXT: cincoffset a0, gp, a0
; ASM-NEXT: delin a0
; ASM-NEXT: lcc a1, a0, 2
; ASM-NEXT: li a2, 8
; ASM-NEXT: add a2, a1, a2
; ASM-NEXT: shrink a0, a1, a2
; ASM-NEXT: ld a0, 0(a0)
; ASM: cjalr zero, 0(ra)
; GPCT-LABEL: <load_global>:
; GPCT-NEXT: ldc a0, 0x0(gp)
; GPCT-NEXT: ld a0, 0x0(a0)
; GPCT-NEXT: ret
  %v = load i64, ptr addrspace(200) @g
  ret i64 %v
}
