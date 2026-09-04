; PseudoTRUNC_CAP (capability -> integer address) and PseudoCapGlobalBase
; (gp-relative global base) are expanded after ISel.  This pins the expansion
; through the OBJECT path -- llc -filetype=obj, then llvm-objdump -- so the
; bytes a linker sees carry exactly the instructions the text path prints.
;
; Since 2026-09-05 (C-31) every capability-to-integer read is an INTEGER WRITE:
; PseudoTRUNC_CAP expands to `addi rd, rs, 0` (printed `mv`), even when rd and
; rs coincide (@cursor_a0: `mv a0, a0`), because a bare sub-register read left
; the consumer reading a register whose metadata shadow was still tagged, which
; the RTL rejects as the rs2 of cincoffset/scc.  When the source is another
; register the expansion is the integer `mv a0, a1` (@cursor_a1) -- it used to
; coalesce into a FULL capability move, `movc a0, a1`, which on the RTL also
; nulls an untagged source (C-32).  The pseudo stays a pseudo until MC lowering
; so that copy propagation cannot delete the self-move.  The same read gives
; the SHRINK sequence its base (@load_global: `mv a1, a0`, formerly
; `lcc a1, a0, 2`, a query that is not total).
; Measured 2026-09-05 on the cycle-2 tools; objdump prints immediates in hex.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -filetype=obj -o %t.o %s
; RUN: llvm-objdump -d %t.o | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -o - %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-captable -filetype=obj -o %t.gpct.o %s
; RUN: llvm-objdump -d %t.gpct.o | FileCheck %s --check-prefix=GPCT

@g = addrspace(200) global i64 0

define i64 @cursor_a0(ptr addrspace(200) %p) {
; CHECK-LABEL: <cursor_a0>:
; CHECK-NEXT: mv a0, a0
; CHECK-NEXT: cjalr zero, 0x0(ra)
; GPCT-LABEL: <cursor_a0>:
; GPCT-NEXT: mv a0, a0
; GPCT-NEXT: ret
  %v = ptrtoint ptr addrspace(200) %p to i64
  ret i64 %v
}

define i64 @cursor_a1(ptr addrspace(200) %p, ptr addrspace(200) %q) {
; CHECK-LABEL: <cursor_a1>:
; CHECK-NEXT: mv a0, a1
; CHECK-NEXT: cjalr zero, 0x0(ra)
; ASM-LABEL: cursor_a1:
; ASM: mv a0, a1
; ASM: cjalr zero, 0(ra)
; GPCT-LABEL: <cursor_a1>:
; GPCT-NEXT: mv a0, a1
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
; CHECK-NEXT: mv a1, a0
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
; ASM-NEXT: mv a1, a0
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
