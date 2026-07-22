; gp-free / plain-call-ret domain ABI (-capstone-gp-free), Stage 1: calls and
; returns.
;
; Default (capability ABI): a direct call forms a code capability via
; `cincoffset gp, <pcrel>` + CJALR; an indirect call and every return are CJALR
; (which needs a code capability in the target/return register).
;
; With -capstone-gp-free: calls/returns lower to plain jal/jalr that stay inside
; PCC (bounds-checked on fetch), and a direct call becomes a plain PC-relative
; `call` with no gp root. This is the reference monitor's within-PCC call/ret ABI
; and lets a real globals-using domain run on silicon, where `gp = PCC(cursor 0)`
; is not establishable. The flag defaults off, so ordinary codegen is unchanged.
;
; See CapstoneAsmPrinter.cpp emitInstruction, CapstoneISelDAGToDAG.cpp selectCall,
; and plans/compatibility-eval-silicon-app.md §2.

; RUN: llc -mtriple=capstone64 -mattr=+m < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,CAP
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-free < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,GPFREE

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

declare void @ext_fn()

@g = addrspace(200) global [16 x i8] zeroinitializer, align 1

; Global data addressing: default derives the object cap from gp by ADDING the
; absolute VA (cincoffset), which needs gp.cursor==0 (the QEMU-fabricated
; gp = PCC(cursor 0), unrepresentable on the RTL). gp-free SETS the cursor
; absolutely (scc), so a representable image-bounded gp works. gp is still the
; base register in both (data loads require a capability base; there is no
; PCC->data-cap instruction), so this decouples from cursor-0, it does not remove
; gp. The object-narrowing SHRINK is unchanged.
; CHECK-LABEL: load_global:
; CAP:        cincoffset {{s?a?[0-9]+}}, gp, {{a[0-9]+}}
; CAP-NOT:    scc {{.*}}, gp,
; GPFREE:     scc {{s?a?[0-9]+}}, gp, {{a[0-9]+}}
; GPFREE-NOT: cincoffset {{.*}}, gp,
define i8 @load_global() {
  %p = getelementptr [16 x i8], ptr addrspace(200) @g, i64 0, i64 3
  %v = load i8, ptr addrspace(200) %p
  ret i8 %v
}

; A direct call: default derives a code cap from gp then cjalr; gp-free is a
; plain PC-relative call with no gp and no cjalr.
; CHECK-LABEL: calls_direct:
; CAP:      cincoffset {{a[0-9]+}}, gp, {{a[0-9]+}}
; CAP:      cjalr ra, 0({{a[0-9]+}})
; CAP:      cjalr zero, 0(ra)
; GPFREE:     auipc [[T:a[0-9]+]], %pcrel_hi(ext_fn)
; GPFREE:     jalr [[T]]
; GPFREE-NOT: cjalr
; GPFREE:     ret
define void @calls_direct() {
  call void @ext_fn()
  ret void
}

; An indirect call through a function pointer: default cjalr; gp-free plain jalr.
; CHECK-LABEL: calls_indirect:
; CAP:        cjalr ra, 0({{a[0-9]+}})
; CAP:        cjalr zero, 0(ra)
; GPFREE:     jalr {{a[0-9]+}}
; GPFREE-NOT: cjalr
; GPFREE:     ret
define void @calls_indirect(ptr addrspace(200) %fp) {
  call void %fp()
  ret void
}

; A leaf function's return: default capability return; gp-free plain ret.
; CHECK-LABEL: leaf:
; CAP:        cjalr zero, 0(ra)
; GPFREE-NOT: cjalr
; GPFREE:     ret
define i32 @leaf(i32 %x) {
  %y = add i32 %x, 1
  ret i32 %y
}
