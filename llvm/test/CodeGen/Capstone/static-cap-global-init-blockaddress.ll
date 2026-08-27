; A computed-goto dispatch table is an array of BLOCK ADDRESSES, and a block
; address is neither a GlobalVariable nor a Function -- so CapstoneCapGlobalInit
; used to walk straight past it. The table kept its link-time bytes, and since a
; domain is loaded at a runtime base and processes no load-time relocations, the
; first `goto *tbl[i]` jumped outside the image.
;
; That is not an ABI limit: lowerBlockAddress already materializes a block address
; as the same LGA capability lowerGlobalAddress produces, so the store costs
; nothing new in the backend. It was a missing isa<>. Measured on WAMR's classic
; interpreter: 224 live slots in one table, all silently untagged.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m < %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

@tbl = internal addrspace(200) global [2 x ptr addrspace(200)]
    [ptr addrspace(200) blockaddress(@dispatch, %one),
     ptr addrspace(200) blockaddress(@dispatch, %two)], align 16

; An alias is neither a GlobalVariable nor a Function either, and was skipped for
; the same reason.
; A vector of capabilities cannot be reached by the GEP path this pass builds, so
; it is REPORTED rather than skipped quietly. This is the positive control for that
; warning: without a case that trips it, "no warnings in the build" says nothing
; about the build and something about the check. It is written here rather than in
; the shell probe because clang rejects a vector of pointers at the source level.
@vec = addrspace(200) global <2 x ptr addrspace(200)>
    <ptr addrspace(200) @real, ptr addrspace(200) @real>, align 32

; WARN: warning: capstone-cap-init: vec holds a capability this pass does not materialize (vector of capabilities)

@real = addrspace(200) global i32 7, align 4
@ali = alias i32, ptr addrspace(200) @real
@via_alias = addrspace(200) global ptr addrspace(200) @ali, align 16

; A null slot still needs no tag, and an absolute address cannot carry one:
; neither may produce a store. (MicroPython's MP_ROM_INT is the inttoptr case.)
@untouched = addrspace(200) global [2 x ptr addrspace(200)]
    [ptr addrspace(200) null,
     ptr addrspace(200) inttoptr (i128 14 to ptr addrspace(200))], align 16

define i32 @dispatch(i32 %n) {
entry:
  %i = and i32 %n, 1
  %p = getelementptr inbounds [2 x ptr addrspace(200)], ptr addrspace(200) @tbl, i64 0, i32 %i
  %t = load ptr addrspace(200), ptr addrspace(200) %p, align 16
  indirectbr ptr addrspace(200) %t, [label %one, label %two]
one:
  ret i32 1
two:
  ret i32 2
}

; Both table slots and the alias are stored at run time. Checking the TARGETS,
; not just that three stores appear: a store to the wrong label would satisfy a
; bare count and is exactly the kind of thing this pass could get wrong.
; CHECK-LABEL: __capstone_cap_init:
; CHECK:       %pcrel_hi(.Ltmp0)
; CHECK:       stc
; CHECK:       %pcrel_hi(.Ltmp1)
; CHECK:       stc
; CHECK:       %pcrel_hi(ali)
; CHECK:       stc

; And nothing else: the null slot needs no tag and the absolute address cannot
; carry one, so @untouched must produce no store at all.
; CHECK-NOT:   stc
; CHECK:       cjalr

; The table itself still carries its static (untagged) image bytes, which is what
; the initializer overwrites in place.
; CHECK-LABEL: tbl:
; CHECK:       .quad
