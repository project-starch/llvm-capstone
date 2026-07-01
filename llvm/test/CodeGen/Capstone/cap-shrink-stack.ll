; Object-granularity capability narrowing for address-taken stack objects (C1,
; stack slice, opt-in via -capstone-shrink-stack). When on, materializing the
; address of a stack object narrows its capability to [&obj, &obj+size) via
; lcc(cursor)/add/shrink. This applies both to the bare-FrameIndex address
; (ISD::FrameIndex) and to interior / load-store base materialization
; (materializeFrameIndexAddrBase) via the shared narrowToFrameObjectBounds
; helper, so a load/store *through* a fixed stack object also carries object
; bounds (still object- not subobject-granularity: the base narrows to the whole
; frame object, and a field/element access rides an in-bounds offset). Default
; off -> no narrowing.
;
; See CapstoneISelDAGToDAG.cpp (narrowToFrameObjectBounds) and
; design/capability-bounds-model.md.

; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-shrink-stack=true < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,SHRINK
; RUN: llc -mtriple=capstone64 -mattr=+m < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,NOSHRINK

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

; A 64-byte local indexed by a runtime value: &buf (bare FrameIndex) is narrowed
; to its object when the flag is on, and left at the broad stack bounds when off.
; CHECK-LABEL: stack_idx:
; CHECK: cincoffsetimm {{a[0-9]+}}, sp, 0
; SHRINK: lcc {{a[0-9]+}}, {{a[0-9]+}}, 2
; SHRINK: shrink
; NOSHRINK-NOT: shrink
define i8 @stack_idx(i64 %i) addrspace(200) {
  %buf = alloca [64 x i8], align 1, addrspace(200)
  %p = getelementptr [64 x i8], ptr addrspace(200) %buf, i64 0, i64 %i
  %v = load i8, ptr addrspace(200) %p
  ret i8 %v
}

; A capability stored/loaded through a stack slot: the ldc/stc base is a frame
; index, materialized via materializeFrameIndexAddrBase, so it is now narrowed
; to the slot's bounds when the flag is on (broad stack bounds when off).
; CHECK-LABEL: cap_slot:
; CHECK: cincoffsetimm {{a[0-9]+}}, sp, 0
; SHRINK: lcc {{a[0-9]+}}, {{a[0-9]+}}, 2
; SHRINK: shrink
; SHRINK: stc
; NOSHRINK-NOT: shrink
define ptr addrspace(200) @cap_slot(ptr addrspace(200) %x) addrspace(200) {
  %slot = alloca ptr addrspace(200), align 16, addrspace(200)
  store ptr addrspace(200) %x, ptr addrspace(200) %slot
  %v = load ptr addrspace(200), ptr addrspace(200) %slot
  ret ptr addrspace(200) %v
}

; A scalar store into a field of a stack struct at a constant offset: the base
; capability is narrowed to the whole frame object [&s, &s+16), and the store
; rides an in-bounds displacement (sd ..., 8(base)). Object-granularity: the
; base is not narrowed to the subobject field.
; CHECK-LABEL: field_store:
; SHRINK: shrink
; SHRINK: sd {{a[0-9]+}}, 8({{a[0-9]+}})
; NOSHRINK-NOT: shrink
define void @field_store(i64 %v) addrspace(200) {
  %s = alloca { i64, i64 }, align 8, addrspace(200)
  %f = getelementptr { i64, i64 }, ptr addrspace(200) %s, i64 0, i32 1
  store i64 %v, ptr addrspace(200) %f
  ret void
}
