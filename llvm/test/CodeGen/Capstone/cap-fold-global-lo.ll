; -capstone-fold-global-lo: under the image-gp ABI, a global whose gp-derived
; capability is only the base of loads and stores at one common offset is
; reached as `auipc; scc gp; <access> %pcrel_lo(...)`, without the addi. Any
; other use of the capability (stored, passed, two offsets, bounded) keeps the
; complete address. The flag is off by default, and off it changes nothing.
; The rule this relaxes, and the link-script margins it relies on:
; capstone/docs/design/capability-loop-addressing.md, "The global-address
; exception".
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-free -capstone-image-gp \
; RUN:   -capstone-shrinkto-globals -capstone-shrinkto-end-adjust=1 \
; RUN:   -capstone-drop-redundant-shrink -capstone-drop-redundant-shrink-declarations \
; RUN:   -capstone-fold-global-lo < %s | FileCheck %s --check-prefixes=CHECK,FOLD
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-free -capstone-image-gp \
; RUN:   -capstone-shrinkto-globals -capstone-shrinkto-end-adjust=1 \
; RUN:   -capstone-drop-redundant-shrink -capstone-drop-redundant-shrink-declarations \
; RUN:   < %s | FileCheck %s --check-prefixes=CHECK,KEEP
; Without the image-gp ABI the flag has no effect.
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-free \
; RUN:   -capstone-fold-global-lo < %s | FileCheck %s --check-prefix=NOIMG

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
@p = addrspace(200) global ptr addrspace(200) null, align 16
@a = addrspace(200) global [4 x i32] zeroinitializer, align 4
@x = addrspace(200) global i32 0, align 4
@c = addrspace(200) global i64 0, align 8
@e = addrspace(200) global i64 0, align 8
declare void @use(ptr addrspace(200)) addrspace(200)

; A load at offset 0.
; CHECK-LABEL: load_word:
; FOLD:      .Lpcrel_hi[[N:[0-9]+]]:
; FOLD-NEXT: auipc [[HI:[a-z0-9]+]], %pcrel_hi(x)
; FOLD-NEXT: scc [[CAP:[a-z0-9]+]], gp, [[HI]]
; FOLD-NEXT: lw a0, %pcrel_lo(.Lpcrel_hi[[N]])([[CAP]])
; KEEP:      addi
; KEEP:      scc
; KEEP:      lw a0, 0(
; NOIMG-LABEL: load_word:
; NOIMG-NOT: %pcrel_lo(.Lpcrel_hi{{[0-9]+}})(
; NOIMG:     ret
define i32 @load_word() addrspace(200) {
  %v = load i32, ptr addrspace(200) @x
  ret i32 %v
}

; A capability load, and a store, at one nonzero common offset: the offset
; moves into the auipc's symbol.
; CHECK-LABEL: rmw_elem:
; FOLD:      auipc [[HI:[a-z0-9]+]], %pcrel_hi(a+12)
; FOLD-NOT:  addi {{.*}}%pcrel_lo
; FOLD:      lw {{[a-z0-9]+}}, %pcrel_lo(
; FOLD:      sw {{[a-z0-9]+}}, %pcrel_lo(
; KEEP:      lw {{[a-z0-9]+}}, 12(
define void @rmw_elem() addrspace(200) {
  %q = getelementptr [4 x i32], ptr addrspace(200) @a, i64 0, i64 3
  %v = load i32, ptr addrspace(200) %q
  %w = add i32 %v, 1
  store i32 %w, ptr addrspace(200) %q
  ret void
}

; CHECK-LABEL: load_cap:
; FOLD:      scc
; FOLD-NEXT: ldc a0, %pcrel_lo(
define ptr addrspace(200) @load_cap() addrspace(200) {
  %v = load ptr addrspace(200), ptr addrspace(200) @p
  ret ptr addrspace(200) %v
}

; Two different offsets: one %pcrel_lo cannot serve both, so no fold.
; CHECK-LABEL: two_offsets:
; CHECK:     addi {{[a-z0-9]+}}, {{[a-z0-9]+}}, %pcrel_lo(
; CHECK:     lw {{[a-z0-9]+}}, 4(
; CHECK:     lw {{[a-z0-9]+}}, 8(
define i32 @two_offsets() addrspace(200) {
  %q1 = getelementptr [4 x i32], ptr addrspace(200) @a, i64 0, i64 1
  %q2 = getelementptr [4 x i32], ptr addrspace(200) @a, i64 0, i64 2
  %v1 = load i32, ptr addrspace(200) %q1
  %v2 = load i32, ptr addrspace(200) %q2
  %s = add i32 %v1, %v2
  ret i32 %s
}

; The capability escapes as a call argument: its cursor survives, no fold.
; CHECK-LABEL: call_arg:
; CHECK:     addi {{[a-z0-9]+}}, {{[a-z0-9]+}}, %pcrel_lo(
; CHECK-NOT: %pcrel_lo({{.*}})(
; CHECK:     ret
define void @call_arg() addrspace(200) {
  call void @use(ptr addrspace(200) @c)
  ret void
}

; The capability is stored as data: no fold.
; CHECK-LABEL: store_cap:
; CHECK:     addi {{[a-z0-9]+}}, {{[a-z0-9]+}}, %pcrel_lo(
; CHECK-NOT: %pcrel_lo({{.*}})(
; CHECK:     ret
define void @store_cap(ptr addrspace(200) %d) addrspace(200) {
  store ptr addrspace(200) @e, ptr addrspace(200) %d
  ret void
}
