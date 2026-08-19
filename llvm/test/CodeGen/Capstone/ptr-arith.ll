; Pinned to -capstone-shrink-stack=false (default on since 2026-07-03): this test
; checks pointer-arithmetic lowering, not stack narrowing (the narrowed path is
; covered by cap-shrink-{stack,dynalloca}.ll).
; RUN: llc -mtriple=capstone64 -capstone-shrink-stack=false -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: test_imm:
; CHECK: cincoffsetimm a0, a0, 1
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @test_imm(ptr addrspace(200) %p) {
entry:
  %add.ptr = getelementptr inbounds i8, ptr addrspace(200) %p, i128 1
  ret ptr addrspace(200) %add.ptr
}

; CHECK-LABEL: test_reg:
; CHECK: cincoffset a0, a0, a1
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @test_reg(ptr addrspace(200) %p, i128 %offset) {
entry:
  %add.ptr = getelementptr inbounds i8, ptr addrspace(200) %p, i128 %offset
  ret ptr addrspace(200) %add.ptr
}

; Scaled-index GEP: index is i64, multiplied by element size to shl(sext, k).
; lowerADD must recognise shl(sext, k) as an integer offset so the capability
; ends up as rs1 in cincoffset.
; CHECK-LABEL: test_scaled_gep:
; CHECK: cincoffset a0, a0, a1
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @test_scaled_gep(ptr addrspace(200) %arr, i64 %i) {
  %p = getelementptr i32, ptr addrspace(200) %arr, i64 %i
  ret ptr addrspace(200) %p
}

; Two independent scaled GEPs in one function force the backend to emit two
; cincoffsets.  Both must have the capability in rs1, not the integer index.
; CHECK-LABEL: test_two_geps:
; CHECK: slli [[OFF1:a[0-9]+]], {{a[0-9]+}}, 2
; CHECK: slli [[OFF2:a[0-9]+]], {{a[0-9]+}}, 2
; CHECK: cincoffset [[P1:a[0-9]+]], a0, [[OFF1]]
; CHECK: cincoffset [[P2:a[0-9]+]], a0, [[OFF2]]
define ptr addrspace(200) @test_two_geps(ptr addrspace(200) %arr, i64 %i, i64 %j) {
  %p1 = getelementptr i32, ptr addrspace(200) %arr, i64 %i
  %p2 = getelementptr i32, ptr addrspace(200) %arr, i64 %j
  ; Use both results so the compiler cannot CSE or eliminate one GEP.
  store i32 0, ptr addrspace(200) %p1
  store i32 0, ptr addrspace(200) %p2
  ret ptr addrspace(200) %p1
}

; Capability comes from a memory load (simulates spill-reload after a call).
; lowerADD must recognise ISD::LOAD i128 as a capability value so it is not
; placed in the integer-offset slot of cincoffset.
; CHECK-LABEL: test_cap_from_load:
; CHECK: ldc [[CAP:a[0-9]+]], 0(a0)
; CHECK: slli [[OFF:a[0-9]+]], a1, 2
; CHECK: cincoffset {{a[0-9]+}}, [[CAP]], [[OFF]]
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @test_cap_from_load(ptr %arr_ptr, i64 %i) {
  %arr = load ptr addrspace(200), ptr %arr_ptr
  %p = getelementptr i32, ptr addrspace(200) %arr, i64 %i
  ret ptr addrspace(200) %p
}

; Signed i32 offsets used in pointer arithmetic used to expose
; sign_extend_inreg(any_extend(i64), i32) and unselectable i128 nodes. The
; backend must normalize the integer offset back to xlen before cincoffset.
; CHECK-LABEL: test_signed_i32_gep:
; CHECK: cincoffset a0, a0, a1
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @test_signed_i32_gep(ptr addrspace(200) %p, i32 %idx) {
  %idx64 = sext i32 %idx to i64
  %q = getelementptr i8, ptr addrspace(200) %p, i64 %idx64
  ret ptr addrspace(200) %q
}

; CAST-128-style pointer offset: signed i32 length adjusted by a constant before
; indexing through a capability pointer. This is the source pattern that
; motivated the i128 sign_extend_inreg lowering fix.
; CHECK-LABEL: test_signed_i32_load_gep:
; CHECK: addi [[OFF:a[0-9]+]], a1, -4
; CHECK: cincoffset [[PTR:a[0-9]+]], a0, [[OFF]]
; CHECK: lbu a0, 0([[PTR]])
; CHECK: cjalr zero, 0(ra)
define i32 @test_signed_i32_load_gep(ptr addrspace(200) %key, i32 %length) {
  %sub = add nsw i32 %length, -4
  %idx = sext i32 %sub to i64
  %p = getelementptr i8, ptr addrspace(200) %key, i64 %idx
  %v = load i8, ptr addrspace(200) %p, align 1
  %ext = zext i8 %v to i32
  ret i32 %ext
}

; Pointer decrement can reach instruction selection as sub i128 after
; DAGCombine rewrites add(ptr, neg(offset)). Lower it as cincoffset with a
; negated xlen offset, not as scalar i128 subtraction.
; CHECK-LABEL: test_ptr_sub_i64:
; CHECK: neg [[NEG:a[0-9]+]], a1
; CHECK-NEXT: cincoffset a0, a0, [[NEG]]
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @test_ptr_sub_i64(ptr addrspace(200) %p, i64 %offset) {
  %base = ptrtoint ptr addrspace(200) %p to i128
  %wide = sext i64 %offset to i128
  %sub = sub i128 %base, %wide
  %q = inttoptr i128 %sub to ptr addrspace(200)
  ret ptr addrspace(200) %q
}

; The i32 form commonly appears as sign_extend_inreg(any_extend(...), i32)
; before selection. Keep the negate in xlen and feed cincoffset.
; CHECK-LABEL: test_ptr_sub_sext_i32:
; CHECK: neg [[NEG:a[0-9]+]], a1
; CHECK-NEXT: cincoffset a0, a0, [[NEG]]
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @test_ptr_sub_sext_i32(ptr addrspace(200) %p, i32 %offset) {
  %base = ptrtoint ptr addrspace(200) %p to i128
  %wide = sext i32 %offset to i128
  %sub = sub i128 %base, %wide
  %q = inttoptr i128 %sub to ptr addrspace(200)
  ret ptr addrspace(200) %q
}

; add(ptr, -offset) is the canonical source of this backend blocker. This IR
; shape should combine to the same sub-i128 lowering as an explicit pointer
; decrement.
; CHECK-LABEL: test_ptr_add_neg_i64:
; CHECK: neg [[NEG:a[0-9]+]], a1
; CHECK-NEXT: cincoffset a0, a0, [[NEG]]
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @test_ptr_add_neg_i64(ptr addrspace(200) %p, i64 %offset) {
  %base = ptrtoint ptr addrspace(200) %p to i128
  %wide = sext i64 %offset to i128
  %neg = sub i128 0, %wide
  %addr = add i128 %base, %neg
  %q = inttoptr i128 %addr to ptr addrspace(200)
  ret ptr addrspace(200) %q
}

; C pointer subtraction returns an integer difference between two capability
; cursors. Do not select this as full-width i128 scalar subtraction.
; The address reads are `mv` (PseudoTRUNC_CAP, i.e. addi rd, rs, 0) and NOT
; `lcc rd, rs, 2`, which is what this test pinned before 2026-08-15. lcc raises
; "Unexpected operand type (24)" on an untagged operand per
; capstone-spec/parts/cap-man-insn.adoc:164-168, so it made `NULL - NULL` fault --
; ordinary C, and what mruby's VM entry does (vm.c:1120). The subject of this test
; is unchanged by that: see cap-ptrdiff-untagged.ll.
; CHECK-LABEL: test_ptrdiff:
; CHECK-DAG: mv [[CUR0:a[0-9]+]], a0
; CHECK-DAG: mv [[CUR1:a[0-9]+]], a1
; CHECK: sub a0, [[CUR0]], [[CUR1]]
; CHECK: cjalr zero, 0(ra)
define i64 @test_ptrdiff(ptr addrspace(200) %p, ptr addrspace(200) %q) {
  %pi = ptrtoint ptr addrspace(200) %p to i128
  %qi = ptrtoint ptr addrspace(200) %q to i128
  %diff = sub i128 %pi, %qi
  %n = trunc i128 %diff to i64
  ret i64 %n
}

; InstCombine can express pointer addition by a known disjoint byte offset as
; `or disjoint` on the i128 pointer carrier. Select it as capability cursor
; arithmetic, not scalar integer OR.
; CHECK-LABEL: test_or_disjoint_frame_const:
; CHECK: cincoffsetimm a0, {{(sp|s0)}}, 0
; CHECK-NEXT: cincoffsetimm a0, a0, 8
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @test_or_disjoint_frame_const() {
entry:
  %buf = alloca [32 x i8], align 16, addrspace(200)
  %base = ptrtoint ptr addrspace(200) %buf to i128
  %off = or disjoint i128 %base, 8
  %ptr = inttoptr i128 %off to ptr addrspace(200)
  ret ptr addrspace(200) %ptr
}
