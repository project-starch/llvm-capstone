; RUN: opt -mtriple=capstone64 -passes=loop-reduce -verify-each -S %s | FileCheck %s --implicit-check-not=ptrtoint --implicit-check-not=inttoptr
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs -O2 < %s | FileCheck %s --check-prefix=ASM
;
; A single inbounds access per iteration is sufficient. The start is the
; original iteration-zero cursor, not a normalized cursor one step earlier.
; Both signs and invariant starts are valid. These checks complement the
; skipped-iteration and phys56 witnesses in cursor-boundaries.ll.

target datalayout = "e-p:64:64-p200:128:128:128:64-i64:64-n32:64-ni:200"

; CHECK-LABEL: @single(
; CHECK: [[IV:%.*]] = phi ptr addrspace(200) [ %b, %entry ], [ [[NEXT:%.*]], %loop ]
; CHECK: load volatile i64, ptr addrspace(200) [[IV]]
; CHECK: [[NEXT]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 8
; ASM-LABEL: single:
; ASM-NOT: slli
; ASM: ld zero, 0([[P:[a-z0-9]+]])
; ASM: cincoffsetimm [[P]], [[P]], 8
; ASM-NOT: slli
; ASM: cjalr
define void @single(ptr addrspace(200) %b, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%next, %loop]
  %p = getelementptr inbounds i64, ptr addrspace(200) %b, i64 %i
  %v = load volatile i64, ptr addrspace(200) %p
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; CHECK-LABEL: @descending(
; CHECK: entry:
; CHECK: [[START:%.*]] = getelementptr i64, ptr addrspace(200) %b, i64 %start
; CHECK: loop:
; CHECK: [[IV:%.*]] = phi ptr addrspace(200) [ [[START]], %entry ], [ [[NEXT:%.*]], %loop ]
; CHECK: load volatile i64, ptr addrspace(200) [[IV]]
; CHECK: [[NEXT]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 -8
; ASM-LABEL: descending:
; ASM: slli
; ASM: cincoffset
; ASM: [[LOOP:.LBB[0-9_]+]]:
; ASM-NOT: slli
; ASM: ld zero, 0([[P:[a-z0-9]+]])
; ASM: cincoffsetimm [[P]], [[P]], -8
; ASM: bne {{.*}}, [[LOOP]]
define void @descending(ptr addrspace(200) %b, i64 %start) {
entry:
  br label %loop
loop:
  %i = phi i64 [%start, %entry], [%next, %loop]
  %p = getelementptr inbounds i64, ptr addrspace(200) %b, i64 %i
  %v = load volatile i64, ptr addrspace(200) %p
  %next = add nsw i64 %i, -1
  %done = icmp eq i64 %i, 0
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; A k+j destination needs its own iteration-zero anchor, while the source
; starts at zero. Preserve the exact base for both arrays.
; CHECK-LABEL: @copy_offset(
; CHECK: entry:
; CHECK: [[START:%.*]] = getelementptr i32, ptr addrspace(200) %dst, i64 %k
; CHECK: loop:
; CHECK-DAG: [[SRC:%.*]] = phi ptr addrspace(200) [ %src, %entry ], [ [[SNEXT:%.*]], %loop ]
; CHECK-DAG: [[DST:%.*]] = phi ptr addrspace(200) [ [[START]], %entry ], [ [[DNEXT:%.*]], %loop ]
; CHECK: load i32, ptr addrspace(200) [[SRC]]
; CHECK: store i32 {{.*}}, ptr addrspace(200) [[DST]]
; CHECK-DAG: [[SNEXT]] = getelementptr i8, ptr addrspace(200) [[SRC]], i64 4
; CHECK-DAG: [[DNEXT]] = getelementptr i8, ptr addrspace(200) [[DST]], i64 4
; ASM-LABEL: copy_offset:
; ASM: [[LOOP:.LBB[0-9_]+]]:
; ASM-NOT: slli
; ASM: lw
; ASM-DAG: sw
; ASM-DAG: cincoffsetimm
; ASM-DAG: cincoffsetimm
; ASM-NOT: slli
; ASM: bne {{.*}}, [[LOOP]]
define void @copy_offset(ptr addrspace(200) %src, ptr addrspace(200) %dst, i64 %k, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%next, %loop]
  %sp = getelementptr inbounds i32, ptr addrspace(200) %src, i64 %i
  %v = load i32, ptr addrspace(200) %sp
  %ki = add i64 %k, %i
  %dp = getelementptr inbounds i32, ptr addrspace(200) %dst, i64 %ki
  store i32 %v, ptr addrspace(200) %dp
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; The extension of a wrapping narrow counter is not an affine 64-bit index.
; With b=2^31 and start=INT32_MAX, the two byte cursors are 2^32-1 and 0.
; Stepping the first cursor by one would access 2^32 instead.
; CHECK-LABEL: @narrow_wrap(
; CHECK-NOT: phi ptr addrspace(200)
; CHECK: sext i32
; CHECK: getelementptr inbounds i8, ptr addrspace(200) %b,
; CHECK-NOT: phi ptr addrspace(200)
; CHECK: ret void
define void @narrow_wrap(ptr addrspace(200) %b, i32 %start) {
entry:
  br label %loop
loop:
  %i = phi i32 [%start, %entry], [%next, %loop]
  %first = phi i1 [true, %entry], [false, %loop]
  %idx = sext i32 %i to i64
  %p = getelementptr inbounds i8, ptr addrspace(200) %b, i64 %idx
  %v = load volatile i8, ptr addrspace(200) %p
  %next = add i32 %i, 1
  br i1 %first, label %loop, label %exit
exit:
  ret void
}
