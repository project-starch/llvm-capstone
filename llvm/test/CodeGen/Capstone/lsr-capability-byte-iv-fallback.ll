; RUN: opt -mtriple=capstone64 -passes=loop-reduce -verify-each -S < %s | FileCheck %s --check-prefix=DEFAULT
; RUN: opt -mtriple=capstone64 -passes=loop-reduce -lsr-capability-byte-iv-fallback -verify-each -S < %s | FileCheck %s --check-prefix=FALLBACK --implicit-check-not=ptrtoint --implicit-check-not=inttoptr
; RUN: opt -mtriple=capstone64 -passes=loop-reduce -lsr-capability-byte-iv -lsr-capability-byte-iv-fallback -verify-each -S < %s | FileCheck %s --check-prefix=BOTH
; RUN: opt -mtriple=capstone64 -passes=loop-reduce -lsr-capability-byte-iv-fallback -lsr-capability-group-log=- -disable-output < %s 2>&1 | FileCheck %s --check-prefix=LOG
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -lsr-capability-byte-iv-fallback -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
;
; -lsr-capability-byte-iv-fallback chooses the form per address group. A group
; whose anchor is accessed on every iteration keeps the capability recurrence
; (the default); a group whose proof fails gets the shared integer byte IV of
; -lsr-capability-byte-iv instead of no reduction. One loop here holds both, on
; the same index: %flags is loaded through its anchor in the header on every
; iteration; %a is stored only under a test, and its unconditional store goes
; through a constant-offset GEP of the anchor, which the proof does not accept
; (perl's S_regcppop). The option makes two passes: the default reduction,
; then the byte form for what it left. So %flags gets exactly the default's
; recurrence (from %flags, step 1), %a gets the byte IV (from 0, step 24), and
; every %a address is a GEP from %a itself. With -lsr-capability-byte-iv also
; given, every group gets the byte form, as with that flag alone.
;
; @order and @absorb are the claim audit's witnesses (2026-11-15) against a
; one-pass version. In @order the family's first GEP fails the proof and its
; second passes: the family keeps the default's recurrence (cap 1 byte 0), not
; the byte form. In @absorb a byte-form i32 GEP must not be taken over by the
; later anchored i8 group: its store stays a GEP from %p on the byte IV, never
; from the i8 group's recurrence.
;
; MUTATION (2026-11-15): with the fallback not taken (the opt run without the
; flag), the FALLBACK and LOG checks fail: %a is not reduced, the log says byte 0.

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

%pair = type { i64, i64, i64 }

; DEFAULT-LABEL: define void @mixed(
; DEFAULT: header:
; DEFAULT: phi ptr addrspace(200) [ %flags,
; DEFAULT-NOT: lsr.byte.iv
; DEFAULT: getelementptr inbounds nuw %pair, ptr addrspace(200) %a, i64 %i
; DEFAULT: ret void
;
; FALLBACK-LABEL: define void @mixed(
; FALLBACK: header:
; FALLBACK-DAG: [[CAP:%lsr.cap.iv[^ ]*]] = phi ptr addrspace(200) [ %flags, %{{[^ ]+}} ], [ [[CNEXT:%[^ ]+]], %latch ]
; FALLBACK-DAG: [[IV:%lsr.byte.iv[^ ]*]] = phi i64 [ 0, %{{[^ ]+}} ], [ [[NEXT:%[^ ]+]], %latch ]
; FALLBACK: load i8, ptr addrspace(200) [[CAP]]
; FALLBACK: start:
; FALLBACK: [[P:%[^ ]+]] = getelementptr i8, ptr addrspace(200) %a, i64 [[IV]]
; FALLBACK: store i64 -1, ptr addrspace(200) [[P]]
; FALLBACK: latch:
; FALLBACK: [[Q:%[^ ]+]] = getelementptr i8, ptr addrspace(200) %a, i64 [[IV]]
; FALLBACK: getelementptr inbounds nuw i8, ptr addrspace(200) [[Q]], i64 8
; FALLBACK-DAG: [[CNEXT]] = getelementptr i8, ptr addrspace(200) [[CAP]], i64 1
; FALLBACK-DAG: [[NEXT]] = add i64 [[IV]], 24
; FALLBACK: ret void
;
; BOTH-LABEL: define void @mixed(
; BOTH-NOT: phi ptr addrspace(200)
; BOTH: lsr.byte.iv
; BOTH-NOT: phi ptr addrspace(200)
; BOTH: ret void
;
; LOG: lsr-capability-groups: {{.*}} mixed header cap 1 byte 1
;
; ASM-LABEL: mixed:
; ASM-NOT: slli
; ASM: cjalr zero, 0(ra)
define void @mixed(ptr addrspace(200) %flags, ptr addrspace(200) %a, i64 %n, i64 %max) {
entry:
  %enter = icmp ugt i64 %n, 0
  br i1 %enter, label %header, label %exit
header:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %fp = getelementptr inbounds i8, ptr addrspace(200) %flags, i64 %i
  %f = load i8, ptr addrspace(200) %fp
  %take = icmp ugt i64 %i, %max
  br i1 %take, label %start, label %latch
start:
  %pa = getelementptr inbounds nuw %pair, ptr addrspace(200) %a, i64 %i
  store i64 -1, ptr addrspace(200) %pa
  br label %latch
latch:
  %qa = getelementptr inbounds nuw %pair, ptr addrspace(200) %a, i64 %i
  %end = getelementptr inbounds nuw i8, ptr addrspace(200) %qa, i64 8
  %fv = zext i8 %f to i64
  store i64 %fv, ptr addrspace(200) %end
  %next = add nuw nsw i64 %i, 1
  %again = icmp ult i64 %next, %n
  br i1 %again, label %header, label %exit
exit:
  ret void
}

; FALLBACK-LABEL: define void @order(
; FALLBACK: [[OC:%lsr.cap.iv[^ ]*]] = phi ptr addrspace(200) [ %p,
; FALLBACK-NOT: lsr.byte.iv
; FALLBACK: load i32, ptr addrspace(200) [[OC]]
; FALLBACK: ret void
; LOG: lsr-capability-groups: {{.*}} order header cap 1 byte 0
define void @order(ptr addrspace(200) %p, i64 %n, i64 %max) {
entry:
  %enter = icmp ugt i64 %n, 0
  br i1 %enter, label %header, label %exit
header:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %i1 = add i64 %i, 1
  %g1 = getelementptr inbounds i32, ptr addrspace(200) %p, i64 %i1
  %g0 = getelementptr inbounds i32, ptr addrspace(200) %p, i64 %i
  %v = load i32, ptr addrspace(200) %g0
  %take = icmp ugt i64 %i, %max
  br i1 %take, label %cond, label %latch
cond:
  store i32 %v, ptr addrspace(200) %g1
  br label %latch
latch:
  %next = add nuw nsw i64 %i, 1
  %again = icmp ult i64 %next, %n
  br i1 %again, label %header, label %exit
exit:
  ret void
}

; FALLBACK-LABEL: define void @absorb(
; FALLBACK-DAG: [[AC:%lsr.cap.iv[^ ]*]] = phi ptr addrspace(200) [ %p,
; FALLBACK-DAG: [[AB:%lsr.byte.iv[^ ]*]] = phi i64 [ 0,
; FALLBACK-DAG: [[AG:%[^ ]+]] = getelementptr i8, ptr addrspace(200) %p, i64 [[AB]]
; FALLBACK: load i8, ptr addrspace(200) [[AC]]
; FALLBACK: store i32 %w, ptr addrspace(200) [[AG]]
; FALLBACK: ret void
; LOG: lsr-capability-groups: {{.*}} absorb header cap 1 byte 1
define void @absorb(ptr addrspace(200) %p, i64 %n, i64 %max) {
entry:
  %enter = icmp ugt i64 %n, 0
  br i1 %enter, label %header, label %exit
header:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %ga = getelementptr inbounds i32, ptr addrspace(200) %p, i64 %i
  %gb = getelementptr inbounds i8, ptr addrspace(200) %p, i64 %i
  %v = load i8, ptr addrspace(200) %gb
  %take = icmp ugt i64 %i, %max
  br i1 %take, label %cond, label %latch
cond:
  %w = zext i8 %v to i32
  store i32 %w, ptr addrspace(200) %ga
  br label %latch
latch:
  %next = add nuw nsw i64 %i, 1
  %again = icmp ult i64 %next, %n
  br i1 %again, label %header, label %exit
exit:
  ret void
}
