; RUN: opt -mtriple=capstone64 -passes=loop-reduce -verify-each -S < %s | FileCheck %s --check-prefix=CURSOR
; RUN: opt -mtriple=capstone64 -passes=loop-reduce -lsr-capability-byte-iv -verify-each -S < %s | FileCheck %s --check-prefix=BYTE --implicit-check-not=ptrtoint --implicit-check-not=inttoptr
; RUN: llc -mtriple=capstone64 -mattr=+m,+f,+d -O2 -lsr-capability-byte-iv -capstone-gp-free -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
;
; Conditional accesses to two bases share one integer byte IV. No capability
; derivation moves above the access guard, and no neighbor capability is
; precomputed. The integer step need not fit the capability window.

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

; CURSOR-LABEL: define void @shared_byte_iv(
; CURSOR-NOT: lsr.cap.iv
; CURSOR-NOT: lsr.byte.iv
; CURSOR: ret void
;
; BYTE-LABEL: define void @shared_byte_iv(
; BYTE: header:
; BYTE: [[IV:%lsr.byte.iv[^ ]*]] = phi i64 [ 0, %header.preheader ], [ [[NEXT:%[^ ]+]], %latch ]
; BYTE-NOT: phi
; BYTE: br i1 %take, label %access, label %latch
; BYTE: access:
; BYTE: getelementptr i8, ptr addrspace(200) %a, i64 [[IV]]
; BYTE: [[PLUS:%[^ ]+]] = add i64 [[IV]], 16008
; BYTE: getelementptr i8, ptr addrspace(200) %a, i64 [[PLUS]]
; BYTE: getelementptr i8, ptr addrspace(200) %b, i64 [[IV]]
; BYTE: [[MINUS:%[^ ]+]] = add i64 [[IV]], -15984
; BYTE: getelementptr i8, ptr addrspace(200) %b, i64 [[MINUS]]
; BYTE: latch:
; BYTE: [[NEXT]] = add i64 [[IV]], 2400
; BYTE: ret void
;
; ASM-LABEL: shared_byte_iv:
; ASM-NOT: slli {{.*}}, 3{{$}}
; ASM: cincoffset
; ASM-NOT: slli {{.*}}, 3{{$}}
; ASM: ret
define void @shared_byte_iv(ptr addrspace(200) %a, ptr addrspace(200) %b, ptr addrspace(200) %flags, i64 %n, double %v) {
entry:
  %enter = icmp ugt i64 %n, 0
  br i1 %enter, label %header, label %exit
header:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %fp = getelementptr inbounds i8, ptr addrspace(200) %flags, i64 %i
  %f = load i8, ptr addrspace(200) %fp
  %take = icmp eq i8 %f, 0
  br i1 %take, label %access, label %latch
access:
  %pa = getelementptr inbounds double, ptr addrspace(200) %a, i64 %i
  store double %v, ptr addrspace(200) %pa
  %plus = add i64 %i, 2001
  %qa = getelementptr inbounds double, ptr addrspace(200) %a, i64 %plus
  store double %v, ptr addrspace(200) %qa
  %pb = getelementptr inbounds double, ptr addrspace(200) %b, i64 %i
  store double %v, ptr addrspace(200) %pb
  %minus = add i64 %i, -1998
  %qb = getelementptr inbounds double, ptr addrspace(200) %b, i64 %minus
  store double %v, ptr addrspace(200) %qb
  br label %latch
latch:
  %next = add nuw nsw i64 %i, 300
  %again = icmp ult i64 %next, %n
  br i1 %again, label %header, label %exit
exit:
  ret void
}
