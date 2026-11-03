; RUN: opt -mtriple=capstone64 -passes=loop-reduce -verify-each -S < %s | FileCheck %s --check-prefix=IR --implicit-check-not=ptrtoint --implicit-check-not=inttoptr
; RUN: llc -mtriple=capstone64 -mattr=+m,+f,+d -O2 -capstone-gp-free -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
;
; The same current-cell address is accessed on both arms. Reuse that cursor
; with an in-window step; keep the far neighbor derivations inside their arms.
; The counter starts at zero, so the preheader needs no cursor derivation.

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

; IR-LABEL: define void @diamond(
; IR: header:
; IR: [[CURSOR:%[^ ]+]] = phi ptr addrspace(200) [ %dst, %header.preheader ], [ [[NEXT:%[^ ]+]], %latch ]
; IR: left:
; IR: store double 1.000000e+00, ptr addrspace(200) [[CURSOR]]
; IR: getelementptr double, ptr addrspace(200) [[CURSOR]], i64 2001
; IR: right:
; IR: store double 3.000000e+00, ptr addrspace(200) [[CURSOR]]
; IR: getelementptr double, ptr addrspace(200) [[CURSOR]], i64 -1998
; IR: latch:
; IR: [[NEXT]] = getelementptr i8, ptr addrspace(200) [[CURSOR]], i64 160
; IR: ret void
;
; ASM-LABEL: diamond:
; ASM-NOT: slli {{.*}}, 3{{$}}
; ASM: cincoffsetimm {{[a-z0-9]+}}, {{[a-z0-9]+}}, 160
; ASM-NOT: slli {{.*}}, 3{{$}}
; ASM: ret
define void @diamond(ptr addrspace(200) %dst, ptr addrspace(200) %flags, i64 %n) {
entry:
  %enter = icmp ugt i64 %n, 0
  br i1 %enter, label %header, label %exit
header:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %fp = getelementptr inbounds i8, ptr addrspace(200) %flags, i64 %i
  %f = load i8, ptr addrspace(200) %fp
  %take = icmp eq i8 %f, 0
  br i1 %take, label %left, label %right
left:
  %a = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %i
  store double 1.0, ptr addrspace(200) %a
  %plus = add i64 %i, 2001
  %p = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %plus
  store double 2.0, ptr addrspace(200) %p
  br label %latch
right:
  %b = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %i
  store double 3.0, ptr addrspace(200) %b
  %minus = add i64 %i, -1998
  %q = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %minus
  store double 4.0, ptr addrspace(200) %q
  br label %latch
latch:
  %next = add nuw nsw i64 %i, 20
  %again = icmp ult i64 %next, %n
  br i1 %again, label %header, label %exit
exit:
  ret void
}

; A skip path does not access the cursor. Even a small step cannot authorize
; stepping a sealed or uninitialized base on the skipped iterations.
; IR-LABEL: define void @skipped(
; IR-NOT: phi ptr addrspace(200) [ %dst,
; IR: ret void
define void @skipped(ptr addrspace(200) %dst, ptr addrspace(200) %flags, i64 %n) {
entry:
  %enter = icmp ugt i64 %n, 0
  br i1 %enter, label %header, label %exit
header:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %fp = getelementptr inbounds i8, ptr addrspace(200) %flags, i64 %i
  %f = load i8, ptr addrspace(200) %fp
  %take = icmp eq i8 %f, 0
  br i1 %take, label %left, label %latch
left:
  %a = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %i
  store double 1.0, ptr addrspace(200) %a
  %plus = add i64 %i, 2001
  %p = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %plus
  store double 2.0, ptr addrspace(200) %p
  br label %latch
latch:
  %next = add nuw nsw i64 %i, 20
  %again = icmp ult i64 %next, %n
  br i1 %again, label %header, label %exit
exit:
  ret void
}

; Every consumed cursor is accessed; the final unused step may lose its tag.
; IR-LABEL: define void @large_step(
; IR: [[CURSOR:%[^ ]+]] = phi ptr addrspace(200) [ %dst, %header.preheader ], [ [[NEXT:%[^ ]+]], %latch ]
; IR: store double 1.000000e+00, ptr addrspace(200) [[CURSOR]]
; IR: store double 3.000000e+00, ptr addrspace(200) [[CURSOR]]
; IR: [[NEXT]] = getelementptr i8, ptr addrspace(200) [[CURSOR]], i64 2400
; IR: ret void
define void @large_step(ptr addrspace(200) %dst, ptr addrspace(200) %flags, i64 %n) {
entry:
  %enter = icmp ugt i64 %n, 0
  br i1 %enter, label %header, label %exit
header:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %fp = getelementptr inbounds i8, ptr addrspace(200) %flags, i64 %i
  %f = load i8, ptr addrspace(200) %fp
  %take = icmp eq i8 %f, 0
  br i1 %take, label %left, label %right
left:
  %a = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %i
  store double 1.0, ptr addrspace(200) %a
  %plus = add i64 %i, 2001
  %p = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %plus
  store double 2.0, ptr addrspace(200) %p
  br label %latch
right:
  %b = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %i
  store double 3.0, ptr addrspace(200) %b
  %minus = add i64 %i, -1998
  %q = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %minus
  store double 4.0, ptr addrspace(200) %q
  br label %latch
latch:
  %next = add nuw nsw i64 %i, 300
  %again = icmp ult i64 %next, %n
  br i1 %again, label %header, label %exit
exit:
  ret void
}

; A call before the anchors may exit without ever deriving the destination.
; IR-LABEL: define void @may_exit(
; IR-NOT: phi ptr addrspace(200) [ %dst,
; IR: ret void
define void @may_exit(ptr addrspace(200) %dst, ptr addrspace(200) %flags, i64 %n) {
entry:
  %enter = icmp ugt i64 %n, 0
  br i1 %enter, label %header, label %exit
header:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  call void @escape()
  %fp = getelementptr inbounds i8, ptr addrspace(200) %flags, i64 %i
  %f = load i8, ptr addrspace(200) %fp
  %take = icmp eq i8 %f, 0
  br i1 %take, label %left, label %right
left:
  %a = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %i
  store double 1.0, ptr addrspace(200) %a
  %plus = add i64 %i, 2001
  %p = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %plus
  store double 2.0, ptr addrspace(200) %p
  br label %latch
right:
  %b = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %i
  store double 3.0, ptr addrspace(200) %b
  %minus = add i64 %i, -1998
  %q = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %minus
  store double 4.0, ptr addrspace(200) %q
  br label %latch
latch:
  %next = add nuw nsw i64 %i, 20
  %again = icmp ult i64 %next, %n
  br i1 %again, label %header, label %exit
exit:
  ret void
}

; Similar SCEVs or nearby addresses do not substitute for the exact anchor.
; IR-LABEL: define void @different_anchor(
; IR-NOT: phi ptr addrspace(200) [ %dst,
; IR: ret void
define void @different_anchor(ptr addrspace(200) %dst, ptr addrspace(200) %flags, i64 %n) {
entry:
  %enter = icmp ugt i64 %n, 0
  br i1 %enter, label %header, label %exit
header:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %fp = getelementptr inbounds i8, ptr addrspace(200) %flags, i64 %i
  %f = load i8, ptr addrspace(200) %fp
  %take = icmp eq i8 %f, 0
  br i1 %take, label %left, label %right
left:
  %a = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %i
  store double 1.0, ptr addrspace(200) %a
  %plus = add i64 %i, 2001
  %p = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %plus
  store double 2.0, ptr addrspace(200) %p
  br label %latch
right:
  %other = add i64 %i, 1
  %b = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %other
  store double 3.0, ptr addrspace(200) %b
  %minus = add i64 %i, -1998
  %q = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %minus
  store double 4.0, ptr addrspace(200) %q
  br label %latch
latch:
  %next = add nuw nsw i64 %i, 20
  %again = icmp ult i64 %next, %n
  br i1 %again, label %header, label %exit
exit:
  ret void
}

; Keep the narrowed base, and leave the narrowing below the entry guard.
; IR-LABEL: define void @shrunk(
; IR: br i1 %enter, label %prepare, label %exit
; IR: prepare:
; IR: [[BOUNDED:%[^ ]+]] = call addrspace(200) ptr addrspace(200) @llvm.capstone.cap.shrink.p200
; IR: header:
; IR: phi ptr addrspace(200) [ [[BOUNDED]], %prepare ], [ %{{[^ ]+}}, %latch ]
; IR: ret void
define void @shrunk(ptr addrspace(200) %dst, ptr addrspace(200) %flags, i64 %n) {
entry:
  %enter = icmp ugt i64 %n, 0
  br i1 %enter, label %prepare, label %exit
prepare:
  %bounded = call ptr addrspace(200) @llvm.capstone.cap.shrink.p200(ptr addrspace(200) %dst, i64 0, i64 65536)
  br label %header
header:
  %i = phi i64 [ 0, %prepare ], [ %next, %latch ]
  %fp = getelementptr inbounds i8, ptr addrspace(200) %flags, i64 %i
  %f = load i8, ptr addrspace(200) %fp
  %take = icmp eq i8 %f, 0
  br i1 %take, label %left, label %right
left:
  %a = getelementptr inbounds double, ptr addrspace(200) %bounded, i64 %i
  store double 1.0, ptr addrspace(200) %a
  %plus = add i64 %i, 2001
  %p = getelementptr inbounds double, ptr addrspace(200) %bounded, i64 %plus
  store double 2.0, ptr addrspace(200) %p
  br label %latch
right:
  %b = getelementptr inbounds double, ptr addrspace(200) %bounded, i64 %i
  store double 3.0, ptr addrspace(200) %b
  %minus = add i64 %i, -1998
  %q = getelementptr inbounds double, ptr addrspace(200) %bounded, i64 %minus
  store double 4.0, ptr addrspace(200) %q
  br label %latch
latch:
  %next = add nuw nsw i64 %i, 20
  %again = icmp ult i64 %next, %n
  br i1 %again, label %header, label %exit
exit:
  ret void
}

; Index-width modular arithmetic preserves the address even without nsw.
; The inbounds accessed anchor proves each consumed cursor remains valid.
; IR-LABEL: define void @unsigned_wrap(
; IR: [[CURSOR:%[^ ]+]] = phi ptr addrspace(200) [ %dst, %header.preheader ], [ [[NEXT:%[^ ]+]], %latch ]
; IR: store double 1.000000e+00, ptr addrspace(200) [[CURSOR]]
; IR: store double 3.000000e+00, ptr addrspace(200) [[CURSOR]]
; IR: [[NEXT]] = getelementptr i8, ptr addrspace(200) [[CURSOR]], i64 160
; IR: ret void
define void @unsigned_wrap(ptr addrspace(200) %dst, ptr addrspace(200) %flags, i64 %n) {
entry:
  %enter = icmp ugt i64 %n, 0
  br i1 %enter, label %header, label %exit
header:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %fp = getelementptr inbounds i8, ptr addrspace(200) %flags, i64 %i
  %f = load i8, ptr addrspace(200) %fp
  %take = icmp eq i8 %f, 0
  br i1 %take, label %left, label %right
left:
  %a = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %i
  store double 1.0, ptr addrspace(200) %a
  %plus = add i64 %i, 2001
  %p = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %plus
  store double 2.0, ptr addrspace(200) %p
  br label %latch
right:
  %b = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %i
  store double 3.0, ptr addrspace(200) %b
  %minus = add i64 %i, -1998
  %q = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %minus
  store double 4.0, ptr addrspace(200) %q
  br label %latch
latch:
  %next = add nuw i64 %i, 20
  %again = icmp ult i64 %next, %n
  br i1 %again, label %header, label %exit
exit:
  ret void
}

; An explicit exit edge before the anchors is an uncovered successor of the
; header: an iteration may leave the loop without accessing the cursor, so the
; backwards proof never reaches the header and no cursor is formed.
; IR-LABEL: define void @early_exit(
; IR-NOT: phi ptr addrspace(200) [ %dst,
; IR: ret void
define void @early_exit(ptr addrspace(200) %dst, ptr addrspace(200) %flags, i64 %n) {
entry:
  %enter = icmp ugt i64 %n, 0
  br i1 %enter, label %header, label %exit
header:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %fp = getelementptr inbounds i8, ptr addrspace(200) %flags, i64 %i
  %f = load i8, ptr addrspace(200) %fp
  %stop = icmp eq i8 %f, 2
  br i1 %stop, label %exit, label %select
select:
  %take = icmp eq i8 %f, 0
  br i1 %take, label %left, label %right
left:
  %a = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %i
  store double 1.0, ptr addrspace(200) %a
  %plus = add i64 %i, 2001
  %p = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %plus
  store double 2.0, ptr addrspace(200) %p
  br label %latch
right:
  %b = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %i
  store double 3.0, ptr addrspace(200) %b
  %minus = add i64 %i, -1998
  %q = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %minus
  store double 4.0, ptr addrspace(200) %q
  br label %latch
latch:
  %next = add nuw nsw i64 %i, 20
  %again = icmp ult i64 %next, %n
  br i1 %again, label %header, label %exit
exit:
  ret void
}

; A far neighbor from either arm is live out of the loop. The reuse applies,
; and the live-out value is still the residual derivation from the cursor,
; whose only source is the original base: no integer reconstruction, and in
; assembly both candidates are cincoffset from the register the step advances.
; IR-LABEL: define ptr addrspace(200) @live_out(
; IR: header:
; IR: [[CURSOR:%[^ ]+]] = phi ptr addrspace(200) [ %dst, %header.preheader ], [ [[NEXT:%[^ ]+]], %latch ]
; IR: left:
; IR: store double 1.000000e+00, ptr addrspace(200) [[CURSOR]]
; IR: [[P:%[^ ]+]] = getelementptr double, ptr addrspace(200) [[CURSOR]], i64 2001
; IR: right:
; IR: store double 3.000000e+00, ptr addrspace(200) [[CURSOR]]
; IR: [[Q:%[^ ]+]] = getelementptr double, ptr addrspace(200) [[CURSOR]], i64 -1998
; IR: latch:
; IR-NEXT: [[SEL:%[^ ]+]] = phi ptr addrspace(200) [ [[P]], %left ], [ [[Q]], %right ]
; IR: [[NEXT]] = getelementptr i8, ptr addrspace(200) [[CURSOR]], i64 160
; IR: exit.loopexit:
; IR-NEXT: [[LCSSA:%[^ ]+]] = phi ptr addrspace(200) [ [[SEL]], %latch ]
; IR: exit:
; IR-NEXT: [[RES:%[^ ]+]] = phi ptr addrspace(200) [ null, %entry ], [ [[LCSSA]], %exit.loopexit ]
; IR-NEXT: ret ptr addrspace(200) [[RES]]
;
; ASM-LABEL: live_out:
; ASM-NOT: slli {{.*}}, 3{{$}}
; ASM-DAG: cincoffsetimm [[CUR:[a-z0-9]+]], [[CUR]], 160
; ASM-DAG: cincoffset [[P:[a-z0-9]+]], [[CUR]], {{[a-z0-9]+}}
; ASM-DAG: cincoffset [[Q:[a-z0-9]+]], [[CUR]], {{[a-z0-9]+}}
; ASM-DAG: sd {{[a-z0-9]+}}, 0([[CUR]])
; ASM-DAG: sd {{[a-z0-9]+}}, 0([[P]])
; ASM-DAG: sd {{[a-z0-9]+}}, 0([[Q]])
; ASM-NOT: slli {{.*}}, 3{{$}}
; ASM: movc a0, [[P]]
; ASM-NEXT: ret
define ptr addrspace(200) @live_out(ptr addrspace(200) %dst, ptr addrspace(200) %flags, i64 %n) {
entry:
  %enter = icmp ugt i64 %n, 0
  br i1 %enter, label %header, label %exit
header:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %fp = getelementptr inbounds i8, ptr addrspace(200) %flags, i64 %i
  %f = load i8, ptr addrspace(200) %fp
  %take = icmp eq i8 %f, 0
  br i1 %take, label %left, label %right
left:
  %a = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %i
  store double 1.0, ptr addrspace(200) %a
  %plus = add i64 %i, 2001
  %p = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %plus
  store double 2.0, ptr addrspace(200) %p
  br label %latch
right:
  %b = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %i
  store double 3.0, ptr addrspace(200) %b
  %minus = add i64 %i, -1998
  %q = getelementptr inbounds double, ptr addrspace(200) %dst, i64 %minus
  store double 4.0, ptr addrspace(200) %q
  br label %latch
latch:
  %sel = phi ptr addrspace(200) [ %p, %left ], [ %q, %right ]
  %next = add nuw nsw i64 %i, 20
  %again = icmp ult i64 %next, %n
  br i1 %again, label %header, label %exit
exit:
  %res = phi ptr addrspace(200) [ null, %entry ], [ %sel, %latch ]
  ret ptr addrspace(200) %res
}

declare void @escape()
declare ptr addrspace(200) @llvm.capstone.cap.shrink.p200(ptr addrspace(200), i64, i64)
