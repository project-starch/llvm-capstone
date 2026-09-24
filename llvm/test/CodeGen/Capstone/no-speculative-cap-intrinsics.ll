; C-58, the capability intrinsics: SCC, TIGHTEN, SEAL and INIT are side-effect
; free to LLVM, and on an untagged operand SEAL traps (UNEXPECTED_OPERAND) and
; the QEMU model of SCC, TIGHTEN and INIT stops on an assertion. Each is formed
; here in the guarded preheader of an inner loop, where only a non-null %p
; reaches it, and must stay there: hoisting it into `entry` executes it on the
; null path too. The control, the same SCC without the guard, is still hoisted.
; SEAL and INIT are selected as their tied pseudos, which is what MachineLICM
; sees.
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs -stop-after=early-machinelicm < %s | FileCheck %s
; RUN: %llc_cap -O2 < %s -o /dev/null

; CHECK-LABEL: name: guarded_scc
; CHECK: bb.{{[0-9]+}}.inner.ph:
; CHECK-NOT: {{^ *bb\.[0-9]+}}
; CHECK: SCC 
; CHECK-LABEL: name: guarded_tighten
; CHECK: bb.{{[0-9]+}}.inner.ph:
; CHECK-NOT: {{^ *bb\.[0-9]+}}
; CHECK: TIGHTEN 
; CHECK-LABEL: name: guarded_seal
; CHECK: bb.{{[0-9]+}}.inner.ph:
; CHECK-NOT: {{^ *bb\.[0-9]+}}
; CHECK: PseudoSEAL 
; CHECK-LABEL: name: guarded_init
; CHECK: bb.{{[0-9]+}}.inner.ph:
; CHECK-NOT: {{^ *bb\.[0-9]+}}
; CHECK: PseudoINIT 
; CHECK-LABEL: name: control_unguarded_scc
; CHECK: bb.0.entry:
; CHECK-NOT: {{^ *bb\.[0-9]+}}
; CHECK: SCC 

define void @guarded_scc(ptr addrspace(200) %p, i64 %n, i64 %m, ptr addrspace(200) %out) addrspace(200) {
entry:
  br label %outer
outer:
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  %isnull = icmp eq ptr addrspace(200) %p, null
  br i1 %isnull, label %outer.latch, label %inner.ph
inner.ph:
  %c = call ptr addrspace(200) @llvm.capstone.cap.scc.p200(ptr addrspace(200) %p, i64 64)
  br label %inner
inner:
  %i = phi i64 [ 0, %inner.ph ], [ %i.next, %inner ]
  store ptr addrspace(200) %c, ptr addrspace(200) %out, align 16
  %i.next = add i64 %i, 1
  %idone = icmp eq i64 %i.next, %n
  br i1 %idone, label %outer.latch, label %inner
outer.latch:
  %j.next = add i64 %j, 1
  %odone = icmp eq i64 %j.next, %m
  br i1 %odone, label %exit, label %outer
exit:
  ret void
}
define void @guarded_tighten(ptr addrspace(200) %p, i64 %n, i64 %m, ptr addrspace(200) %out) addrspace(200) {
entry:
  br label %outer
outer:
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  %isnull = icmp eq ptr addrspace(200) %p, null
  br i1 %isnull, label %outer.latch, label %inner.ph
inner.ph:
  %c = call ptr addrspace(200) @llvm.capstone.cap.tighten.p200(ptr addrspace(200) %p, i64 3)
  br label %inner
inner:
  %i = phi i64 [ 0, %inner.ph ], [ %i.next, %inner ]
  store ptr addrspace(200) %c, ptr addrspace(200) %out, align 16
  %i.next = add i64 %i, 1
  %idone = icmp eq i64 %i.next, %n
  br i1 %idone, label %outer.latch, label %inner
outer.latch:
  %j.next = add i64 %j, 1
  %odone = icmp eq i64 %j.next, %m
  br i1 %odone, label %exit, label %outer
exit:
  ret void
}
define void @guarded_seal(ptr addrspace(200) %p, i64 %n, i64 %m, ptr addrspace(200) %out) addrspace(200) {
entry:
  br label %outer
outer:
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  %isnull = icmp eq ptr addrspace(200) %p, null
  br i1 %isnull, label %outer.latch, label %inner.ph
inner.ph:
  %c = call ptr addrspace(200) @llvm.capstone.cap.seal.p200(ptr addrspace(200) %p)
  br label %inner
inner:
  %i = phi i64 [ 0, %inner.ph ], [ %i.next, %inner ]
  store ptr addrspace(200) %c, ptr addrspace(200) %out, align 16
  %i.next = add i64 %i, 1
  %idone = icmp eq i64 %i.next, %n
  br i1 %idone, label %outer.latch, label %inner
outer.latch:
  %j.next = add i64 %j, 1
  %odone = icmp eq i64 %j.next, %m
  br i1 %odone, label %exit, label %outer
exit:
  ret void
}
define void @guarded_init(ptr addrspace(200) %p, i64 %n, i64 %m, ptr addrspace(200) %out) addrspace(200) {
entry:
  br label %outer
outer:
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  %isnull = icmp eq ptr addrspace(200) %p, null
  br i1 %isnull, label %outer.latch, label %inner.ph
inner.ph:
  %c = call ptr addrspace(200) @llvm.capstone.cap.init.p200(ptr addrspace(200) %p, i64 64)
  br label %inner
inner:
  %i = phi i64 [ 0, %inner.ph ], [ %i.next, %inner ]
  store ptr addrspace(200) %c, ptr addrspace(200) %out, align 16
  %i.next = add i64 %i, 1
  %idone = icmp eq i64 %i.next, %n
  br i1 %idone, label %outer.latch, label %inner
outer.latch:
  %j.next = add i64 %j, 1
  %odone = icmp eq i64 %j.next, %m
  br i1 %odone, label %exit, label %outer
exit:
  ret void
}
define void @control_unguarded_scc(ptr addrspace(200) %p, i64 %n, i64 %m, ptr addrspace(200) %out) addrspace(200) {
entry:
  br label %outer
outer:
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  br label %inner.ph
inner.ph:
  %c = call ptr addrspace(200) @llvm.capstone.cap.scc.p200(ptr addrspace(200) %p, i64 64)
  br label %inner
inner:
  %i = phi i64 [ 0, %inner.ph ], [ %i.next, %inner ]
  store ptr addrspace(200) %c, ptr addrspace(200) %out, align 16
  %i.next = add i64 %i, 1
  %idone = icmp eq i64 %i.next, %n
  br i1 %idone, label %outer.latch, label %inner
outer.latch:
  %j.next = add i64 %j, 1
  %odone = icmp eq i64 %j.next, %m
  br i1 %odone, label %exit, label %outer
exit:
  ret void
}

declare ptr addrspace(200) @llvm.capstone.cap.scc.p200(ptr addrspace(200), i64)
declare ptr addrspace(200) @llvm.capstone.cap.tighten.p200(ptr addrspace(200), i64 immarg)
declare ptr addrspace(200) @llvm.capstone.cap.seal.p200(ptr addrspace(200))
declare ptr addrspace(200) @llvm.capstone.cap.init.p200(ptr addrspace(200), i64)
