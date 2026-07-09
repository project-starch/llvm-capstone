; RUN: opt -O2 -S < %s | FileCheck %s --check-prefix=IR
; RUN: llc -mtriple=capstone64 < %s | FileCheck %s --check-prefix=ASM

; MREV and DELIN mutate the revocation tree, so they were wrongly modelled as
; IntrNoMem. MREV allocates a node senior to its source and increments the
; source's depth: every MREV must produce a distinct node, so two MREVs of the
; same capability must not be CSE'd into one, and an unused MREV must not be
; eliminated. DELIN clears the node's `linear` flag, which decides whether a
; later REVOKE yields LIN (data retained) or UNINIT.
;
; The two enforcement points are independent: `Const`/IntrNoMem let the IR
; optimizer CSE the calls, and `hasSideEffects = 0` on the MachineInstr let
; machine DCE delete an unused one. Hence both run lines.

declare ptr addrspace(200) @llvm.capstone.cap.mrev.p200(ptr addrspace(200))
declare ptr addrspace(200) @llvm.capstone.cap.delin.p200(ptr addrspace(200))
declare void @sink(ptr addrspace(200), ptr addrspace(200))

; The intrinsics read and write hidden state, and nothing else. (The matching
; `attributes` line is asserted at the end of the file, where opt prints it.)
; IR: declare ptr addrspace(200) @llvm.capstone.cap.mrev.p200({{.*}}) #[[ATTR:[0-9]+]]

; An MREV whose result is unused still allocates a revocation node.
define void @mrev_unused(ptr addrspace(200) %p) {
; IR-LABEL: @mrev_unused
; IR: call ptr addrspace(200) @llvm.capstone.cap.mrev.p200
;
; ASM-LABEL: mrev_unused:
; ASM: mrev
; The destination must never be x0: helper_csmrev writes rd unconditionally, so
; allocating the hardwired-zero register would clobber it.
; ASM-NOT: mrev zero
  %r = call ptr addrspace(200) @llvm.capstone.cap.mrev.p200(ptr addrspace(200) %p)
  ret void
}

; Two MREVs of the same capability yield two nodes at two depths. Collapsing
; them into one would hand back a single revocation handle where the source
; asked for two.
define void @mrev_twice(ptr addrspace(200) %p) {
; IR-LABEL: @mrev_twice
; IR-COUNT-2: call ptr addrspace(200) @llvm.capstone.cap.mrev.p200
; IR-NOT: call ptr addrspace(200) @llvm.capstone.cap.mrev.p200
;
; ASM-LABEL: mrev_twice:
; ASM-COUNT-2: mrev
  %a = call ptr addrspace(200) @llvm.capstone.cap.mrev.p200(ptr addrspace(200) %p)
  %b = call ptr addrspace(200) @llvm.capstone.cap.mrev.p200(ptr addrspace(200) %p)
  call void @sink(ptr addrspace(200) %a, ptr addrspace(200) %b)
  ret void
}

; A DELIN whose result is unused still delinearises the node.
define void @delin_unused(ptr addrspace(200) %p) {
; IR-LABEL: @delin_unused
; IR: call ptr addrspace(200) @llvm.capstone.cap.delin.p200
;
; ASM-LABEL: delin_unused:
; ASM: delin
  %r = call ptr addrspace(200) @llvm.capstone.cap.delin.p200(ptr addrspace(200) %p)
  ret void
}

; Two DELINs of the same capability must likewise both survive.
define void @delin_twice(ptr addrspace(200) %p) {
; IR-LABEL: @delin_twice
; IR-COUNT-2: call ptr addrspace(200) @llvm.capstone.cap.delin.p200
; IR-NOT: call ptr addrspace(200) @llvm.capstone.cap.delin.p200
;
; ASM-LABEL: delin_twice:
; ASM-COUNT-2: delin
  %a = call ptr addrspace(200) @llvm.capstone.cap.delin.p200(ptr addrspace(200) %p)
  %b = call ptr addrspace(200) @llvm.capstone.cap.delin.p200(ptr addrspace(200) %p)
  call void @sink(ptr addrspace(200) %a, ptr addrspace(200) %b)
  ret void
}

; MREV must not be hoisted out of a loop: each iteration mints its own node.
define void @mrev_in_loop(ptr addrspace(200) %p, i64 %n) {
; IR-LABEL: @mrev_in_loop
; IR: call ptr addrspace(200) @llvm.capstone.cap.mrev.p200
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %r = call ptr addrspace(200) @llvm.capstone.cap.mrev.p200(ptr addrspace(200) %p)
  call void @sink(ptr addrspace(200) %r, ptr addrspace(200) %p)
  %i.next = add i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; IR: attributes #[[ATTR]] = { nounwind memory(inaccessiblemem: readwrite) }
