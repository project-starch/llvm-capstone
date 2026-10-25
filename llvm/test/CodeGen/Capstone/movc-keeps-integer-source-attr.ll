; Q-04, decided 2026-10-14: MOVC keeps an untagged source and a NONLIN source, and consumes every
; other tagged type. The virtual-profile SDK therefore compiles with +movc-keeps-integer-source,
; passed as `-Xclang -target-feature -Xclang +movc-keeps-integer-source`. Clang turns that into the
; function's "target-features" attribute, not into -mattr, and the subtarget is chosen per function.
; This test checks that path.
;
; Each function reads its pointer again after the copy that passes it, inside a loop. Without the
; feature (@loop_rule under the first RUN line), CapstoneLiveSourceCopy rewrites that copy as
; STC+LDC through a stack slot in every iteration. With the feature in the function's attributes
; (@loop_keep), the copy stays one MOVC and the function gets no slot. Each function's checks fail
; on the other's output: deleting attribute #1 fails the loop_keep checks.
;
; RUN: llc -mtriple=capstone64 -O2 -verify-machineinstrs < %s | FileCheck %s
; Given as -mattr, the feature reaches @loop_rule, which has no "target-features" attribute of its
; own (a function's attribute replaces -mattr; it does not add to it).
; RUN: llc -mtriple=capstone64 -mattr=+m,+movc-keeps-integer-source -O2 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=MATTR

declare void @g(ptr addrspace(200)) addrspace(200)

; The rule: the slot adds 16 bytes to the frame, and every iteration stores and reloads.
; CHECK-LABEL: loop_rule:
; CHECK: cincoffsetimm sp, sp, -96
; CHECK: movc [[SRC:s[0-9]+]], a0
; CHECK: .LBB0_[[LOOP:[0-9]+]]: # %loop
; CHECK-NEXT: # =>This Inner Loop Header
; CHECK-NEXT: stc [[SRC]], [[S:[0-9]+\(sp\)]]
; CHECK-NEXT: ldc a0, [[S]]
; CHECK-NEXT: cjalr ra
; CHECK: bltu {{.*}}, .LBB0_[[LOOP]]
; MATTR-LABEL: loop_rule:
; MATTR: cincoffsetimm sp, sp, -80
; MATTR: movc [[SRC:s[0-9]+]], a0
; MATTR: # =>This Inner Loop Header
; MATTR-NEXT: movc a0, [[SRC]]
; MATTR-NEXT: cjalr ra
; MATTR: .Lfunc_end0:
define void @loop_rule(ptr addrspace(200) %p, i64 %n) addrspace(200) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  call addrspace(200) void @g(ptr addrspace(200) %p)
  %i1 = add i64 %i, 1
  %c = icmp ult i64 %i1, %n
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

; The feature: no slot, and the copy in the loop is one MOVC.
; CHECK-LABEL: loop_keep:
; CHECK: cincoffsetimm sp, sp, -80
; CHECK: movc [[SRC:s[0-9]+]], a0
; CHECK: .LBB1_[[LOOP:[0-9]+]]: # %loop
; CHECK-NEXT: # =>This Inner Loop Header
; CHECK-NEXT: movc a0, [[SRC]]
; CHECK-NEXT: cjalr ra
; CHECK-NOT: ldc a0
; CHECK: bltu {{.*}}, .LBB1_[[LOOP]]
; CHECK: .Lfunc_end1:
define void @loop_keep(ptr addrspace(200) %p, i64 %n) addrspace(200) #1 {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  call addrspace(200) void @g(ptr addrspace(200) %p)
  %i1 = add i64 %i, 1
  %c = icmp ult i64 %i1, %n
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

attributes #1 = { "target-features"="+m,+movc-keeps-integer-source" }
