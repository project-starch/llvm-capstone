; C-66: MachineCSE's PRE must not move capability arithmetic above the test
; that guards it. PRE takes two copies of one expression in blocks where
; neither dominates the other, duplicates the expression into their nearest
; common dominator, and lets CSE delete the originals -- so the copy runs on
; every path through that dominator, including the paths that reach neither
; block. CIncOffset of an untagged value (NULL) traps. CPython forms
; &kwnames->ob_item in the preheaders of two inlined find_keyword loops, each
; entered only when `nkwargs > 0`; PRE put the increment into their common
; dominator above that test, and every call without keyword arguments trapped
; in _PyArg_UnpackKeywords. The C-58 fix did not apply: it held only
; MachineLICM to the rule, through MachineLICM's own hook.
;
; The shape here is that one: two guarded blocks form &kw->field in sequence,
; and their nearest common dominator is %entry, where kw may be NULL.
; Control: the same shape with an integer multiply, which cannot trap, is still
; PRE'd into %entry. It is what shows this test can see PRE at all, so that the
; absence of the capability increment from %entry is the rule and not a quiet
; pass.
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs -stop-after=machine-cse < %s | FileCheck %s
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O2 < %s -o /dev/null

; CHECK-LABEL: name: guarded_twice
; CHECK: bb.0.entry:
; CHECK-NOT: CIncOffsetImm
; CHECK: bb.{{[0-9]+}}.first:
; CHECK-NOT: {{^ *bb\.[0-9]+}}
; CHECK: CIncOffsetImm %{{[0-9]+}}, 48
; CHECK: bb.{{[0-9]+}}.second:
; CHECK-NOT: {{^ *bb\.[0-9]+}}
; CHECK: CIncOffsetImm %{{[0-9]+}}, 48

; CHECK-LABEL: name: control_integer
; CHECK: bb.0.entry:
; CHECK-NOT: {{^ *bb\.[0-9]+}}
; CHECK: MUL

declare void @use(ptr addrspace(200)) addrspace(200)
declare void @usei(i64) addrspace(200)

define void @guarded_twice(ptr addrspace(200) %kw, i64 %n, i64 %m) addrspace(200) {
entry:
  %c1 = icmp sgt i64 %n, 0
  br i1 %c1, label %first, label %mid
first:
  %g1 = getelementptr inbounds i8, ptr addrspace(200) %kw, i64 48
  call addrspace(200) void @use(ptr addrspace(200) %g1)
  br label %mid
mid:
  %c2 = icmp sgt i64 %m, 0
  br i1 %c2, label %second, label %exit
second:
  %g2 = getelementptr inbounds i8, ptr addrspace(200) %kw, i64 48
  call addrspace(200) void @use(ptr addrspace(200) %g2)
  br label %exit
exit:
  ret void
}

define void @control_integer(i64 %a, i64 %b, i64 %n, i64 %m) addrspace(200) {
entry:
  %c1 = icmp sgt i64 %n, 0
  br i1 %c1, label %first, label %mid
first:
  %x1 = mul i64 %a, %b
  call addrspace(200) void @usei(i64 %x1)
  br label %mid
mid:
  %c2 = icmp sgt i64 %m, 0
  br i1 %c2, label %second, label %exit
second:
  %x2 = mul i64 %a, %b
  call addrspace(200) void @usei(i64 %x2)
  br label %exit
exit:
  ret void
}
