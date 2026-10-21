; CapstoneCursorReadFold: a capability's address that is only ever READ by an
; ordinary ALU op or compare is read straight from the capability register;
; the PseudoTRUNC_CAP integer write (`mv rd, rs`, C-31) stays wherever the
; address goes on as a value -- into a capability instruction's integer
; operand, a store, a copy, a call, a return, or a select's data operands.
;
; MUTATION: disable the pass -> every positive CHECK below finds a `mv` before
; the compare, sub or mask. The OFF run below is that mutation, kept as a flag
; (-capstone-enable-cursor-read-fold=0) so a board regression can be bisected
; to this pass without a rebuild.
;
; The gp-free ABI is the application profile; its returns print as `ret`.
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -capstone-gp-free -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -O1 -capstone-gp-free -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -capstone-gp-free -verify-machineinstrs \
; RUN:   -capstone-enable-cursor-read-fold=0 < %s | FileCheck %s --check-prefix=OFF
; RUN: %llc_cap -O0 -capstone-gp-free < %s -o /dev/null
; RUN: %llc_cap -O2 < %s -o /dev/null

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

; A loop whose exit compares two pointers: the branch reads both registers.
; CHECK-LABEL: zero_to:
; CHECK:      beq a0, a1,
; CHECK:      sd zero, 0(a0)
; CHECK-NEXT: cincoffsetimm a0, a0, 8
; CHECK-NEXT: bne a0, a1,
; OFF-LABEL: zero_to:
; OFF:       mv {{a[0-9]+}}, a0
; OFF:       beq
define void @zero_to(ptr addrspace(200) %p, ptr addrspace(200) %end) {
entry:
  %c = icmp eq ptr addrspace(200) %p, %end
  br i1 %c, label %done, label %loop
loop:
  %q = phi ptr addrspace(200) [ %p, %entry ], [ %n, %loop ]
  store i64 0, ptr addrspace(200) %q
  %n = getelementptr i64, ptr addrspace(200) %q, i64 1
  %e = icmp eq ptr addrspace(200) %n, %end
  br i1 %e, label %done, label %loop
done:
  ret void
}

; A pointer difference: the subtrahend is read straight from its register; the
; minuend keeps its integer write, because `sub rd, rs, x0` is a copy of rs and
; the subtrahend register could still become x0 after the pass.
; CHECK-LABEL: diff:
; CHECK-NOT: mv a0, a0
; CHECK:     mv a1, a1
; CHECK-NEXT: sub a0, a1, a0
; CHECK-NEXT: ret
; OFF-LABEL: diff:
; OFF:       mv a0, a0
; OFF:       sub
define i64 @diff(ptr addrspace(200) %a, ptr addrspace(200) %b) {
  %ia = ptrtoint ptr addrspace(200) %a to i64
  %ib = ptrtoint ptr addrspace(200) %b to i64
  %d = sub i64 %ib, %ia
  ret i64 %d
}

; An alignment mask.
; CHECK-LABEL: low_bits:
; CHECK-NOT: mv
; CHECK:     andi a0, a0, 7
; CHECK-NOT: mv
; CHECK:     ret
; OFF-LABEL: low_bits:
; OFF:       mv a0, a0
; OFF-NEXT:  andi a0, a0, 7
define i64 @low_bits(ptr addrspace(200) %p) {
  %i = ptrtoint ptr addrspace(200) %p to i64
  %m = and i64 %i, 7
  ret i64 %m
}

; A null test.
; CHECK-LABEL: is_null:
; CHECK-NOT: mv
; CHECK:     seqz a0, a0
; CHECK-NOT: mv
; CHECK:     ret
; OFF-LABEL: is_null:
; OFF:       mv a0, a0
; OFF-NEXT:  seqz a0, a0
define i64 @is_null(ptr addrspace(200) %p) {
  %c = icmp eq ptr addrspace(200) %p, null
  %z = zext i1 %c to i64
  ret i64 %z
}

; CONTROL (C-31): the address is the rs2 of a cincoffset -- the integer write
; stays, and the cincoffset reads the written register, not the capability.
; CHECK-LABEL: as_offset:
; CHECK:     mv [[R:a[0-9]+]], a1
; CHECK:     cincoffset a0, a0, [[R]]
define ptr addrspace(200) @as_offset(ptr addrspace(200) %q, ptr addrspace(200) %p) {
  %i = ptrtoint ptr addrspace(200) %p to i64
  %r = getelementptr i8, ptr addrspace(200) %q, i64 %i
  ret ptr addrspace(200) %r
}

; CONTROL: the address leaves as a value (returned): the write stays.
; CHECK-LABEL: as_int:
; CHECK:     mv a0, a0
; CHECK:     ret
define i64 @as_int(ptr addrspace(200) %p) {
  %i = ptrtoint ptr addrspace(200) %p to i64
  ret i64 %i
}

; CONTROL: the address is stored: the write stays.
; CHECK-LABEL: store_int:
; CHECK:     mv [[R:a[0-9]+]], a0
; CHECK:     sd [[R]], 0(a1)
define void @store_int(ptr addrspace(200) %p, ptr addrspace(200) %out) {
  %i = ptrtoint ptr addrspace(200) %p to i64
  store i64 %i, ptr addrspace(200) %out
  ret void
}

; CONTROL: one reader is a compare, the other a cincoffset operand -- the
; write stays for both.
; CHECK-LABEL: mixed:
; CHECK:     mv [[R:a[0-9]+]], a1
; CHECK:     cincoffset {{a[0-9]+}}, a0, [[R]]
define ptr addrspace(200) @mixed(ptr addrspace(200) %q, ptr addrspace(200) %p) {
  %i = ptrtoint ptr addrspace(200) %p to i64
  %r = getelementptr i8, ptr addrspace(200) %q, i64 %i
  %c = icmp eq i64 %i, 0
  %s = select i1 %c, ptr addrspace(200) %q, ptr addrspace(200) %r
  ret ptr addrspace(200) %s
}
