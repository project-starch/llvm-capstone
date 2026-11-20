; A byte-copy loop: the pointers' increments come out of instruction selection
; above the load and store that still read the old values, so the register
; allocator closes the iteration with copies. -capstone-enable-sink-loop-increment
; moves an increment below its base's last use when the first effectful
; instruction after it is a LOAD through that base: a sealed or uninitialised
; base then traps at the load with the increment's cause (26), on the RTL and on
; QEMU alike. A STORE keeps the order, because QEMU accepts a store through an
; uninitialised capability; -capstone-sink-loop-increment-past-stores lifts that
; for measurement only.
; REQUIRES: asserts
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -capstone-enable-sink-loop-increment -verify-machineinstrs -stats < %s 2>%t.stats | FileCheck %s --check-prefix=ON
; RUN: FileCheck %s --check-prefix=STATS < %t.stats
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -capstone-enable-sink-loop-increment -capstone-sink-loop-increment-past-stores -verify-machineinstrs < %s | FileCheck %s --check-prefix=STORES
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -capstone-enable-sink-loop-increment < %s -o /dev/null

; Off: the source pointer's next value is made beside the old one and the loop
; spills and reloads it; the destination closes with a copy.
; OFF-LABEL: copy_bytes:
; OFF: lbu
; OFF: sb
; OFF: movc
; OFF: bne

; On: the source increment follows its load; the destination's stays above the
; store, and so does its copy.
; ON-LABEL: copy_bytes:
; ON: lbu [[V:[a-z0-9]+]], 0([[S:[a-z0-9]+]])
; ON-NEXT: cincoffsetimm [[S]], [[S]], 1
; ON-NEXT: cincoffsetimm [[N:[a-z0-9]+]], [[D:[a-z0-9]+]], 1
; ON-NEXT: sb [[V]], 0([[D]])
; ON-NEXT: movc [[D]], [[N]]
; ON-NOT: stc
; ON: bne [[S]]
; STATS: 1 capstone-sink-loop-increment - Trapping increments kept in place to preserve trap order
; STATS: 1 capstone-sink-loop-increment - Loop increments sunk below their base's last use

; Past stores (measurement only): both increments follow their accesses, no copy.
; STORES-LABEL: copy_bytes:
; STORES-NOT: movc
; STORES: lbu [[V:[a-z0-9]+]], 0([[S:[a-z0-9]+]])
; STORES-NEXT: cincoffsetimm [[S]], [[S]], 1
; STORES-NEXT: sb [[V]], 0([[D:[a-z0-9]+]])
; STORES-NEXT: cincoffsetimm [[D]], [[D]], 1
; STORES-NOT: movc
; STORES: bne

define void @copy_bytes(ptr addrspace(200) %d, ptr addrspace(200) %s, i64 %n) addrspace(200) {
entry:
  %tobool.not3 = icmp eq i64 %n, 0
  br i1 %tobool.not3, label %while.end, label %while.body

while.body:
  %n.addr.06 = phi i64 [ %dec, %while.body ], [ %n, %entry ]
  %d.addr.05 = phi ptr addrspace(200) [ %incdec.ptr1, %while.body ], [ %d, %entry ]
  %s.addr.04 = phi ptr addrspace(200) [ %incdec.ptr, %while.body ], [ %s, %entry ]
  %dec = add i64 %n.addr.06, -1
  %incdec.ptr = getelementptr inbounds i8, ptr addrspace(200) %s.addr.04, i64 1
  %0 = load i8, ptr addrspace(200) %s.addr.04, align 1
  %incdec.ptr1 = getelementptr inbounds i8, ptr addrspace(200) %d.addr.05, i64 1
  store i8 %0, ptr addrspace(200) %d.addr.05, align 1
  %tobool.not = icmp eq i64 %dec, 0
  br i1 %tobool.not, label %while.end, label %while.body

while.end:
  ret void
}
