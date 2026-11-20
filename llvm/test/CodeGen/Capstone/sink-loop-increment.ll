; A pointer's increment leaves instruction selection above the access that still
; reads the old value, so the register allocator closes each iteration with a copy
; (here a capability spill and reload). -capstone-enable-sink-loop-increment moves
; the increment below the old value's last use. A trapping increment (CIncOffset,
; 26 on a sealed or uninitialised base) moves only when its base is proven usable:
; derived in the function from a stack slot or a global through offsets,
; narrowings, copies and PHIs. Otherwise a load or store through an invalid base
; would read or write before the trap, since in a domain silicon does not check a
; scalar access's base type (ISSUES Q-36).
; REQUIRES: asserts
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -capstone-enable-sink-loop-increment -verify-machineinstrs -stats < %s 2>%t.stats | FileCheck %s --check-prefix=ON
; RUN: FileCheck %s --check-prefix=STATS < %t.stats
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -capstone-enable-sink-loop-increment < %s -o /dev/null

; Argument pointers: not proven, so the order and the copy stay.
; ON-LABEL: copy_bytes:
; ON: sb {{[a-z0-9]+}}, 0([[D:[a-z0-9]+]])
; ON-NEXT: movc [[D]],

; A global's elements: the increment follows the store and the loop is three
; instructions, where the default keeps a spill and a reload.
; OFF-LABEL: fill_global:
; OFF: stc
; OFF: ldc
; ON-LABEL: fill_global:
; ON: sb [[V:[a-z0-9]+]], 0([[P:[a-z0-9]+]])
; ON-NEXT: cincoffsetimm [[P]], [[P]], 1
; ON-NEXT: bne [[P]]

; A local array: likewise.
; ON-LABEL: fill_local:
; ON: sb [[V2:[a-z0-9]+]], 0([[P2:[a-z0-9]+]])
; ON-NEXT: cincoffsetimm [[P2]], [[P2]], 1
; ON-NEXT: bne [[P2]]

; STATS: 2 capstone-sink-loop-increment - Trapping increments kept in place: base not proven usable
; STATS: 2 capstone-sink-loop-increment - Loop increments sunk below their base's last use
; STATS: 2 capstone-sink-loop-increment - Trapping increments sunk because their base is proven usable

@g = addrspace(200) global [256 x i8] zeroinitializer, align 16
declare void @use(ptr addrspace(200)) addrspace(200)

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

define void @fill_global(i8 %v) addrspace(200) {
entry:
  br label %body
body:
  %p = phi ptr addrspace(200) [ @g, %entry ], [ %p.next, %body ]
  %p.next = getelementptr inbounds i8, ptr addrspace(200) %p, i64 1
  store i8 %v, ptr addrspace(200) %p, align 1
  %done = icmp eq ptr addrspace(200) %p.next, getelementptr (i8, ptr addrspace(200) @g, i64 256)
  br i1 %done, label %exit, label %body
exit:
  ret void
}

define void @fill_local(i8 %v) addrspace(200) {
entry:
  %buf = alloca [64 x i8], align 16, addrspace(200)
  %end = getelementptr inbounds i8, ptr addrspace(200) %buf, i64 64
  br label %body
body:
  %p = phi ptr addrspace(200) [ %buf, %entry ], [ %p.next, %body ]
  %p.next = getelementptr inbounds i8, ptr addrspace(200) %p, i64 1
  store i8 %v, ptr addrspace(200) %p, align 1
  %done = icmp eq ptr addrspace(200) %p.next, %end
  br i1 %done, label %exit, label %body
exit:
  call addrspace(200) void @use(ptr addrspace(200) %buf)
  ret void
}
