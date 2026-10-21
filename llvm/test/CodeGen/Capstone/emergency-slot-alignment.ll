; RUN: llc -mtriple=capstone64 -O1 -frame-pointer=none -stop-after=prologepilog < %s | FileCheck %s --check-prefix=NOFP
; RUN: llc -mtriple=capstone64 -O1 -frame-pointer=all -stop-after=prologepilog < %s | FileCheck %s --check-prefix=FP
;
; The emergency spill slot holds a capability, so it is 16 bytes at 16-byte
; alignment. With no frame pointer the frame is laid out from the incoming
; stack pointer, the slot goes last, and the 8 bytes of outgoing stack
; arguments are added afterwards, which rounds the frame up by 8. PEI then
; moves the slot down by that padding to keep it near SP; a slot moved by less
; than its alignment ended up 8 bytes off a 16-byte boundary, and a capability
; spilled there with stc faults (cause 6). The slot may only move by a multiple
; of its alignment.
;
; The offsets below are relative to the incoming SP and the stack size is a
; multiple of 16, so a 16-byte slot is aligned if and only if its offset is.

target triple = "capstone64"

declare void @use(ptr addrspace(200), i64, i64, i64, i64, i64, i64, i64, i64) addrspace(200)

; NOFP-LABEL: name: f
; NOFP: stackSize: 2096
; NOFP: maxCallFrameSize: 8
; NOFP: type: spill-slot, offset: -2080, size: 16, alignment: 16
; NOFP-NOT: offset: -2088

; FP-LABEL: name: f
; FP: stackSize: 2112
define void @f() addrspace(200) {
  %a = alloca [2048 x i8], align 8, addrspace(200)
  call addrspace(200) void @use(ptr addrspace(200) %a, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8)
  ret void
}
