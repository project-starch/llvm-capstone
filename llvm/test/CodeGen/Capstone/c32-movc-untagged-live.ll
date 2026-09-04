; C-32: `movc rd, rs1` on the RTL NULLS the source register for every source
; that is not NONLIN -- including an UNTAGGED one (capstone_flu_unit.anvil:13-26
; has no NOT_CAP exclusion; rtl-oracle 2026-09-04), while QEMU only nulls a
; tagged, non-copyable source (op_helper.c:580-585).  copyPhysReg emits movc for
; every GPCR copy, so a value bridged from an integer (inttoptr: an integer
; written into the address half of an undefined capability, untagged by
; construction) that is COPIED while it stays live loses its value on silicon
; and keeps it under QEMU.
;
; The shape below is ordinary C: `void *p = (void *)x; use(p); return p;` -- p
; is saved across the call with a copy, and on the RTL the copy zeroes the
; register the call then reads as its argument.  Measured 2026-09-05: -O2 emits
; `movc` for that copy.  The fix is not written yet: the copy of a value known
; to be untagged must be an integer move (which clears the destination's shadow
; and leaves the source alone), and the register classes do not currently tell
; a bridged integer from a capability.  XFAIL until it lands.
;
; MUTATION: n/a until the fix lands -- the CHECK-NOT below fails today, which
; is the XFAIL; @real_cap_copy is the control that a genuine capability copy
; still IS a movc, so the negative check is bounded.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: %llc_cap -O1 < %s -o /dev/null
; XFAIL: *

declare void @use(ptr addrspace(200))

; CHECK-LABEL: bridged_copied_live:
; CHECK-NOT: movc
; CHECK: cjalr ra
; CHECK-NOT: movc
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @bridged_copied_live(i64 %x) {
  %p = inttoptr i64 %x to ptr addrspace(200)
  call void @use(ptr addrspace(200) %p)
  ret ptr addrspace(200) %p
}

; CONTROL: a real capability saved across a call is copied with movc.
; CHECK-LABEL: real_cap_copy:
; CHECK: movc
define ptr addrspace(200) @real_cap_copy(ptr addrspace(200) %p) {
  call void @use(ptr addrspace(200) %p)
  ret ptr addrspace(200) %p
}
