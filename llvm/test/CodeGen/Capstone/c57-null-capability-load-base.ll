; C-57: a load through the null capability selected $x0, a GPR, as its base, and
; the machine verifier rejected it ("Illegal physical register for instruction").
; The address matcher's constant case now names C0, the null capability, for a
; capability-typed base. x0 and c0 share an encoding, so the emitted instruction
; is unchanged: it still addresses through the null capability and traps, which
; is what the program asked for.
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -verify-machineinstrs < %s -o /dev/null
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=MIR

; MIR-LABEL: name: load_null
; MIR: LW $c0, 0
; CHECK-LABEL: load_null:
; CHECK: lw a0, 0(zero)
define i32 @load_null() addrspace(200) {
  %v = load volatile i32, ptr addrspace(200) null
  ret i32 %v
}

; Null plus a small constant already went through a copy of C0 (a control).
; MIR-LABEL: name: load_null_small
; MIR: [[B:%[0-9]+]]:gpcr = COPY $c0
; MIR: LW [[B]], 64
; CHECK-LABEL: load_null_small:
; CHECK: lw a0, 64(zero)
define i32 @load_null_small() addrspace(200) {
  %p = getelementptr i8, ptr addrspace(200) null, i64 64
  %v = load volatile i32, ptr addrspace(200) %p
  ret i32 %v
}

; The store path was already right (COPY $c0); a control.
; CHECK-LABEL: store_null:
; CHECK: sw a0, 0(zero)
define void @store_null(i32 %x) addrspace(200) {
  store volatile i32 %x, ptr addrspace(200) null
  ret void
}
