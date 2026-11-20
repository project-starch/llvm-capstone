; C-68: a scalar integer wider than two XLen words (i256 after seven i64s) is
; split and passed indirectly: the caller stores the parts in a stack temporary
; and passes its address. LowerCall built the part addresses with an i64 ADD over
; the c128 frame index and asserted ("Binary operator types must match!"), and
; LowerFormalArguments did the same over the received address. The address was
; also passed in an X register, which drops a capability's tag. Under a capability
; alloca address space the address now travels as a capability: a C register, or
; a 16-byte, 16-aligned stack slot, and the parts are addressed by PTRADD.
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -verify-machineinstrs < %s -o /dev/null
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=MIR

declare void @sink(i64, i64, i64, i64, i64, i64, i64, i256) addrspace(200)
declare void @sink_stack(i64, i64, i64, i64, i64, i64, i64, i64, i256) addrspace(200)

; The caller stores the four parts into one 32-byte slot and passes it in C17.
; MIR-LABEL: name: pass_in_register
; MIR: $c17 = COPY
; CHECK-LABEL: pass_in_register:
; CHECK: cincoffsetimm a7, sp, 0
; CHECK: sd {{[a-z0-9]+}}, 0(a7)
; CHECK: sd {{[a-z0-9]+}}, 8(a7)
; CHECK: sd {{[a-z0-9]+}}, 16(a7)
; CHECK: sd {{[a-z0-9]+}}, 24(a7)
define void @pass_in_register(i256 %v) addrspace(200) {
  call addrspace(200) void @sink(i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i256 %v)
  ret void
}

; No register left: the address goes on the stack as a capability (stc).
; CHECK-LABEL: pass_on_stack:
; CHECK: stc {{[a-z0-9]+}}, 0(sp)
define void @pass_on_stack(i256 %v) addrspace(200) {
  call addrspace(200) void @sink_stack(i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i256 %v)
  ret void
}

; The callee takes the address in C17 and loads the parts through it.
; MIR-LABEL: name: receive_in_register
; MIR: liveins: {{.*}}$c17
; MIR: {{%[0-9]+}}:gpcr = COPY $c17
; CHECK-LABEL: receive_in_register:
; CHECK-NOT: mv a0, a7
; CHECK: ld {{[a-z0-9]+}}, 24(a7)
define i64 @receive_in_register(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i256 %v) addrspace(200) {
  %h = lshr i256 %v, 192
  %u = trunc i256 %h to i64
  ret i64 %u
}

; The callee loads the address from its stack slot as a capability (ldc).
; CHECK-LABEL: receive_on_stack:
; CHECK: ldc [[A:[a-z0-9]+]], 0(sp)
; CHECK: ld {{[a-z0-9]+}}, 24([[A]])
define i64 @receive_on_stack(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h0, i256 %v) addrspace(200) {
  %h = lshr i256 %v, 192
  %u = trunc i256 %h to i64
  ret i64 %u
}
