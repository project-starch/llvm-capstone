; C-53: an inline-asm "m" INPUT operand on a capability pointer asserted in
; SelectionDAGBuilder::visitInlineAsm, whose check named only the address-space-0
; pointer type (i64) while the operand is a c128 capability. Fixed by C-80's
; change to that check (7ce213020f87): a memory operand may be a pointer in its
; own address space. The memory is addressed through the capability itself.
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -verify-machineinstrs < %s -o /dev/null
; RUN: llc -mtriple=capstone64 -mattr=+m -O1 -verify-machineinstrs < %s | FileCheck %s

@cw = addrspace(200) global i16 0, align 2

; A pointer argument: the load goes through the argument's capability.
; CHECK-LABEL: via_pointer:
; CHECK: #APP
; CHECK-NEXT: lw zero, 0(a0)
define void @via_pointer(ptr addrspace(200) %p) addrspace(200) {
  call addrspace(200) void asm sideeffect "lw zero, $0", "*m"(ptr addrspace(200) elementtype(i32) %p)
  ret void
}

; A local: a stack slot's capability.
; CHECK-LABEL: via_local:
; CHECK: cincoffsetimm [[L:a[0-9]+]], sp,
; CHECK: #APP
; CHECK-NEXT: lhu zero, 0({{a[0-9]+}})
define void @via_local() addrspace(200) {
  %cw = alloca i16, align 2, addrspace(200)
  store i16 1, ptr addrspace(200) %cw, align 2
  call addrspace(200) void asm sideeffect "lhu zero, $0", "*m"(ptr addrspace(200) elementtype(i16) %cw)
  ret void
}

; A global: the global's capability.
; CHECK-LABEL: via_global:
; CHECK: auipc {{a[0-9]+}}, %pcrel_hi(cw)
; CHECK: #APP
; CHECK-NEXT: lhu zero, 0({{a[0-9]+}})
define void @via_global() addrspace(200) {
  call addrspace(200) void asm sideeffect "lhu zero, $0", "*m"(ptr addrspace(200) elementtype(i16) @cw)
  ret void
}
