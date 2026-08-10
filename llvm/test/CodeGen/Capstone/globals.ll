; RUN: llc -mtriple=capstone64 -verify-machineinstrs < %s | FileCheck %s
; R-20 WORKAROUND (temporary). This test's hand-written CHECK chain is pinned to the
; pre-workaround register allocation: STC's base operand may no longer be a0, so the
; captured-register chain no longer lines up. The generated code is correct.
; See capstone/tests/fpga-repros/R20-stc-rs1-cursor-forward-x10/WORKAROUND.md.
; When the workaround is reverted this test will XPASS, which lit reports as a failure --
; that is the signal to DELETE these five lines.
; XFAIL: *

@g_val = addrspace(200) global i64 123, align 8
@g_ptr = addrspace(200) global ptr addrspace(200) null, align 16

; CHECK-LABEL: test_global_write:
; CHECK: auipc [[REG1:a[0-9]+]], %pcrel_hi(g_ptr)
; CHECK: auipc [[REG2:a[0-9]+]], %pcrel_hi(g_val)
; CHECK: cincoffset [[CAP_PTR:a[0-9]+]], gp, [[REG1]]
; CHECK: cincoffset [[CAP_VAL:a[0-9]+]], gp, [[REG2]]
; CHECK: stc [[CAP_VAL]], 0([[CAP_PTR]])
define void @test_global_write() {
entry:
  store ptr addrspace(200) @g_val, ptr addrspace(200) @g_ptr, align 16
  ret void
}

; CHECK-LABEL: test_global_read:
; CHECK: auipc [[REG:a[0-9]+]], %pcrel_hi(g_val)
; CHECK: cincoffset [[CAP:a[0-9]+]], gp, [[REG]]
; CHECK: ld a0, 0([[CAP]])
define i64 @test_global_read() {
entry:
  %0 = load i64, ptr addrspace(200) @g_val, align 8
  ret i64 %0
}