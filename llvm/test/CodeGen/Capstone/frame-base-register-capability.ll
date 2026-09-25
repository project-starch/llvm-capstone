; C-52: LocalStackSlotAllocation gives loads and stores far from sp a virtual
; base register, which materializeFrameBaseRegister built as RISC-V does: a GPR
; from ADDI. Every Capstone load and store takes a capability base, so the
; rewritten accesses were ill-typed; the verifier rejects them, and the Greedy
; allocator segfaulted on them in SplitEditor::rematWillIncreaseRestriction
; (CPython's compiler_visit_stmt). The base is now a GPCR from CIncOffsetImm.
;
; THE C-52 GUARD IS NOW frame-base-register-capability.mir, NOT THIS FILE.
; C-50's fix gives a caller-side byval copy a capability frame index, so no
; store carries a bare frame index any more and LocalStackSlotAllocation
; materialises NO base register from any C source in this tree -- measured:
; zero of the 102 .ll tests here reach it. The MIR checks this file used to
; carry therefore became vacuous: they would have passed whether or not the
; C-52 fix was present, which is not a guard. They moved to the .mir test,
; which feeds the pass its recorded input and is negative-tested (forcing
; materializeFrameBaseRegister back to GPR+ADDI makes it fail).
;
; What remains here is a verifier smoke test over the original C shape, which
; is worth keeping and is not a C-52 guard.
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null
; RUN: %llc_cap -O2 < %s -o /dev/null

%pair = type { i64, i64 }
%blob = type { [4096 x i8] }
declare void @take(ptr addrspace(200) byval(%pair)) addrspace(200)
declare void @take_blob(ptr addrspace(200) byval(%blob)) addrspace(200)

define void @far_byval(ptr addrspace(200) %p, ptr addrspace(200) %q) addrspace(200) {
  call void @take(ptr addrspace(200) byval(%pair) %p)
  call void @take(ptr addrspace(200) byval(%pair) %p)
  call void @take_blob(ptr addrspace(200) byval(%blob) %q)
  ret void
}
