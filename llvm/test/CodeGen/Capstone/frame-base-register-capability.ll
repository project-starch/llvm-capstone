; C-52: LocalStackSlotAllocation gives loads and stores far from sp a virtual
; base register, which materializeFrameBaseRegister built as RISC-V does: a GPR
; from ADDI. Every Capstone load and store takes a capability base, so the
; rewritten accesses were ill-typed; the verifier rejects them, and the Greedy
; allocator segfaulted on them in SplitEditor::rematWillIncreaseRestriction
; (CPython's compiler_visit_stmt). The base is now a GPCR from CIncOffsetImm.
;
; The shape: two byval copies whose temporaries end up more than 2047 bytes
; from sp, because a larger byval temporary is created after them, so the
; pass builds one base register for their stores to offset 0.
; RUN: llc -mtriple=capstone64 -verify-machineinstrs -stop-after=localstackalloc < %s | FileCheck %s --check-prefix=MIR
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null
; RUN: %llc_cap -O2 < %s -o /dev/null

%pair = type { i64, i64 }
%blob = type { [4096 x i8] }
declare void @take(ptr addrspace(200) byval(%pair)) addrspace(200)
declare void @take_blob(ptr addrspace(200) byval(%blob)) addrspace(200)

; MIR-LABEL: name: far_byval
; MIR-NOT: = ADDI %stack.
; MIR: [[BASE:%[0-9]+]]:gpcr = CIncOffsetImm %stack.1, 0
; MIR: SD {{.*}}, [[BASE]], 16 ::
; MIR: SD {{.*}}, [[BASE]], 0 ::
define void @far_byval(ptr addrspace(200) %p, ptr addrspace(200) %q) addrspace(200) {
  call void @take(ptr addrspace(200) byval(%pair) %p)
  call void @take(ptr addrspace(200) byval(%pair) %p)
  call void @take_blob(ptr addrspace(200) byval(%blob) %q)
  ret void
}
