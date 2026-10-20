; C-77: the caller's copy of a by-value aggregate, when it is a memcpy call,
; passes both operands as capabilities.
;
; The byval copy is made with SelectionDAG::getMemcpy, which types a memcpy
; call's pointer arguments from the copy's MachinePointerInfo address spaces.
; The byval lowering passed default-constructed infos, address space 0, an
; integer on this target, so the call moved both capabilities into a0/a1 with
; `mv` and memcpy faulted on its untagged source. Perl's lexer passes the
; constant `no_code` (five pointers) by value; a zero constant's copy becomes a
; memcpy call, and every program reaching it faulted (cause 24) in the virtual
; profile. Since C-81 a byval copy without a known 16-byte alignment (a struct
; of longs) is a memcpy call too.
;
; Checked after instruction selection, where the call's argument registers are
; explicit: the memcpy call takes its operands in $c10 and $c11 (capability
; registers), not $x10 and $x11, and only the size in $x12. The 16-aligned copy
; from a pointer stays an inline capability copy (control, checked in the
; emitted code).
;
; MUTATION: with the byval copy's default-constructed pointer infos restored
; and nothing else changed, both calls take `$x10 = COPY` and `$x11 = COPY`
; (in the emitted code `mv a0, s0` and `mv a1, a4`) and fail; the control
; passes (performed 2026-10-16).
;
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs \
; RUN:   -capstone-gp-free -capstone-image-gp -stop-after=finalize-isel < %s \
; RUN:   | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs \
; RUN:   -capstone-gp-free -capstone-image-gp < %s | FileCheck %s --check-prefix=CTRL
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O0 -capstone-gp-free -capstone-image-gp < %s -o /dev/null

%struct.pc = type { ptr addrspace(200), ptr addrspace(200), ptr addrspace(200), ptr addrspace(200), ptr addrspace(200), i64, i8 }
%struct.l6 = type { [6 x i64] }

@k_pc = internal addrspace(200) constant { ptr addrspace(200), ptr addrspace(200), ptr addrspace(200), ptr addrspace(200), ptr addrspace(200), i64, i8, [7 x i8] } zeroinitializer, align 16

declare i64 @t_pc(ptr addrspace(200) byval(%struct.pc) align 16) addrspace(200)
declare i64 @t_l6(ptr addrspace(200) byval(%struct.l6) align 8) addrspace(200)

; CHECK-LABEL: name: pass_const_zero
; CHECK:       PseudoLLA &memcpy
; CHECK-NOT:   $x1{{[01]}} = COPY
; CHECK:       $c10 = COPY
; CHECK-NEXT:  $c11 = COPY
; CHECK-NEXT:  $x12 = COPY
; CHECK-NOT:   $x1{{[01]}} = COPY
; CHECK:       PseudoCALLIndirect {{.*}}, implicit $c10, implicit $c11, implicit $x12,
define i64 @pass_const_zero() addrspace(200) {
entry:
  %call = call addrspace(200) i64 @t_pc(ptr addrspace(200) byval(%struct.pc) align 16 @k_pc)
  ret i64 %call
}

; CHECK-LABEL: name: pass_longs
; CHECK:       PseudoLLA &memcpy
; CHECK-NOT:   $x1{{[01]}} = COPY
; CHECK:       $c10 = COPY
; CHECK-NEXT:  $c11 = COPY
; CHECK-NEXT:  $x12 = COPY
; CHECK-NOT:   $x1{{[01]}} = COPY
; CHECK:       PseudoCALLIndirect {{.*}}, implicit $c10, implicit $c11, implicit $x12,
define i64 @pass_longs(ptr addrspace(200) %p) addrspace(200) {
entry:
  %call = call addrspace(200) i64 @t_l6(ptr addrspace(200) byval(%struct.l6) align 8 %p)
  ret i64 %call
}

; CHECK-LABEL: name: pass_pointers
; CHECK-NOT:   memcpy
; CHECK:       PseudoCALLIndirect
; CTRL-LABEL: pass_pointers:
; CTRL-NOT:   memcpy
; CTRL-COUNT-6: stc a{{[0-9]}}, {{[0-9]+}}(a1)
; CTRL-NOT:   memcpy
; CTRL:       auipc a0, %pcrel_hi(t_pc)
define i64 @pass_pointers(ptr addrspace(200) %p) addrspace(200) {
entry:
  %call = call addrspace(200) i64 @t_pc(ptr addrspace(200) byval(%struct.pc) align 16 %p)
  ret i64 %call
}
