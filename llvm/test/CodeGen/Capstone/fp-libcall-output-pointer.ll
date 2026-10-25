; C-86: a floating-point libcall's output pointer (modf, modff, frexp) is
; passed as a capability.
;
; SelectionDAG expands llvm.modf and llvm.frexp into a call that writes one
; result through a pointer to a stack temporary
; (SelectionDAG::expandMultipleResultFPLibCall). The pointer argument was typed
; in address space 0, an integer on this target, so the call moved the frame
; capability into a1 with `mv` and modf faulted storing the integral part
; (cause 24 in musl's modf, `fsd fa5, 0(a1)`; perlbench's unpack checksums call
; modf). The pointer is now typed in the alloca address space.
;
; Checked after instruction selection: the call takes the pointer in $c11, a
; capability register, not $x11.
;
; MUTATION (each with nothing else changed, performed 2026-10-16): with the
; address-space-0 type restored in SelectionDAG::expandMultipleResultFPLibCall,
; @modf_sum, @modf_out, @modff_sum and @frexp_exp take the pointer in $x11 and
; fail; restored in the two soft-float legalizer sites, @modfl_sum and
; @frexpl_exp take it in $x12 and fail.
;
; RUN: llc -mtriple=capstone64 -mattr=+m,+f,+d -target-abi lp64 -verify-machineinstrs \
; RUN:   -capstone-gp-free -capstone-image-gp -stop-after=finalize-isel < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m,+f,+d -target-abi lp64 -verify-machineinstrs \
; RUN:   -capstone-gp-captable -stop-after=finalize-isel < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs \
; RUN:   -stop-after=finalize-isel < %s | FileCheck %s
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: llc -mtriple=capstone64 -mattr=+m,+f,+d -target-abi lp64 -verify-machineinstrs \
; RUN:   -capstone-gp-free -capstone-image-gp -O0 < %s -o /dev/null

declare { double, double } @llvm.modf.f64(double)
declare { float, float } @llvm.modf.f32(float)
declare { double, i32 } @llvm.frexp.f64.i32(double)

; CHECK-LABEL: name: modf_sum
; CHECK:       PseudoLLA &modf
; CHECK-NOT:   $x11 = COPY
; CHECK:       PseudoCALLIndirect {{.*}}, implicit $x10, implicit $c11,
define double @modf_sum(double %x) addrspace(200) {
  %r = call { double, double } @llvm.modf.f64(double %x)
  %f = extractvalue { double, double } %r, 0
  %i = extractvalue { double, double } %r, 1
  %s = fadd double %f, %i
  ret double %s
}

; The results are stored through capability pointers, which are not folded
; into the call; the call still writes a stack temporary.
; CHECK-LABEL: name: modf_out
; CHECK:       PseudoLLA &modf
; CHECK-NOT:   $x11 = COPY
; CHECK:       PseudoCALLIndirect {{.*}}, implicit $x10, implicit $c11,
define void @modf_out(double %x, ptr addrspace(200) %ip, ptr addrspace(200) %fp) addrspace(200) {
  %r = call { double, double } @llvm.modf.f64(double %x)
  %f = extractvalue { double, double } %r, 0
  %i = extractvalue { double, double } %r, 1
  store double %i, ptr addrspace(200) %ip
  store double %f, ptr addrspace(200) %fp
  ret void
}

; CHECK-LABEL: name: modff_sum
; CHECK:       PseudoLLA &modff
; CHECK-NOT:   $x11 = COPY
; CHECK:       PseudoCALLIndirect {{.*}}, implicit $x10, implicit $c11,
define float @modff_sum(float %x) addrspace(200) {
  %r = call { float, float } @llvm.modf.f32(float %x)
  %f = extractvalue { float, float } %r, 0
  %i = extractvalue { float, float } %r, 1
  %s = fadd float %f, %i
  ret float %s
}

; CHECK-LABEL: name: frexp_exp
; CHECK:       PseudoLLA &frexp
; CHECK-NOT:   $x11 = COPY
; CHECK:       PseudoCALLIndirect {{.*}}, implicit $x10, implicit $c11,
define i32 @frexp_exp(double %x) addrspace(200) {
  %r = call { double, i32 } @llvm.frexp.f64.i32(double %x)
  %e = extractvalue { double, i32 } %r, 1
  ret i32 %e
}

; long double is fp128 and always softened: the soft-float legalizer builds
; the same call (DAGTypeLegalizer::SoftenFloatRes_UnaryWithTwoFPResults and
; SoftenFloatRes_FFREXP). The value takes a0/a1, so the pointer is $c12.
declare { fp128, fp128 } @llvm.modf.f128(fp128)
declare { fp128, i32 } @llvm.frexp.f128.i32(fp128)

; CHECK-LABEL: name: modfl_sum
; CHECK:       PseudoLLA &modfl
; CHECK-NOT:   $x12 = COPY
; CHECK:       PseudoCALLIndirect {{.*}}, implicit $x10, implicit $x11, implicit $c12,
define fp128 @modfl_sum(fp128 %x) addrspace(200) {
  %r = call { fp128, fp128 } @llvm.modf.f128(fp128 %x)
  %f = extractvalue { fp128, fp128 } %r, 0
  %i = extractvalue { fp128, fp128 } %r, 1
  %s = fadd fp128 %f, %i
  ret fp128 %s
}

; CHECK-LABEL: name: frexpl_exp
; CHECK:       PseudoLLA &frexpl
; CHECK-NOT:   $x12 = COPY
; CHECK:       PseudoCALLIndirect {{.*}}, implicit $x10, implicit $x11, implicit $c12,
define i32 @frexpl_exp(fp128 %x) addrspace(200) {
  %r = call { fp128, i32 } @llvm.frexp.f128.i32(fp128 %x)
  %e = extractvalue { fp128, i32 } %r, 1
  ret i32 %e
}
