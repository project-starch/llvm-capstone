; C-20: `llvm.cttz` crashes the legalizer on capstone64 at every optimisation
; level (LegalizeDAG assertion `Res.getValueType() == Node->getValueType(0)`),
; for i32 and i64 alike.  Measured 2026-09-04 on the branch toolchain; the
; registry's C-20/C-24 commits are repro packages, not fixes.  ctlz and ctpop
; compile (3 of 3 functions emitted in the same measurement), so the control
; below is green on its own and the crash is specific to cttz.
;
; Fixed 2026-09-04: the cause was not a missing legalization action but the
; generic de Bruijn table lookup (TargetLowering::CTTZTableLookup) asking for
; the table's address in address space 0, an i64, while lowerConstantPool
; always produces the c128 capability -- "Type mismatch for custom legalized
; operation".  The lookup now uses the default globals address space for the
; constant-pool pointer, so the expansion is a multiply, a shift and a byte
; load through a capability, pinned below.
;
; Under -capstone-gp-captable that constant pool is unreachable: gp is bounded
; to the cap table and nothing else addresses .rodata, so the byte load faults
; out of bounds (measured 2026-09-05 under QEMU, cause 5 at the table's
; address, -O0 and -O2 alike).  Under that ABI cttz lowers to the arithmetic
; popcount form instead and no constant pool exists; the TABLE arm proves the
; default ABI still uses one, so the CAPTABLE-NOT can fail.
;
; MUTATION: change ctlz to cttz in @ctlz32_control and the two functions must
; still both be emitted; drop the gp-captable branch of the CTTZ lowering and
; `.LCPI0_0:` reappears in the CAPTABLE arm.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: %llc_cap -O1 < %s -o /dev/null
; RUN: %llc_cap -O2 < %s | FileCheck %s --check-prefix=TABLE
; RUN: %llc_cap -O2 -capstone-gp-captable < %s | FileCheck %s --check-prefix=CAPTABLE
; RUN: %llc_cap -O0 -capstone-gp-captable < %s | FileCheck %s --check-prefix=CAPTABLE

; TABLE: .LCPI0_0:
; TABLE: cttz32:
;
; CAPTABLE-NOT: .LCPI
; CAPTABLE: cttz32:
; CAPTABLE-NOT: .LCPI
; CAPTABLE: ctlz32_control:
; CAPTABLE-NOT: .LCPI
; CAPTABLE: .Lfunc_end2:

declare i32 @llvm.cttz.i32(i32, i1)
declare i64 @llvm.cttz.i64(i64, i1)
declare i32 @llvm.ctlz.i32(i32, i1)

; CHECK-LABEL: cttz32:
; CHECK: cjalr zero, 0(ra)
define i32 @cttz32(i32 %x) {
  %r = call i32 @llvm.cttz.i32(i32 %x, i1 false)
  ret i32 %r
}

; CHECK-LABEL: cttz64:
; CHECK: cjalr zero, 0(ra)
define i64 @cttz64(i64 %x) {
  %r = call i64 @llvm.cttz.i64(i64 %x, i1 false)
  ret i64 %r
}

; CONTROL: the sibling intrinsic legalizes and is emitted.
; CHECK-LABEL: ctlz32_control:
; CHECK: cjalr zero, 0(ra)
define i32 @ctlz32_control(i32 %x) {
  %r = call i32 @llvm.ctlz.i32(i32 %x, i1 false)
  ret i32 %r
}

; The zero-is-poison form must keep the default ABI's table path with NO zero
; check: marking CTTZ Custom unconditionally made the generic expansion route
; CTTZ_ZERO_UNDEF through CTTZ and add `li a0, 64` + a branch (audit finding
; 2026-09-05; the Custom action is now taken only under gp-captable).
; TABLE-LABEL: cttz64_zu:
; TABLE-NOT: li {{[a-z0-9]+}}, 64
; TABLE: cjalr zero, 0(ra)
; CHECK-LABEL: cttz64_zu:
; CHECK: cjalr zero, 0(ra)
define i64 @cttz64_zu(i64 %x) {
  %r = call i64 @llvm.cttz.i64(i64 %x, i1 true)
  ret i64 %r
}
