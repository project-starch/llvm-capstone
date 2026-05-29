; RUN: llc -mtriple=capstone64 -mattr=+m -stop-after=instruction-select -o - %s | FileCheck %s --check-prefix=MIR
;
; Scalar values sometimes need to flow through the i128 carrier type used by
; Capstone PureCap lowering. Those widenings must stay on the scalar-copy path
; instead of falling back to the capability-preserving COPY/MOVC path.

; MIR-LABEL: name: widen_i32
; MIR: renamable $x10 = AND killed renamable $x10, killed renamable $x11
; MIR-NEXT: renamable $x10 = PseudoSCALAR_COPY_I128 killed renamable $x10
; MIR: PseudoRET implicit $x10
 define i128 @widen_i32(i32 %x) {
entry:
  %wide = zext i32 %x to i128
  ret i128 %wide
}

; MIR-LABEL: name: widen_i64
; MIR: renamable $x10 = PseudoSCALAR_COPY_I128 killed renamable $x10
; MIR: PseudoRET implicit $x10
 define i128 @widen_i64(i64 %x) {
entry:
  %wide = zext i64 %x to i128
  ret i128 %wide
}

