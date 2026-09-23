; C-51: llvm.ptrmask on a capability, and every 8/16-bit atomic, which
; AtomicExpand aligns with ptrmask and turns into a masked LR/SC loop.
; RUN: llc -mtriple=capstone64 -mattr=+a -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+a -verify-machineinstrs -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=MIR
; RUN: %llc_cap -mattr=+a -O0 < %s -o /dev/null
; RUN: %llc_cap -mattr=+a -O1 < %s -o /dev/null
; RUN: llc -mtriple=capstone64 -mattr=+a -verify-machineinstrs -stop-after=capstone-expand-atomic-pseudo < %s | FileCheck %s --check-prefix=EXP
; RUN: llc -mtriple=capstone64 -mattr=+a -verify-machineinstrs -filetype=obj < %s -o /dev/null
;
; Before the fix every function here died in isel ("Shift amount is not an
; integer type!"): the generic ptrmask lowering padded the mask by shifting a
; c128. Now the address is masked and the capability moved by the difference,
; and the masked LR/SC loop runs on the aligned capability with the _CAP
; encodings.

; The address is masked; the capability is moved, never ANDed.
; CHECK-LABEL: align_down16:
; CHECK:      mv [[ADDR:a[0-9]+]], a0
; CHECK-NEXT: andi [[NEW:a[0-9]+]], [[ADDR]], -16
; CHECK-NEXT: sub [[DELTA:a[0-9]+]], [[NEW]], [[ADDR]]
; CHECK-NEXT: cincoffset a0, a0, [[DELTA]]
define ptr addrspace(200) @align_down16(ptr addrspace(200) %p) addrspace(200) {
  %r = call ptr addrspace(200) @llvm.ptrmask.p200.i64(ptr addrspace(200) %p, i64 -16)
  ret ptr addrspace(200) %r
}

; CHECK-LABEL: xchg8:
; CHECK:      cincoffset [[WORD:a[0-9]+]], a0,
; CHECK:      lr.w.aqrl {{a[0-9]+}}, ([[WORD]])
; CHECK:      sc.w.rl {{a[0-9]+}}, {{a[0-9]+}}, ([[WORD]])
; MIR-LABEL: name: xchg8
; MIR: PseudoMaskedAtomicSwap32_CAP
define i8 @xchg8(ptr addrspace(200) %p, i8 %v) addrspace(200) {
  %old = atomicrmw xchg ptr addrspace(200) %p, i8 %v seq_cst
  ret i8 %old
}

; MIR-LABEL: name: add16
; MIR: PseudoMaskedAtomicLoadAdd32_CAP
define i16 @add16(ptr addrspace(200) %p, i16 %v) addrspace(200) {
  %old = atomicrmw add ptr addrspace(200) %p, i16 %v monotonic
  ret i16 %old
}

; MIR-LABEL: name: nand8
; MIR: PseudoMaskedAtomicLoadNand32_CAP
define i8 @nand8(ptr addrspace(200) %p, i8 %v) addrspace(200) {
  %old = atomicrmw nand ptr addrspace(200) %p, i8 %v acquire
  ret i8 %old
}

; MIR-LABEL: name: max8
; MIR: PseudoMaskedAtomicLoadMax32_CAP
define i8 @max8(ptr addrspace(200) %p, i8 %v) addrspace(200) {
  %old = atomicrmw max ptr addrspace(200) %p, i8 %v seq_cst
  ret i8 %old
}

; MIR-LABEL: name: umax16
; MIR: PseudoMaskedAtomicLoadUMax32_CAP
define i16 @umax16(ptr addrspace(200) %p, i16 %v) addrspace(200) {
  %old = atomicrmw umax ptr addrspace(200) %p, i16 %v release
  ret i16 %old
}

; MIR-LABEL: name: cas8
; MIR: PseudoMaskedCmpXchg32_CAP
define i1 @cas8(ptr addrspace(200) %p, i8 %e, i8 %d) addrspace(200) {
  %r = cmpxchg ptr addrspace(200) %p, i8 %e, i8 %d seq_cst seq_cst
  %ok = extractvalue { i8, i1 } %r, 1
  ret i1 %ok
}

; The shape CPython's PyMutex_Lock reaches: a one-byte lock at a struct field.
%PyMutex = type { i8 }
; MIR-LABEL: name: pymutex_lock
; MIR: PseudoMaskedCmpXchg32_CAP
define void @pymutex_lock(ptr addrspace(200) %m) addrspace(200) {
  %r = cmpxchg ptr addrspace(200) %m, i8 0, i8 1 seq_cst seq_cst
  ret void
}

declare ptr addrspace(200) @llvm.ptrmask.p200.i64(ptr addrspace(200), i64)

; After expansion every LR/SC in the file takes its capability address. The
; NOT stands alone so it covers the whole output (a positive EXP after it would
; confine it to the text before that match). MUTATION: fed an expansion with
; one integer-address "LR_W_AQ_RL" line, this check fails (performed 2026-09-23).
; EXP-NOT: {{ (LR|SC)_W(_AQ|_RL|_AQ_RL)? }}
