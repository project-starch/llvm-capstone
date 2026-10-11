; C-55: two cascaded selects on a capability with a null operand. The cascaded
; lowering built its PHI straight from the operand registers, so the physical
; null capability $c0 became a PHI input; LiveVariables (-O1+) and
; PHIElimination (-O0) assert on it. The single-select path had been fixed by
; copying a physical source into a virtual register (d5b5de228b38); the
; cascaded path now does the same.
; RUN: llc -mtriple=capstone64 -verify-machineinstrs -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=MIR
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null
; RUN: %llc_cap -O2 < %s -o /dev/null

; No PHI may take a physical register; the null reaches it through a COPY.
; MIR-LABEL: name: two_selects_null{{$}}
; MIR-NOT: PHI {{.*}}$c0
; MIR: [[NULL:%[0-9]+]]:gpcr = COPY $c0
; MIR-NOT: PHI {{.*}}$c0
; MIR: PHI [[NULL]], %bb.0,
; MIR-NOT: PHI {{.*}}$c0
define ptr addrspace(200) @two_selects_null(ptr addrspace(200) %a, i1 %c) addrspace(200) {
  %x = select i1 %c, ptr addrspace(200) %a, ptr addrspace(200) null
  %y = select i1 %c, ptr addrspace(200) null, ptr addrspace(200) %x
  ret ptr addrspace(200) %y
}

; The same with the null in the other arm of each select.
; MIR-LABEL: name: two_selects_null_swapped
; MIR: PHI
define ptr addrspace(200) @two_selects_null_swapped(ptr addrspace(200) %a, i1 %c) addrspace(200) {
  %x = select i1 %c, ptr addrspace(200) null, ptr addrspace(200) %a
  %y = select i1 %c, ptr addrspace(200) %x, ptr addrspace(200) null
  ret ptr addrspace(200) %y
}
