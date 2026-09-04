; C-17: a capability-typed constant that does not fit in 64 bits cannot be
; materialised -- capabilities are unforgeable -- and the backend must say so
; with a clean diagnostic that names the function and the value, at every
; optimisation level, never with a crash.  A 64-bit-representable constant in
; the same positions compiles to a plain `li`.  Measured 2026-09-04 on the
; branch tools.  (CapstoneISelDAGToDAG.cpp getI128NumericValueOrFatal is the
; site; C-17 in the registry.)
;
; RUN: split-file %s %t
; RUN: not llc -mtriple=capstone64 -mattr=+m -O2 -o /dev/null %t/wide.ll 2>&1 | FileCheck %s --check-prefix=DIAG
; RUN: not llc -mtriple=capstone64 -mattr=+m -O0 -o /dev/null %t/wide.ll 2>&1 | FileCheck %s --check-prefix=DIAG
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -o - %t/narrow.ll | FileCheck %s --check-prefix=CTL
; RUN: %llc_cap -O0 < %t/narrow.ll -o /dev/null
; RUN: %llc_cap -O1 < %t/narrow.ll -o /dev/null

; DIAG: error: {{.*}}in function wide_arm{{.*}}Capstone PureCap: Cannot materialize arbitrary >64-bit constants as capabilities; capabilities are unforgeable (value 0x10000000000000000)
; DIAG: error: {{.*}}in function wide_const{{.*}}Capstone PureCap: Cannot materialize arbitrary >64-bit constants as capabilities; capabilities are unforgeable (value 0x10000000000000000)
; DIAG-NOT: PLEASE submit a bug report
; DIAG-NOT: Stack dump

; CTL-LABEL: narrow_arm:
; CTL: li a0, 14
; CTL-LABEL: narrow_const:
; CTL: li a0, 14
; CTL-NEXT: cjalr zero, 0(ra)

;--- wide.ll
target triple = "capstone64"
define ptr addrspace(200) @wide_arm(i1 %c, ptr addrspace(200) %p) {
  %s = select i1 %c, ptr addrspace(200) inttoptr (i128 18446744073709551616 to ptr addrspace(200)), ptr addrspace(200) %p
  ret ptr addrspace(200) %s
}
define ptr addrspace(200) @wide_const() {
  ret ptr addrspace(200) inttoptr (i128 18446744073709551616 to ptr addrspace(200))
}

;--- narrow.ll
target triple = "capstone64"
define ptr addrspace(200) @narrow_arm(i1 %c, ptr addrspace(200) %p) {
  %s = select i1 %c, ptr addrspace(200) inttoptr (i128 14 to ptr addrspace(200)), ptr addrspace(200) %p
  ret ptr addrspace(200) %s
}
define ptr addrspace(200) @narrow_const() {
  ret ptr addrspace(200) inttoptr (i128 14 to ptr addrspace(200))
}
