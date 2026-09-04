; The two diagnostic knobs of CapstoneCapGlobalInit, each at both values.
; -capstone-cap-init-limit=N emits only the first N capability-initializer
; stores (0 = all); -capstone-cap-init-print writes one line per leaf to
; stderr.  Two leaves here: px (holds x) and fp (holds f).  Measured 2026-09-04
; on the branch tools.
;
; MUTATION: limit=0 and limit=1 are each other's mutation -- the ONE arm's
; implicit-check-not on stc fires on the two-store output (performed
; 2026-09-04 by running both).
;
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-cap-init-limit=0 -capstone-cap-init-print=false < %s 2>&1 | FileCheck %s --check-prefix=ALL --implicit-check-not='capstone-cap-init:'
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-cap-init-limit=1 < %s | FileCheck %s --check-prefix=ONE --implicit-check-not=stc
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-cap-init-print=true < %s 2>&1 >/dev/null | FileCheck %s --check-prefix=PRINT
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

@x = addrspace(200) global i64 7
@px = addrspace(200) global ptr addrspace(200) @x

define i64 @f() {
  ret i64 1
}

@fp = addrspace(200) constant ptr addrspace(200) @f

; ALL-LABEL: __capstone_cap_init:
; ALL-COUNT-2: stc
; ALL: cjalr zero, 0(ra)

; ONE-LABEL: __capstone_cap_init:
; ONE: stc
; ONE: cjalr zero, 0(ra)

; PRINT: capstone-cap-init: leaf 0 holder=px path= value=x holder_size=16
; PRINT-NEXT: capstone-cap-init: leaf 1 holder=fp path= value=f holder_size=16
