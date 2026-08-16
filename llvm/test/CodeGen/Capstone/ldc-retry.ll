; S-07 instrument: after every ldc, query the loaded value's type and re-issue
; the identical load if it came back NOT_CAP.
;
; TWO ARMS ON PURPOSE. Checking only the instrumented output would pass just as
; happily if the pass silently did nothing, or if it ran unconditionally. The OFF
; arm pins the uninstrumented sequence, so if the default ever flips or the flag
; stops being honoured, this test fails rather than quietly agreeing.
;
; RUN: llc -mtriple=capstone64 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=capstone64 -verify-machineinstrs \
; RUN:     -capstone-retry-untagged-ldc=true < %s \
; RUN:   | FileCheck %s --check-prefix=ON

; A dependent chain: the second load's address IS the first load's result. This
; is the shape all four silicon S-07 wedges share.

; OFF-LABEL: chase:
; OFF:       ldc
; OFF:       ldc
; OFF-NOT:   lcc

; ON-LABEL: chase:
; The guard: total type query (field 1), compare against NOT_CAP (7), skip the
; retry unless it matches.
; ON:       ldc     [[RD:a[0-9]+]], 0([[BASE:a[0-9]+]])
; ON-NEXT:  lcc     [[T:a[0-9]+]], [[RD]], 1
; ON-NEXT:  addi    [[T]], [[T]], -7
; ON-NEXT:  bnez    [[T]], [[SKIP:\.LBB[0-9_]+]]
; The retry must re-issue the IDENTICAL address into the same register.
; ON:       ldc     [[RD]], 0([[BASE]])
; ON:     [[SKIP]]:
define ptr addrspace(200) @chase(ptr addrspace(200) %p) {
entry:
  %a = load ptr addrspace(200), ptr addrspace(200) %p, align 16
  %b = load ptr addrspace(200), ptr addrspace(200) %a, align 16
  ret ptr addrspace(200) %b
}

; A plain integer load must NOT be instrumented -- only ldc is.

; ON-LABEL: scalar_load:
; ON-NOT:   lcc
; ON:       cjalr
define i64 @scalar_load(ptr addrspace(200) %p) {
entry:
  %v = load i64, ptr addrspace(200) %p, align 8
  ret i64 %v
}
