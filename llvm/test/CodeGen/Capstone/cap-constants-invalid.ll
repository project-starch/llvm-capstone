; A capability is unforgeable, so a displacement that does not fit in 64 bits
; has to be refused rather than silently truncated into the metadata bits.
;
; The guards are reached through i128 address arithmetic. A GEP can no longer
; reach them: the index width is 64, so LLVM truncates the index first -- see
; gep-truncates.ll, which pins that down.

; RUN: split-file %s %t
; RUN: not --crash llc -mtriple=capstone64 -verify-machineinstrs < %t/direct.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DIRECT
; RUN: not --crash llc -mtriple=capstone64 -verify-machineinstrs < %t/incoffset.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INCOFFSET
; RUN: not --crash llc -mtriple=capstone64 -verify-machineinstrs < %t/scalar-load.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SCALAR-LOAD
; RUN: not --crash llc -mtriple=capstone64 -verify-machineinstrs < %t/cap-load.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CAP-LOAD
; RUN: llc -mtriple=capstone64 -verify-machineinstrs < %t/gep-truncates.ll -o - | FileCheck %s --check-prefix=GEP

; DIRECT: LLVM ERROR: Capstone PureCap: Cannot materialize arbitrary >64-bit constants as capabilities; capabilities are unforgeable
; The other three arrive at the shared guard in CapstoneDAGToDAGISel::Select, which
; refuses the constant before the generated matcher reads it as an int64_t. They used
; to be refused further along, at three different sites with three different messages,
; because each shape reached a different piece of the capability-arithmetic guessing.
; INCOFFSET: LLVM ERROR: Capstone PureCap: Address displacement must fit in 64 bits
; SCALAR-LOAD: LLVM ERROR: Capstone PureCap: Address displacement must fit in 64 bits
; CAP-LOAD: LLVM ERROR: Capstone PureCap: Address displacement must fit in 64 bits

;--- direct.ll
define ptr addrspace(200) @wide_const() {
entry:
  ret ptr addrspace(200) inttoptr (i128 18446744073709551616 to ptr addrspace(200))
}
;--- incoffset.ll
define ptr addrspace(200) @wide_incoffset(ptr addrspace(200) %p) {
entry:
  %i = ptrtoint ptr addrspace(200) %p to i128
  %a = add i128 %i, 18446744073709551616
  %q = inttoptr i128 %a to ptr addrspace(200)
  ret ptr addrspace(200) %q
}
;--- scalar-load.ll
define i32 @wide_scalar_load(ptr addrspace(200) %p) {
entry:
  %i = ptrtoint ptr addrspace(200) %p to i128
  %a = add i128 %i, 18446744073709551616
  %q = inttoptr i128 %a to ptr addrspace(200)
  %v = load i32, ptr addrspace(200) %q, align 4
  ret i32 %v
}
;--- cap-load.ll
define ptr addrspace(200) @wide_cap_load(ptr addrspace(200) %p) {
entry:
  %i = ptrtoint ptr addrspace(200) %p to i128
  %a = add i128 %i, 18446744073709551616
  %q = inttoptr i128 %a to ptr addrspace(200)
  %v = load ptr addrspace(200), ptr addrspace(200) %q, align 16
  ret ptr addrspace(200) %v
}
;--- gep-truncates.ll
; 2^64 taken modulo the 64-bit index width is 0, so the pointer comes back
; untouched. This is a defined result, not a dropped offset.
; GEP-LABEL: wide_gep:
; GEP-NOT: cincoffset
; GEP: cjalr zero, 0(ra)
define ptr addrspace(200) @wide_gep(ptr addrspace(200) %p) {
entry:
  %gep = getelementptr i8, ptr addrspace(200) %p, i128 18446744073709551616
  ret ptr addrspace(200) %gep
}
