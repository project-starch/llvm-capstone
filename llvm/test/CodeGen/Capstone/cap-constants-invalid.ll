; RUN: split-file %s %t
; RUN: not --crash llc -mtriple=capstone64 -verify-machineinstrs < %t/direct.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DIRECT
; RUN: not --crash llc -mtriple=capstone64 -verify-machineinstrs < %t/gep.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=GEP
; RUN: not --crash llc -mtriple=capstone64 -verify-machineinstrs < %t/scalar-load.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SCALAR-LOAD
; RUN: not --crash llc -mtriple=capstone64 -verify-machineinstrs < %t/cap-load.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CAP-LOAD
; DIRECT: LLVM ERROR: Capstone PureCap: Cannot materialize arbitrary >64-bit constants as capabilities; capabilities are unforgeable
; GEP: LLVM ERROR: Capstone PureCap: CIncOffset displacement must fit in 64 bits
; SCALAR-LOAD: LLVM ERROR: Capstone PureCap: Address displacement must fit in 64 bits
; CAP-LOAD: LLVM ERROR: Capstone PureCap: Folded load/store displacement must fit in 64 bits
;--- direct.ll
define ptr addrspace(200) @wide_const() {
entry:
  ret ptr addrspace(200) inttoptr (i128 18446744073709551616 to ptr addrspace(200))
}
;--- gep.ll
define ptr addrspace(200) @wide_gep(ptr addrspace(200) %p) {
entry:
  %gep = getelementptr i8, ptr addrspace(200) %p, i128 18446744073709551616
  ret ptr addrspace(200) %gep
}
;--- scalar-load.ll
define i32 @wide_gep_scalar_load(ptr addrspace(200) %p) {
entry:
  %gep = getelementptr i8, ptr addrspace(200) %p, i128 18446744073709551616
  %v = load i32, ptr addrspace(200) %gep, align 4
  ret i32 %v
}
;--- cap-load.ll
define ptr addrspace(200) @wide_gep_cap_load(ptr addrspace(200) %p) {
entry:
  %gep = getelementptr i8, ptr addrspace(200) %p, i128 18446744073709551616
  %v = load ptr addrspace(200), ptr addrspace(200) %gep, align 16
  ret ptr addrspace(200) %v
}
