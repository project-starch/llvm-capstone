; RUN: split-file %s %t
; RUN: not --crash llc -mtriple=capstone64 -verify-machineinstrs < %t/direct.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DIRECT
; RUN: not --crash llc -mtriple=capstone64 -verify-machineinstrs < %t/gep.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=GEP
; DIRECT: LLVM ERROR: Capstone PureCap: Cannot materialize arbitrary >64-bit constants as capabilities; capabilities are unforgeable
; GEP: LLVM ERROR: Capstone PureCap: CIncOffset displacement must fit in signed 64-bits
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
