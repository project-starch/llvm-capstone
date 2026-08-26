; RUN: split-file %s %t
; RUN: not llc -mtriple=capstone64 -verify-machineinstrs < %t/direct.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DIRECT
; RUN: not llc -mtriple=capstone64 -verify-machineinstrs < %t/gep.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=GEP
; RUN: not llc -mtriple=capstone64 -verify-machineinstrs < %t/scalar-load.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SCALAR-LOAD
; RUN: not llc -mtriple=capstone64 -verify-machineinstrs < %t/cap-load.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CAP-LOAD

; `not`, NOT `not --crash`. These are target limitations, and they are now reported
; as ordinary errors: report_fatal_error reached the user through clang's crash
; handler, which printed "PLEASE submit a bug report" and a stack dump for what is
; simply an unsupported construct. Same treatment the 128-bit shift already had.
;
; The checks pin the FUNCTION NAME and the offending VALUE, because without them
; the diagnostic says only that some constant somewhere in the module was too wide
; -- which cost a reduction run over a 17 MB module to localise once already.

; DIRECT: error: {{.*}} in function wide_const {{.*}}: Capstone PureCap: Cannot materialize arbitrary >64-bit constants as capabilities; capabilities are unforgeable (value 0x10000000000000000)
; GEP: error: {{.*}} in function wide_gep {{.*}}: Capstone PureCap: CIncOffset displacement must fit in 64 bits (value 0x10000000000000000)
; SCALAR-LOAD: error: {{.*}} in function wide_gep_scalar_load {{.*}}: Capstone PureCap: Address displacement must fit in 64 bits (value 0x10000000000000000)
; CAP-LOAD: error: {{.*}} in function wide_gep_cap_load {{.*}}: Capstone PureCap: Folded load/store displacement must fit in 64 bits (value 0x10000000000000000)
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
