; A capability is unforgeable. This file pins WHERE that is enforced, which the
; register-class split moved.
;
; It used to be enforced by REFUSING to compile, at four sites, because i128 was
; the capability carrier: a 128-bit integer value became the capability
; bit-for-bit, so an arithmetic result wider than the address really could land in
; the metadata half. Refusing was the only defence available.
;
; It is now enforced by the REGISTER FILE. An address is i64 in GPR, a capability
; is c128 in GPCR, and the only path from one to the other is INSERT_SUBREG on
; sub_cap_addr -- a write to the address half that clears the tag. There is no
; instruction sequence in which integer arithmetic reaches the metadata bits, so
; the three arithmetic cases below are no longer refused: they truncate, which is
; what LLVM's inttoptr specifies, and which gep-truncates.ll already documented as
; the defined answer for exactly this value. The file used to give the same
; expression two different answers depending on how it was spelled.
;
; What still cannot be done at all is MATERIALIZING a capability from an arbitrary
; 128-bit constant -- there is no such instruction -- so that guard stays, and
; direct.ll is this file's proof that the refusal machinery still fires.

; RUN: split-file %s %t
; RUN: not --crash llc -mtriple=capstone64 -verify-machineinstrs < %t/direct.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DIRECT
; RUN: llc -mtriple=capstone64 -verify-machineinstrs < %t/incoffset.ll -o - | FileCheck %s --check-prefix=INCOFFSET
; RUN: llc -mtriple=capstone64 -verify-machineinstrs < %t/scalar-load.ll -o - | FileCheck %s --check-prefix=SCALAR-LOAD
; RUN: llc -mtriple=capstone64 -verify-machineinstrs < %t/cap-load.ll -o - | FileCheck %s --check-prefix=CAP-LOAD
; RUN: llc -mtriple=capstone64 -verify-machineinstrs < %t/gep-truncates.ll -o - | FileCheck %s --check-prefix=GEP

; DIRECT: LLVM ERROR: Capstone PureCap: Cannot materialize arbitrary >64-bit constants as capabilities; capabilities are unforgeable

;--- direct.ll
define ptr addrspace(200) @wide_const() {
entry:
  ret ptr addrspace(200) inttoptr (i128 18446744073709551616 to ptr addrspace(200))
}

;--- incoffset.ll
; 2^64 is 0 at address width, so the pointer comes back untouched. Nothing may
; compute an offset from it -- a cincoffset here would mean the high half had been
; taken for a displacement.
; INCOFFSET-LABEL: wide_incoffset:
; INCOFFSET-NOT: cincoffset
; INCOFFSET-NOT: scc
; INCOFFSET: cjalr zero, 0(ra)
define ptr addrspace(200) @wide_incoffset(ptr addrspace(200) %p) {
entry:
  %i = ptrtoint ptr addrspace(200) %p to i128
  %a = add i128 %i, 18446744073709551616
  %q = inttoptr i128 %a to ptr addrspace(200)
  ret ptr addrspace(200) %q
}

;--- scalar-load.ll
; SCALAR-LOAD-LABEL: wide_scalar_load:
; SCALAR-LOAD-NOT: cincoffset
; SCALAR-LOAD: lw a0, 0(a0)
define i32 @wide_scalar_load(ptr addrspace(200) %p) {
entry:
  %i = ptrtoint ptr addrspace(200) %p to i128
  %a = add i128 %i, 18446744073709551616
  %q = inttoptr i128 %a to ptr addrspace(200)
  %v = load i32, ptr addrspace(200) %q, align 4
  ret i32 %v
}

;--- cap-load.ll
; CAP-LOAD-LABEL: wide_cap_load:
; CAP-LOAD-NOT: cincoffset
; CAP-LOAD: ldc a0, 0(a0)
define ptr addrspace(200) @wide_cap_load(ptr addrspace(200) %p) {
entry:
  %i = ptrtoint ptr addrspace(200) %p to i128
  %a = add i128 %i, 18446744073709551616
  %q = inttoptr i128 %a to ptr addrspace(200)
  %v = load ptr addrspace(200), ptr addrspace(200) %q, align 16
  ret ptr addrspace(200) %v
}

;--- gep-truncates.ll
; The shape that always truncated, kept as the control the other three now match.
; GEP-LABEL: wide_gep:
; GEP-NOT: cincoffset
; GEP: cjalr zero, 0(ra)
define ptr addrspace(200) @wide_gep(ptr addrspace(200) %p) {
entry:
  %gep = getelementptr i8, ptr addrspace(200) %p, i128 18446744073709551616
  ret ptr addrspace(200) %gep
}
