; C-56: under -capstone-image-gp an extern-weak global's base is formed only
; when its address is not 0. lld resolves an undefined weak symbol's pc-relative
; pair to an absolute 0 (lld/test/ELF/capstone-undefined-weak-pcrel.s), and the
; integer 0 is then the result: an untagged null, not a capability with gp's
; authority whose SHRINK faults. A defined global, a strong declaration and a
; global with a strong declaration keeps the plain sequence. An offset from a
; weak global is applied after the guarded base (on null when undefined, as on
; any null capability).
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -capstone-gp-free -capstone-image-gp -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -capstone-gp-free -capstone-image-gp -verify-machineinstrs < %s | FileCheck %s --check-prefix=O0

@weak_obj = extern_weak addrspace(200) global i32
@big = extern_weak addrspace(200) global [4096 x i8]
@strong_obj = external addrspace(200) global i32
declare extern_weak void @hook() addrspace(200)

; CHECK-LABEL: obj:
; CHECK:       auipc [[A:[a-z0-9]+]], %pcrel_hi(weak_obj)
; CHECK-NEXT:  addi [[A]], [[A]], %pcrel_lo(
; CHECK:       beqz [[A]], [[L:\.Ltmp[0-9]+]]
; CHECK-NEXT:  scc [[R:[a-z0-9]+]], gp, [[A]]
; CHECK-NEXT:  shrinkto [[R]], [[R]], 4
; CHECK-NEXT:  [[L]]:
; O0-LABEL:    obj:
; O0:          beqz [[A:[a-z0-9]+]], [[L:\.Ltmp[0-9]+]]
; O0-NEXT:     scc {{[a-z0-9]+}}, gp, [[A]]
; O0-NEXT:     shrinkto
; O0-NEXT:     [[L]]:
define ptr addrspace(200) @obj() addrspace(200) {
  ret ptr addrspace(200) @weak_obj
}

; A function is not SHRINKed.
; CHECK-LABEL: fn:
; CHECK:       beqz [[A:[a-z0-9]+]], [[L:\.Ltmp[0-9]+]]
; CHECK-NEXT:  scc {{[a-z0-9]+}}, gp, [[A]]
; CHECK-NEXT:  [[L]]:
define ptr addrspace(200) @fn() addrspace(200) {
  ret ptr addrspace(200) @hook
}

; Too large for SHRINKTO's immediate: guarded, not SHRINKed.
; CHECK-LABEL: large:
; CHECK:       beqz [[A:[a-z0-9]+]], [[L:\.Ltmp[0-9]+]]
; CHECK-NEXT:  scc {{[a-z0-9]+}}, gp, [[A]]
; CHECK-NEXT:  [[L]]:
define ptr addrspace(200) @large() addrspace(200) {
  ret ptr addrspace(200) @big
}

; Control: a strong declaration is unguarded and SHRINKed as before.
; CHECK-LABEL: strong:
; CHECK-NOT:   beqz
; CHECK:       scc [[R:[a-z0-9]+]], gp, [[A:[a-z0-9]+]]
; CHECK-NEXT:  shrink [[R]], [[A]],
; CHECK-LABEL: offset:
; CHECK:       beqz [[A:[a-z0-9]+]], [[L:\.Ltmp[0-9]+]]
; CHECK-NEXT:  scc [[R:[a-z0-9]+]], gp, [[A]]
; CHECK-NEXT:  [[L]]:
; CHECK-NEXT:  cincoffsetimm {{[a-z0-9]+}}, [[R]], 16
define ptr addrspace(200) @strong() addrspace(200) {
  ret ptr addrspace(200) @strong_obj
}
define ptr addrspace(200) @offset() addrspace(200) {
  ret ptr addrspace(200) getelementptr (i8, ptr addrspace(200) @big, i64 16)
}
