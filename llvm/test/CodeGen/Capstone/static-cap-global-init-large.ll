; A large synthetic capability initializer used to become one enormous SelectionDAG.
; Register pressure then spilled live capabilities with scalar sd, dropping their tags;
; a later ldc reloaded an untagged base and the initializer faulted before domain_main.
;
; THE IR ARM WAS REMOVED 2026-08-18 AND THE ASM ARM IS THE POINT.
;
; This test used to also assert that the initializer was split into `cap.init.N` basic
; blocks every 32 stores. That split was REVERTED as C-19: it broke silicon
; deterministically, raising CAPABLITY_OUT_OF_BOUND inside __capstone_cap_init before any
; workload ran (`capstone/tests/compiler-repros/C19-capinit-block-split-oob/`). Asserting
; it here would pin a behaviour the target no longer has, and a test that fails by design
; teaches the next reader to ignore a red result.
;
; What actually guards the defect is the ASM arm below, and it still holds WITHOUT the
; split: no scalar `sd` appears inside __capstone_cap_init, so no live capability is
; spilled in a way that drops its tag. Verified on the reverted compiler before this edit.
;
; If the split is ever reintroduced -- the plan is one piece per synthesis, smallest first
; -- restore an IR arm then, against whatever shape it actually takes.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -capstone-shrink-globals=false < %s | FileCheck %s --check-prefix=ASM

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

@.s = private addrspace(200) constant [2 x i8] c"x\00"
@tab = addrspace(200) global [65 x ptr addrspace(200)] [
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s,
     ptr addrspace(200) @.s], align 16

; 65 leaves: deliberately more than the 32 that used to trigger a block split, so this stays
; a large-initializer case even though the split itself is gone.

; The regression is the absence of scalar capability spills in the emitted initializer.
; ASM-LABEL: __capstone_cap_init:
; ASM-NOT: sd
; ASM: cjalr zero, 0(ra)
