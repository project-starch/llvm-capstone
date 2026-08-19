; A large synthetic capability initializer used to become one enormous SelectionDAG.
; Register pressure then spilled live capabilities with scalar sd, dropping their tags;
; a later ldc reloaded an untagged base and the initializer faulted before domain_main.
; The ASM half below is that guard and it is the reason this test exists.
;
; THE IR HALF NOW ASSERTS THE OPPOSITE OF WHAT IT ORIGINALLY DID, on purpose.
; It used to require the initializer to be SPLIT into a basic block every 32 stores.
; That split was reverted because it makes the SQLite silicon domain fault
; deterministically with CAPABLITY_OUT_OF_BOUND inside __capstone_cap_init, before any
; workload runs -- a matched pair, wedged 2 of 2 as merged and passed 3 of 3 with that
; one file reverted. See tests/compiler-repros/C19-capinit-block-split-oob/.
;
; The test was left behind by that revert and had been failing ever since. Rather than
; delete it, the IR half is inverted so it now GUARDS the revert: if the split is
; reintroduced before C-19 is resolved, this fails. Restoring the original checks is
; part of fixing C-19, not something to do to make the suite green.
;
; The two halves are independent, which is what makes this safe: the tag-spill guard
; holds without the split -- measured, 0 scalar sd and 65 stc in the emitted
; initializer -- so dropping the split costs nothing the test was protecting.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -capstone-shrink-globals=false -print-after=capstone-cap-global-init -o /dev/null < %s 2>&1 | FileCheck %s --check-prefix=IR
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

; 65 leaves produce ONE block. cap.init.N would mean the reverted split is back.
; IR-LABEL: define internal void @__capstone_cap_init()
; IR: entry:
; IR-NOT: cap.init.
; IR: ret void

; The regression is the absence of scalar capability spills in the emitted initializer.
; ASM-LABEL: __capstone_cap_init:
; ASM-NOT: sd
; ASM: cjalr zero, 0(ra)
