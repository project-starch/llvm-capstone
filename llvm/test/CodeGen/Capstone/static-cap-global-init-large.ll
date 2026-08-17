; A large synthetic capability initializer used to become one enormous SelectionDAG.
; Register pressure then spilled live capabilities with scalar sd, dropping their tags;
; a later ldc reloaded an untagged base and the initializer faulted before domain_main.
; Keep the synthesized initializer split into bounded IR basic blocks.
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

; 65 leaves produce entry plus cap.init.1 and cap.init.2 at 32-store boundaries.
; IR-LABEL: define internal void @__capstone_cap_init()
; IR: cap.init.1:
; IR: cap.init.2:
; IR: ret void

; The regression is the absence of scalar capability spills in the emitted initializer.
; ASM-LABEL: __capstone_cap_init:
; ASM-NOT: sd
; ASM: cjalr zero, 0(ra)
