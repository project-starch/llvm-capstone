; Nested capability globals: an array of structs, each holding a function
; pointer and a string pointer (the shape of SQLite's builtin-function table).
; CapstoneCapGlobalInit must recurse into the nested aggregate and materialize
; every capability-pointer leaf in place, not just the two flat shapes (one-field
; struct / flat pointer array). Each struct is two 16-byte capabilities (32 bytes),
; so element i field j lands at offset i*32 + j*16: stores at 0, 16, 32, 48.
; See capstone/agent-handoff/design/capability-globals-init-decision.md.
;
; RUN: llc -mtriple=capstone64 -mattr=+m < %s | FileCheck %s

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

%entry = type { ptr addrspace(200), ptr addrspace(200) }

@.n0 = private addrspace(200) constant [3 x i8] c"ab\00"
@.n1 = private addrspace(200) constant [3 x i8] c"cd\00"

define void @f0() addrspace(200) { ret void }
define void @f1() addrspace(200) { ret void }

@tab = addrspace(200) global [2 x %entry]
    [%entry { ptr addrspace(200) @f0, ptr addrspace(200) @.n0 },
     %entry { ptr addrspace(200) @f1, ptr addrspace(200) @.n1 }], align 16

; The synthesized initializer materializes all four nested capability slots in
; place with tagged capability stores (each derived cincoffset gp / delin).
; CHECK-LABEL: __capstone_cap_init:
; CHECK: cincoffset {{a[0-9]+}}, gp, {{a[0-9]+}}
; CHECK: delin
; CHECK-DAG: stc {{a[0-9]+}}, 0({{[a-z][a-z0-9]*}})
; CHECK-DAG: stc {{a[0-9]+}}, 16({{[a-z][a-z0-9]*}})
; CHECK-DAG: stc {{a[0-9]+}}, 32({{[a-z][a-z0-9]*}})
; CHECK-DAG: stc {{a[0-9]+}}, 48({{[a-z][a-z0-9]*}})
; CHECK: cjalr zero, 0(ra)

; Registered via the PC-relative .capstone_cap_init table entry.
; CHECK: .section .capstone_cap_init
; CHECK: [[E:.Lcapstone_cap_init_entry[0-9]+]]:
; CHECK-NEXT: .quad __capstone_cap_init-[[E]]
