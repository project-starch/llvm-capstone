; Capability tags cannot live in a static ELF image, so an initialized capability
; global (here an array of addrspace(200) string pointers, the shape of BEEBS
; dtoa's `char *nums[]`) loads untagged and faults on first dereference. The
; CapstoneCapGlobalInit pass synthesizes a per-module __capstone_cap_init that
; stores each element at runtime; normal isel lowers the store to a tagged
; capability store (stc) derived from the global root (cincoffset gp / delin),
; materializing the global in place. The init function has internal linkage (no
; cross-module collision) and is registered via a PC-relative entry in the
; .capstone_cap_init table; start.S iterates that table before domain_main. (A
; fixed external name and a standard .init_array of absolute addresses both fail
; in multi-module domains -- the former collides, the latter is stale because the
; domain processes no load-time relocations.)
; See capstone/agent-handoff/design/capability-globals-init-decision.md.
;
; RUN: llc -mtriple=capstone64 -mattr=+m < %s | FileCheck %s

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

@.s0 = private addrspace(200) constant [3 x i8] c"ab\00"
@.s1 = private addrspace(200) constant [3 x i8] c"cd\00"
@tab = addrspace(200) global [2 x ptr addrspace(200)]
    [ptr addrspace(200) @.s0, ptr addrspace(200) @.s1], align 16

; An LLVM intrinsic global must NOT be materialized (appending linkage / metadata).
@llvm.compiler.used = appending addrspace(200) global [1 x ptr addrspace(200)]
    [ptr addrspace(200) @tab], section "llvm.metadata"

; The synthesized initializer stores both elements with tagged capability stores
; into @tab in place (offsets 0 and 16), each derived as cincoffset gp + delin.
; CHECK-LABEL: __capstone_cap_init:
; CHECK: cincoffset a0, gp, a0
; CHECK: cincoffset a1, gp, a1
; CHECK: delin a1
; CHECK: stc a1, 0(a0)
; CHECK: cincoffset a1, gp, a1
; CHECK: delin a1
; CHECK: stc a1, 16(a0)
; CHECK: cjalr zero, 0(ra)

; A PC-relative table entry registers the (internal) initializer for start.S to
; run -- an offset (initfn - entry), not an absolute/colliding address.
; CHECK: .section .capstone_cap_init
; CHECK: [[E:.Lcapstone_cap_init_entry[0-9]+]]:
; CHECK-NEXT: .quad __capstone_cap_init-[[E]]

define ptr addrspace(200) @get(i64 %i) addrspace(200) {
  %p = getelementptr [2 x ptr addrspace(200)], ptr addrspace(200) @tab, i64 0, i64 %i
  %v = load ptr addrspace(200), ptr addrspace(200) %p
  ret ptr addrspace(200) %v
}
