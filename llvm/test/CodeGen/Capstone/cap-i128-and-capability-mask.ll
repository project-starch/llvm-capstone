; RUN: not llc -mtriple=capstone64 -filetype=asm < %s 2>&1 | FileCheck %s

; OPEN, and deliberately pinned as a negative test so that fixing it forces this file to be updated
; rather than leaving a stale comment somewhere.
;
; The C is ordinary: `p = (void *)((uintptr_t)p & ~31)`, i.e. align a pointer down. Clang emits
; exactly that -- ptrtoint to i64, a 64-bit and, zext back to capability width -- and DAGCombiner
; then sinks the zext into the and (the standard `zext (and x, C)` -> `and (zext x), zext(C)`
; transform, which is why only AND is affected and the OR/XOR forms in
; cap-shrink-logic-imm-wide-const.ll select fine). The result is a bitwise AND applied to a
; CAPABILITY rather than to its address, which nothing can select, and which nothing SHOULD select
; silently: masking a capability and handing back an untagged value is the C-16 failure mode.
;
; Fixing this means deciding what masking a capability means, not adding a pattern. Found in
; MicroPython's gc_init.

define ptr addrspace(200) @align_down(ptr addrspace(200) %p) addrspace(200) {
; CHECK: Cannot select: {{.*}} i128 = and
  %i = ptrtoint ptr addrspace(200) %p to i64
  %and = and i64 %i, -32
  %conv = zext i64 %and to i128
  %r = inttoptr i128 %conv to ptr addrspace(200)
  ret ptr addrspace(200) %r
}
