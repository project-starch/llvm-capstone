; RUN: llc -mtriple=capstone64-unknown-elf -mattr=+m < %s | FileCheck %s

; A run of adjacent constant stores covering 16 aligned bytes must NOT be merged
; into a single 128-bit store.
;
; On this target i128 is the CAPABILITY carrier, not a 128-bit integer, so a
; 128-bit store is `stc` -- it writes a tagged capability. Merging these six
; ordinary integer stores produces one i128 whose bits are capability metadata
; the program never had authority to name, and ISel then refuses to materialize
; it ("Cannot materialize arbitrary >64-bit constants as capabilities"). Before
; CapstoneTargetLowering::canMergeStoresTo refused the merge, this exact shape --
; SQLite's VdbeOp initialiser, reduced from sqlite3FinishCoding -- made the whole
; SQLite amalgamation fail to build at -O1.
;
; The merged constant this used to produce was 0x10000000000000009: four small
; integers, not an address and certainly not a capability.

define void @vdbeop_init(ptr addrspace(200) %op) {
; CHECK-LABEL: vdbeop_init:
; The stores stay separate and stay INTEGER stores. No capability store, and no
; 128-bit constant to forge.
; CHECK-NOT:  stc {{[a-z0-9]+}}, {{[0-9]+}}(a0)
entry:
  store i8 0, ptr addrspace(200) %op, align 16
  %p2 = getelementptr i8, ptr addrspace(200) %op, i128 2
  store i16 0, ptr addrspace(200) %p2, align 2
  %p4 = getelementptr i8, ptr addrspace(200) %op, i128 4
  store i32 0, ptr addrspace(200) %p4, align 4
  %p8 = getelementptr i8, ptr addrspace(200) %op, i128 8
  store i32 1, ptr addrspace(200) %p8, align 8
  %p12 = getelementptr i8, ptr addrspace(200) %op, i128 12
  store i32 0, ptr addrspace(200) %p12, align 4
  %p1 = getelementptr i8, ptr addrspace(200) %op, i128 1
  store i8 0, ptr addrspace(200) %p1, align 1
  ret void
}
