; A pointer global initialized to an *interior* address of another global --
;   const unsigned char *p = &otherGlobal[N];   (non-zero N)
; -- is a ConstantExpr GEP with non-zero indices. The CapstoneCapGlobalInit pass
; must still materialize it as a tagged capability at runtime; otherwise the slot
; loads untagged and faults on first use.
;
; This is SQLite gap 7: sqlite3aLTb/aEQb/aGTb = &sqlite3UpperToLower[256(+6/+12)-OP_Ne].
; SQLite deliberately uses an index that is only in bounds thanks to appended
; array elements, so clang does NOT mark the GEP `inbounds`. The pass therefore
; must peel the constant GEP regardless of the inbounds flag (a plain
; stripPointerCasts / stripInBoundsConstantOffsets would miss it). The stored
; value keeps the full interior offset, so the tagged capability lands at the
; correct cursor.
;
; RUN: llc -mtriple=capstone64 -mattr=+m < %s | FileCheck %s

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

@base = addrspace(200) constant [300 x i8] zeroinitializer, align 16

; Interior pointer WITHOUT `inbounds` -- the exact gap-7 shape.
@p_interior = addrspace(200) global ptr addrspace(200)
    getelementptr (i8, ptr addrspace(200) @base, i64 209), align 16

; The synthesized initializer materializes @base as a tagged capability
; (cincoffset gp / delin / shrink), applies the interior offset (+209), and stores
; the result into the @p_interior slot with a tagged capability store (stc).
; CHECK-LABEL: __capstone_cap_init:
; CHECK: cincoffset {{a[0-9]+}}, gp, {{a[0-9]+}}
; CHECK: delin
; CHECK: cincoffsetimm {{a[0-9]+}}, {{a[0-9]+}}, 209
; CHECK: stc {{a[0-9]+}}, 0(a0)
; CHECK: cjalr zero, 0(ra)

; Registered via the PC-relative .capstone_cap_init table entry.
; CHECK: .section .capstone_cap_init
; CHECK: [[E:.Lcapstone_cap_init_entry[0-9]+]]:
; CHECK-NEXT: .quad __capstone_cap_init-[[E]]

define ptr addrspace(200) @use() addrspace(200) {
  %v = load ptr addrspace(200), ptr addrspace(200) @p_interior
  ret ptr addrspace(200) %v
}
