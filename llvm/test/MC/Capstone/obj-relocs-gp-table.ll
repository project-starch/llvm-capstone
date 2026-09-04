; Under -capstone-gp-captable every global is reached through a capability
; table indexed off gp (`ldc rd, 16*i(gp)`), with no gp-derived cincoffset and
; NO delin.  The object carries two extra sections: .capstone_gp_table, one
; 24-byte entry per global whose address field is an ADD64/SUB64 pair against
; the global and the entry's own position, and .capstone_gp_initdesc, whose
; address fields are ADD64/SUB64 pairs against the global and the undefined
; symbol __gpfree_globals_base that the glue's cap-table generator defines.
; Pin the pairs, the table offsets and the ldc sequence.  Measured 2026-09-04
; on the branch tools.  Type NAMES print as "Unknown (N)" until C-37, so each
; type is matched by number OR by the name it carries once C-37 lands.
;
; MUTATION: drop -capstone-gp-captable from the object RUN line -> the same
; module's default-ABI object has no .rela.capstone_gp_table section, `use`
; starts with auipc/cincoffset/delin, and the implicit-check-not on delin
; fires (performed 2026-09-04 on the default-ABI object of this module).
;
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-captable -filetype=obj -o %t.o %s
; RUN: llvm-readobj -r --expand-relocs %t.o | FileCheck %s
; RUN: llvm-objdump -d %t.o | FileCheck %s --check-prefix=OBJ --implicit-check-not=delin

@x = addrspace(200) global i64 7
@px = addrspace(200) global ptr addrspace(200) @x

define i64 @f() {
  ret i64 1
}

@fp = addrspace(200) constant ptr addrspace(200) @f

define i64 @use() {
  %p = load ptr addrspace(200), ptr addrspace(200) @px
  %v = load i64, ptr addrspace(200) %p
  ret i64 %v
}

; CHECK: Section ({{[0-9]+}}) .rela.capstone_gp_table {
; CHECK-NEXT: Relocation {
; CHECK-NEXT: Offset: 0x18
; CHECK-NEXT: Type: {{R_Capstone_ADD64|Unknown \(36\)}}
; CHECK-NEXT: Symbol: x (
; CHECK-NEXT: Addend: 0x0
; CHECK-NEXT: }
; CHECK-NEXT: Relocation {
; CHECK-NEXT: Offset: 0x18
; CHECK-NEXT: Type: {{R_Capstone_SUB64|Unknown \(40\)}}
; CHECK-NEXT: Symbol: .L0
; CHECK-NEXT: Addend: 0x0
; CHECK-NEXT: }
; CHECK-NEXT: Relocation {
; CHECK-NEXT: Offset: 0x30
; CHECK-NEXT: Type: {{R_Capstone_ADD64|Unknown \(36\)}}
; CHECK-NEXT: Symbol: px (
; CHECK-NEXT: Addend: 0x0
; CHECK-NEXT: }
; CHECK-NEXT: Relocation {
; CHECK-NEXT: Offset: 0x30
; CHECK-NEXT: Type: {{R_Capstone_SUB64|Unknown \(40\)}}
; CHECK-NEXT: Symbol: .L0
; CHECK-NEXT: Addend: 0x0
; CHECK-NEXT: }
; CHECK-NEXT: Relocation {
; CHECK-NEXT: Offset: 0x48
; CHECK-NEXT: Type: {{R_Capstone_ADD64|Unknown \(36\)}}
; CHECK-NEXT: Symbol: fp (
; CHECK-NEXT: Addend: 0x0
; CHECK-NEXT: }
; CHECK-NEXT: Relocation {
; CHECK-NEXT: Offset: 0x48
; CHECK-NEXT: Type: {{R_Capstone_SUB64|Unknown \(40\)}}
; CHECK-NEXT: Symbol: .L0
; CHECK-NEXT: Addend: 0x0
; CHECK-NEXT: }
; CHECK-NEXT: }

; CHECK: Section ({{[0-9]+}}) .rela.capstone_gp_initdesc {
; CHECK-NEXT: Relocation {
; CHECK-NEXT: Offset: 0x30
; CHECK-NEXT: Type: {{R_Capstone_ADD64|Unknown \(36\)}}
; CHECK-NEXT: Symbol: x (
; CHECK-NEXT: Addend: 0x0
; CHECK-NEXT: }
; CHECK-NEXT: Relocation {
; CHECK-NEXT: Offset: 0x30
; CHECK-NEXT: Type: {{R_Capstone_SUB64|Unknown \(40\)}}
; CHECK-NEXT: Symbol: __gpfree_globals_base (
; CHECK-NEXT: Addend: 0x0
; CHECK-NEXT: }
; CHECK-NEXT: Relocation {
; CHECK-NEXT: Offset: 0x48
; CHECK-NEXT: Type: {{R_Capstone_ADD64|Unknown \(36\)}}
; CHECK-NEXT: Symbol: px (
; CHECK: Offset: 0x60
; CHECK-NEXT: Type: {{R_Capstone_ADD64|Unknown \(36\)}}
; CHECK-NEXT: Symbol: fp (
; CHECK: Symbol: __gpfree_globals_base (

; x is table entry 0, px entry 1 (0x10(gp)), fp entry 2 (0x20(gp)).
; OBJ-LABEL: <use>:
; OBJ-NEXT: ldc a0, 0x10(gp)
; OBJ-NEXT: ldc a0, 0x0(a0)
; OBJ-NEXT: ld a0, 0x0(a0)
; OBJ-NEXT: ret
; OBJ-LABEL: <__capstone_cap_init>:
; OBJ-NEXT: ldc a0, 0x10(gp)
; OBJ-NEXT: ldc a1, 0x0(gp)
; OBJ-NEXT: ldc a2, 0x20(gp)
