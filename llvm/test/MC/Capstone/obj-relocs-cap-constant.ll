; A constant holding a function's address: the integer address lands in
; .rodata under an R_64 relocation, and __capstone_cap_init materialises the
; function's address PC-relatively (PCREL_HI20 on the symbol, PCREL_LO12_I on
; the auipc's label) to mint the capability it stores over the constant at
; start-up.  Pin all three relocations.  Measured 2026-09-04 on the branch
; tools.  Type NAMES print as "Unknown (N)" until C-37, so each type is
; matched by number OR by the name it carries once C-37 lands.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -filetype=obj -o %t.o %s
; RUN: llvm-readobj -r --expand-relocs %t.o | FileCheck %s

define i64 @f() {
  ret i64 1
}

@fp = addrspace(200) constant ptr addrspace(200) @f

; CHECK: Section ({{[0-9]+}}) .rela.text {
; CHECK: Type: {{R_Capstone_PCREL_HI20|Unknown \(23\)}}
; CHECK-NEXT: Symbol: fp (
; CHECK: Type: {{R_Capstone_PCREL_LO12_I|Unknown \(24\)}}
; CHECK-NEXT: Symbol: .Lpcrel_hi0 (
; CHECK: Type: {{R_Capstone_PCREL_HI20|Unknown \(23\)}}
; CHECK-NEXT: Symbol: f (
; CHECK: Type: {{R_Capstone_PCREL_LO12_I|Unknown \(24\)}}
; CHECK-NEXT: Symbol: .Lpcrel_hi1 (

; CHECK: Section ({{[0-9]+}}) .rela.rodata {
; CHECK-NEXT: Relocation {
; CHECK-NEXT: Offset: 0x0
; CHECK-NEXT: Type: {{R_Capstone_64|Unknown \(2\)}}
; CHECK-NEXT: Symbol: f (
; CHECK-NEXT: Addend: 0x0
; CHECK-NEXT: }
; CHECK-NEXT: }

; CHECK: Section ({{[0-9]+}}) .rela.capstone_cap_init {
; CHECK-NEXT: Relocation {
; CHECK-NEXT: Offset: 0x0
; CHECK-NEXT: Type: {{R_Capstone_ADD64|Unknown \(36\)}}
; CHECK-NEXT: Symbol: __capstone_cap_init (
