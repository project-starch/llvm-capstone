; A constant holding a function's address: the integer address lands in
; .rodata under an R_64 relocation, and __capstone_cap_init materialises the
; function's address PC-relatively (PCREL_HI20 on the symbol, PCREL_LO12_I on
; the auipc's label) to mint the capability it stores over the constant at
; start-up.  Pin all three relocations.  Measured 2026-09-04 on the branch
; tools.  Types are matched by NAME: C-37 (fixed 2026-09-09) gave lib/Object an
; EM_CAPSTONE case, so a tool printing "Unknown (N)" here has lost it.  The
; readelf arm pins the other half of C-37: the file header names the machine
; ("Capstone", e_machine 259) instead of printing its number.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -filetype=obj -o %t.o %s
; RUN: llvm-readobj -r --expand-relocs %t.o | FileCheck %s
; RUN: llvm-readelf -h -r %t.o | FileCheck --check-prefixes=READELF,NOUNK %s
;
; READELF checks follow the output order (header, .rela.text, .rela.rodata);
; NOUNK is a prefix of nothing but a NOT, so it covers the whole output.
; READELF: Machine:{{.*}}Capstone
; READELF: R_Capstone_PCREL_HI20
; READELF: R_Capstone_64
; NOUNK-NOT: Unknown

define i64 @f() {
  ret i64 1
}

@fp = addrspace(200) constant ptr addrspace(200) @f

; CHECK: Section ({{[0-9]+}}) .rela.text {
; CHECK: Type: R_Capstone_PCREL_HI20
; CHECK-NEXT: Symbol: fp (
; CHECK: Type: R_Capstone_PCREL_LO12_I
; CHECK-NEXT: Symbol: .Lpcrel_hi0 (
; CHECK: Type: R_Capstone_PCREL_HI20
; CHECK-NEXT: Symbol: f (
; CHECK: Type: R_Capstone_PCREL_LO12_I
; CHECK-NEXT: Symbol: .Lpcrel_hi1 (

; CHECK: Section ({{[0-9]+}}) .rela.rodata {
; CHECK-NEXT: Relocation {
; CHECK-NEXT: Offset: 0x0
; CHECK-NEXT: Type: R_Capstone_64
; CHECK-NEXT: Symbol: f (
; CHECK-NEXT: Addend: 0x0
; CHECK-NEXT: }
; CHECK-NEXT: }

; CHECK: Section ({{[0-9]+}}) .rela.capstone_cap_init {
; CHECK-NEXT: Relocation {
; CHECK-NEXT: Offset: 0x0
; CHECK-NEXT: Type: R_Capstone_ADD64
; CHECK-NEXT: Symbol: __capstone_cap_init (
