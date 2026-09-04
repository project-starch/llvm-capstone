; A global whose initializer is the address of another global cannot be a
; static capability.  The compiler emits the integer address into .data (an
; R_64 relocation) and a per-module __capstone_cap_init routine that mints the
; capability at start-up, registered in .capstone_cap_init as a PC-relative
; entry -- an ADD64/SUB64 pair against the routine and the entry's own
; position.  Pin the relocations of both halves and the routine's store.
; Measured 2026-09-04 on the branch tools.  Relocation TYPE NAMES print as
; "Unknown (N)" until C-37 (lib/Object/ELF.cpp has no EM_CAPSTONE case), so
; each type is matched by its number OR the name it carries once C-37 lands.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -filetype=obj -o %t.o %s
; RUN: llvm-readobj -r --expand-relocs %t.o | FileCheck %s
; RUN: llvm-objdump -d %t.o | FileCheck %s --check-prefix=OBJ

@x = addrspace(200) global i64 7
@px = addrspace(200) global ptr addrspace(200) @x

; x occupies .data+0 (8 bytes); px is 16-byte aligned at .data+0x10 and its
; first quad is the integer address of x.
; CHECK: Section ({{[0-9]+}}) .rela.data {
; CHECK-NEXT: Relocation {
; CHECK-NEXT: Offset: 0x10
; CHECK-NEXT: Type: {{R_Capstone_64|Unknown \(2\)}}
; CHECK-NEXT: Symbol: x (
; CHECK-NEXT: Addend: 0x0
; CHECK-NEXT: }
; CHECK-NEXT: }

; CHECK: Section ({{[0-9]+}}) .rela.capstone_cap_init {
; CHECK-NEXT: Relocation {
; CHECK-NEXT: Offset: 0x0
; CHECK-NEXT: Type: {{R_Capstone_ADD64|Unknown \(36\)}}
; CHECK-NEXT: Symbol: __capstone_cap_init (
; CHECK-NEXT: Addend: 0x0
; CHECK-NEXT: }
; CHECK-NEXT: Relocation {
; CHECK-NEXT: Offset: 0x0
; CHECK-NEXT: Type: {{R_Capstone_SUB64|Unknown \(40\)}}
; CHECK-NEXT: Symbol: .L0
; CHECK-NEXT: Addend: 0x0
; CHECK-NEXT: }
; CHECK-NEXT: }

; The routine derives both px and x from gp, then stores x's capability into
; px with a capability store -- the only stc in the object.
; OBJ-LABEL: <__capstone_cap_init>:
; OBJ: cincoffset a0, gp, a0
; OBJ-NEXT: delin a0
; OBJ: cincoffset a2, gp, a2
; OBJ-NEXT: delin a2
; OBJ: stc a2, 0x0(a0)
; OBJ-NEXT: cjalr zero, 0x0(ra)
