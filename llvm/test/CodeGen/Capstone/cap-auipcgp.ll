; cap-auipcgp: under the image-gp ABI a global's base `auipc; [addi;] scc gp`
; becomes `auipcgp; [cincoffsetimm]`, one instruction fewer, with gp's
; authority. Without the feature, or without image-gp, nothing changes; an
; extern-weak global keeps C-56's null-capable base.
; RUN: llc -mtriple=capstone64 -mattr=+m,+cap-auipcgp -capstone-gp-free -capstone-shrinkto-globals -capstone-shrinkto-end-adjust=1 -capstone-drop-redundant-shrink -capstone-drop-redundant-shrink-declarations -capstone-image-gp \
; RUN:   -capstone-fold-global-lo -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,FOLD
; RUN: llc -mtriple=capstone64 -mattr=+m,+cap-auipcgp -capstone-gp-free -capstone-shrinkto-globals -capstone-shrinkto-end-adjust=1 -capstone-drop-redundant-shrink -capstone-drop-redundant-shrink-declarations -capstone-image-gp \
; RUN:   -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,KEEP
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-free -capstone-shrinkto-globals -capstone-shrinkto-end-adjust=1 -capstone-drop-redundant-shrink -capstone-drop-redundant-shrink-declarations -capstone-image-gp \
; RUN:   -capstone-fold-global-lo < %s | FileCheck %s --check-prefixes=CHECK,OFF
; RUN: llc -mtriple=capstone64 -mattr=+m,+cap-auipcgp -capstone-gp-free -capstone-shrinkto-globals -capstone-shrinkto-end-adjust=1 -capstone-drop-redundant-shrink -capstone-drop-redundant-shrink-declarations \
; RUN:   < %s | FileCheck %s --check-prefixes=CHECK,OFF
; The label moves with the derive: the object's relocations name it.
; RUN: llc -mtriple=capstone64 -mattr=+m,+cap-auipcgp -capstone-gp-free -capstone-shrinkto-globals -capstone-shrinkto-end-adjust=1 -capstone-drop-redundant-shrink -capstone-drop-redundant-shrink-declarations -capstone-image-gp \
; RUN:   -filetype=obj < %s -o %t.o
; RUN: llvm-objdump -dr --mattr=+cap-auipcgp %t.o | FileCheck %s --check-prefix=OBJ

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
@x = addrspace(200) global i32 0, align 4
@w = extern_weak addrspace(200) global i32
declare void @use(ptr addrspace(200)) addrspace(200)

; CHECK-LABEL: load_word:
; FOLD:      .Lpcrel_hi[[N:[0-9]+]]:
; FOLD-NEXT: auipcgp [[CAP:[a-z0-9]+]], %pcrel_hi(x)
; FOLD-NEXT: lw a0, %pcrel_lo(.Lpcrel_hi[[N]])([[CAP]])
; KEEP:      .Lpcrel_hi[[M:[0-9]+]]:
; KEEP-NEXT: auipcgp [[C:[a-z0-9]+]], %pcrel_hi(x)
; KEEP-NEXT: cincoffsetimm [[C]], [[C]], %pcrel_lo(.Lpcrel_hi[[M]])
; KEEP-NEXT: lw a0, 0([[C]])
; OFF-NOT:   auipcgp
; OFF:       auipc
; OFF:       scc
; OBJ-LABEL: <load_word>:
; OBJ:       auipcgp a0, 0x0
; OBJ-NEXT:  R_Capstone_PCREL_HI20 x
; OBJ:       cincoffsetimm a0, a0, 0x0
; OBJ-NEXT:  R_Capstone_PCREL_LO12_I .Lpcrel_hi0
define i32 @load_word() addrspace(200) {
  %v = load i32, ptr addrspace(200) @x
  ret i32 %v
}

; The capability is passed on: the complete address, under both RUNs.
; CHECK-LABEL: pass_addr:
; FOLD:      .Lpcrel_hi[[P:[0-9]+]]:
; FOLD-NEXT: auipcgp [[C2:[a-z0-9]+]], %pcrel_hi(x)
; FOLD:      cincoffsetimm [[C2]], [[C2]], %pcrel_lo(.Lpcrel_hi[[P]])
; KEEP:      .Lpcrel_hi[[Q:[0-9]+]]:
; KEEP-NEXT: auipcgp [[C3:[a-z0-9]+]], %pcrel_hi(x)
; KEEP:      cincoffsetimm [[C3]], [[C3]], %pcrel_lo(.Lpcrel_hi[[Q]])
; OFF-NOT:   auipcgp
define void @pass_addr() addrspace(200) {
  call void @use(ptr addrspace(200) @x)
  ret void
}

; Extern weak: C-56's base (beqz over the scc), never auipcgp.
; CHECK-LABEL: weak_addr:
; CHECK-NOT: auipcgp
; FOLD:      beqz
; KEEP:      beqz
; CHECK:     ret
define void @weak_addr() addrspace(200) {
  call void @use(ptr addrspace(200) @w)
  ret void
}
