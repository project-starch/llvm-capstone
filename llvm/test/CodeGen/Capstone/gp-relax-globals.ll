; -capstone-mark-gp-relaxable: on a derivation that -capstone-fold-global-lo
; folded (auipc; scc gp; accesses %pcrel_lo), and whose auipc feeds nothing but
; the scc, the scc carries an R_Capstone_GP_SCC relocation against the auipc's
; label, written `scc rd, gp, rs, %gp_scc(label)`. lld --relax-gp may then
; delete the auipc and the scc and rebase the accesses on gp. A derivation the
; fold leaves alone (address taken, two offsets) is never marked. The flag is off
; by default, and off it changes nothing.
; -capstone-small-data-limit=N places defined variables of at most N bytes in
; .sdata/.sbss; constants stay where the image-gp ABI puts them.
; RUN: llc -mtriple=capstone64 -mattr=+m,+relax -capstone-gp-free -capstone-image-gp \
; RUN:   -capstone-shrinkto-globals -capstone-shrinkto-end-adjust=1 -capstone-drop-redundant-shrink \
; RUN:   -capstone-drop-redundant-shrink-declarations \
; RUN:   -capstone-fold-global-lo -capstone-mark-gp-relaxable < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,MARK
; RUN: llc -mtriple=capstone64 -mattr=+m,+relax -capstone-gp-free -capstone-image-gp \
; RUN:   -capstone-shrinkto-globals -capstone-shrinkto-end-adjust=1 -capstone-drop-redundant-shrink \
; RUN:   -capstone-drop-redundant-shrink-declarations \
; RUN:   -capstone-fold-global-lo < %s | FileCheck %s --check-prefixes=CHECK,OFF
; Without the fold there is nothing to mark.
; RUN: llc -mtriple=capstone64 -mattr=+m,+relax -capstone-gp-free -capstone-image-gp \
; RUN:   -capstone-shrinkto-globals -capstone-shrinkto-end-adjust=1 -capstone-drop-redundant-shrink \
; RUN:   -capstone-drop-redundant-shrink-declarations \
; RUN:   -capstone-mark-gp-relaxable < %s | FileCheck %s --check-prefix=NOFOLD
; The relocation, and R_Capstone_RELAX only under +relax.
; RUN: llc -mtriple=capstone64 -mattr=+m,+relax -capstone-gp-free -capstone-image-gp \
; RUN:   -capstone-shrinkto-globals -capstone-shrinkto-end-adjust=1 -capstone-drop-redundant-shrink \
; RUN:   -capstone-drop-redundant-shrink-declarations \
; RUN:   -capstone-fold-global-lo -capstone-mark-gp-relaxable -filetype=obj < %s -o %t.relax.o
; RUN: llvm-readelf -r %t.relax.o | FileCheck %s --check-prefix=RELOC
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-free -capstone-image-gp \
; RUN:   -capstone-shrinkto-globals -capstone-shrinkto-end-adjust=1 -capstone-drop-redundant-shrink \
; RUN:   -capstone-drop-redundant-shrink-declarations \
; RUN:   -capstone-fold-global-lo -capstone-mark-gp-relaxable -filetype=obj < %s -o %t.norelax.o
; RUN: llvm-readelf -r %t.norelax.o | FileCheck %s --check-prefix=NORELAX
; Small data.
; RUN: llc -mtriple=capstone64 -mattr=+m,+relax -capstone-gp-free -capstone-image-gp \
; RUN:   -capstone-shrinkto-globals -capstone-shrinkto-end-adjust=1 -capstone-drop-redundant-shrink \
; RUN:   -capstone-drop-redundant-shrink-declarations \
; RUN:   -capstone-small-data-limit=16 < %s | FileCheck %s --check-prefix=SDATA
; RUN: llc -mtriple=capstone64 -mattr=+m,+relax -capstone-gp-free -capstone-image-gp \
; RUN:   -capstone-shrinkto-globals -capstone-shrinkto-end-adjust=1 -capstone-drop-redundant-shrink \
; RUN:   -capstone-drop-redundant-shrink-declarations \
; RUN:   < %s | FileCheck %s --check-prefix=NOSDATA

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
@x = addrspace(200) global i32 0, align 4
@p = addrspace(200) global ptr addrspace(200) null, align 16
@a = addrspace(200) global [4 x i32] zeroinitializer, align 4
@big = addrspace(200) global [8 x i32] zeroinitializer, align 4
@init = addrspace(200) global i64 7, align 8
declare void @use(ptr addrspace(200)) addrspace(200)

; A load through the folded base: marked.
; CHECK-LABEL: load_word:
; MARK:      .Lpcrel_hi[[N:[0-9]+]]:
; MARK-NEXT: auipc [[HI:[a-z0-9]+]], %pcrel_hi(x)
; MARK-NEXT: scc [[CAP:[a-z0-9]+]], gp, [[HI]], %gp_scc(.Lpcrel_hi[[N]])
; MARK-NEXT: lw a0, %pcrel_lo(.Lpcrel_hi[[N]])([[CAP]])
; OFF:       .Lpcrel_hi[[M:[0-9]+]]:
; OFF-NEXT:  auipc [[HI2:[a-z0-9]+]], %pcrel_hi(x)
; OFF-NEXT:  scc [[CAP2:[a-z0-9]+]], gp, [[HI2]]{{$}}
; OFF-NEXT:  lw a0, %pcrel_lo(.Lpcrel_hi[[M]])([[CAP2]])
; NOFOLD-LABEL: load_word:
; NOFOLD-NOT: %gp_scc
; NOFOLD:     ret
define i32 @load_word() addrspace(200) {
  %v = load i32, ptr addrspace(200) @x
  ret i32 %v
}

; A load and a store at one common offset: one marked scc for both.
; CHECK-LABEL: rmw_elem:
; MARK:      auipc [[HI:[a-z0-9]+]], %pcrel_hi(a+12)
; MARK-NEXT: scc [[CAP:[a-z0-9]+]], gp, [[HI]], %gp_scc(
; MARK:      lw {{[a-z0-9]+}}, %pcrel_lo({{.*}})([[CAP]])
; MARK:      sw {{[a-z0-9]+}}, %pcrel_lo({{.*}})([[CAP]])
; OFF-NOT:   %gp_scc
define void @rmw_elem() addrspace(200) {
  %q = getelementptr [4 x i32], ptr addrspace(200) @a, i64 0, i64 3
  %v = load i32, ptr addrspace(200) %q
  %w = add i32 %v, 1
  store i32 %w, ptr addrspace(200) %q
  ret void
}

; The address escapes: the fold keeps the whole address, and nothing is marked.
; CHECK-LABEL: pass_addr:
; MARK-NOT:  %gp_scc
; MARK:      jalr
define void @pass_addr() addrspace(200) {
  call void @use(ptr addrspace(200) @x)
  ret void
}

; Two offsets of one global: not folded, not marked.
; CHECK-LABEL: two_offsets:
; MARK-NOT:  %gp_scc
; MARK:      ret
define i32 @two_offsets() addrspace(200) {
  %q1 = getelementptr [4 x i32], ptr addrspace(200) @a, i64 0, i64 1
  %q2 = getelementptr [4 x i32], ptr addrspace(200) @a, i64 0, i64 2
  %v1 = load i32, ptr addrspace(200) %q1
  %v2 = load i32, ptr addrspace(200) %q2
  %s = add i32 %v1, %v2
  ret i32 %s
}

; RELOC:      R_Capstone_PCREL_HI20 {{.*}} x + 0
; RELOC-NEXT: R_Capstone_RELAX
; RELOC-NEXT: R_Capstone_GP_SCC {{.*}} .Lpcrel_hi0 + 0
; RELOC-NEXT: R_Capstone_RELAX
; RELOC-NEXT: R_Capstone_PCREL_LO12_I {{.*}} .Lpcrel_hi0 + 0
; RELOC-NEXT: R_Capstone_RELAX
; NORELAX:     R_Capstone_GP_SCC {{.*}} .Lpcrel_hi0 + 0
; NORELAX-NOT: R_Capstone_RELAX

; 4- and 16-byte variables are small data at 16, a 32-byte one is not; an
; initialised one goes to .sdata.
; SDATA:      .type x,@object
; SDATA-NEXT: .section .sbss,"aw",@nobits
; SDATA:      p:
; SDATA:      a:
; SDATA:      .type big,@object
; SDATA-NEXT: .bss
; SDATA:      .type init,@object
; SDATA-NEXT: .section .sdata,"aw",@progbits
; NOSDATA-NOT: .sbss
; NOSDATA-NOT: .sdata
