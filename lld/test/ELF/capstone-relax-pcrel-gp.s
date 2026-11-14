## Capstone PC-relative gp relaxation (--relax-gp): `auipc rs, %pcrel_hi(x);
## scc cap, gp, rs, %gp_scc(label); access %pcrel_lo(label)(cap)` becomes
## `access x - __global_pointer$(gp)` when x lies within 2 KiB of
## __global_pointer$. The auipc needs the scc's R_Capstone_GP_SCC marker, every
## relocation of the derivation must be relaxable, and both instructions go.
## Unmarked Capstone code is never relaxed.
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=capstone64 -mattr=+relax a.s -o a.o
# RUN: llvm-readelf -r a.o | FileCheck %s --check-prefix=RELOC
# RUN: ld.lld --relax-gp --undefined=__global_pointer$ a.o lds -o out
# RUN: llvm-objdump -d -M no-aliases --no-show-raw-insn out | FileCheck %s
## Without --relax-gp nothing changes.
# RUN: ld.lld --undefined=__global_pointer$ a.o lds -o out0
# RUN: llvm-objdump -d -M no-aliases --no-show-raw-insn out0 | FileCheck %s --check-prefix=OFF

# RELOC: R_Capstone_GP_SCC {{.*}} .Ltmp0 + 0
# RELOC-NEXT: R_Capstone_RELAX

# CHECK-LABEL: <_start>:
## foo: a marked derivation with a load and a store: both rebased on gp, the
## auipc and the scc gone.
# CHECK-NOT: auipc
# CHECK-NOT: scc
# CHECK:  ld a2, -0x800(gp)
# CHECK-NEXT:  sd a2, -0x800(gp)
## bar, at the last reachable byte: rebased.
# CHECK-NOT: auipc
# CHECK-NOT: scc
# CHECK:  lb a2, 0x7ff(gp)
# CHECK-NOT: (gp)
## foo again, unmarked: unchanged.
# CHECK:  auipc a0,
# CHECK:  scc a1, gp, a0
# CHECK:  ld a2, {{.*}}(a1)
## far, marked but out of reach: unchanged.
# CHECK:  auipc a0,
# CHECK:  scc a1, gp, a0
# CHECK:  ld a2, {{.*}}(a1)
## An integer address on Capstone (no marker): unchanged.
# CHECK:  auipc a0,
# CHECK:  addi a0, a0,
## A marked derivation with a partner that is not relaxable: unchanged.
# CHECK:  auipc a0,
# CHECK:  scc a1, gp, a0
# CHECK:  ld a2, {{.*}}(a1)
# CHECK:  ld a3, {{.*}}(a1)
# CHECK:      <a>:
# CHECK:  addi a0, a0, 0x1
# OFF-LABEL: <_start>:
# OFF-NEXT:  auipc a0,
# OFF-NEXT:  scc a1, gp, a0
# OFF-NEXT:  ld a2,

#--- a.s
.global _start
_start:
.Ltmp0: auipc a0, %pcrel_hi(foo)
  scc a1, gp, a0, %gp_scc(.Ltmp0)
  ld a2, %pcrel_lo(.Ltmp0)(a1)
  sd a2, %pcrel_lo(.Ltmp0)(a1)
.Ltmp1: auipc a0, %pcrel_hi(bar)
  scc a1, gp, a0, %gp_scc(.Ltmp1)
  lb a2, %pcrel_lo(.Ltmp1)(a1)
.Ltmp2: auipc a0, %pcrel_hi(foo)
  scc a1, gp, a0
  ld a2, %pcrel_lo(.Ltmp2)(a1)
.Ltmp3: auipc a0, %pcrel_hi(far)
  scc a1, gp, a0, %gp_scc(.Ltmp3)
  ld a2, %pcrel_lo(.Ltmp3)(a1)
.Ltmp4: auipc a0, %pcrel_hi(foo)
  addi a0, a0, %pcrel_lo(.Ltmp4)
.Ltmp5: auipc a0, %pcrel_hi(foo)
  scc a1, gp, a0, %gp_scc(.Ltmp5)
  ld a2, %pcrel_lo(.Ltmp5)(a1)
.option push
.option norelax
  ld a3, %pcrel_lo(.Ltmp5)(a1)
.option pop
a:
  addi a0, a0, 1

.section .sdata,"aw"
foo:
  .quad 0
  .space 4087
bar:
  .byte 0
far:
  .quad 0

#--- lds
SECTIONS {
  .text : {*(.text) }
  .sdata 0x200000 : { }
}
