# REQUIRES: riscv
## PC-relative gp relaxation (--relax-gp), as GNU ld performs it: an auipc and
## its %pcrel_lo partners become gp-relative when the target lies within 2 KiB
## of __global_pointer$, all or nothing per auipc.
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=riscv64-unknown-elf -mattr=+relax a.s -o a.o
# RUN: ld.lld --relax-gp --undefined=__global_pointer$ a.o lds -o out
# RUN: llvm-objdump -d -M no-aliases --no-show-raw-insn out | FileCheck %s
## Without --relax-gp nothing changes.
# RUN: ld.lld --undefined=__global_pointer$ a.o lds -o out0
# RUN: llvm-objdump -d -M no-aliases --no-show-raw-insn out0 | FileCheck %s --check-prefix=OFF

# CHECK-LABEL: <_start>:
## foo: a load and a store through one auipc.
# CHECK-NOT: auipc
# CHECK:  lw a1, -0x800(gp)
# CHECK-NEXT:  sw a1, -0x800(gp)
## bar: an address, at the last reachable byte.
# CHECK-NOT: auipc
# CHECK:  addi a0, gp, 0x7ff
# CHECK-NOT: (gp)
## far: out of reach, unchanged.
# CHECK:  auipc a0,
# CHECK:  lw a1, {{.*}}(a0)
## one %pcrel_lo partner is not relaxable: the auipc and both partners stay.
# CHECK:  auipc a2,
# CHECK:  lw a1, {{.*}}(a2)
# CHECK:  lw a3, {{.*}}(a2)
# CHECK:      <a>:
# CHECK:  addi a0, a0, 0x1
# OFF-LABEL: <_start>:
# OFF-NEXT:  auipc a0,
# OFF-NEXT:  lw a1,
# OFF-NEXT:  sw a1,

#--- a.s
.global _start
_start:
1: auipc a0, %pcrel_hi(foo)
  lw a1, %pcrel_lo(1b)(a0)
  sw a1, %pcrel_lo(1b)(a0)
2: auipc a0, %pcrel_hi(bar)
  addi a0, a0, %pcrel_lo(2b)
3: auipc a0, %pcrel_hi(far)
  lw a1, %pcrel_lo(3b)(a0)
4: auipc a2, %pcrel_hi(foo)
  lw a1, %pcrel_lo(4b)(a2)
.option push
.option norelax
  lw a3, %pcrel_lo(4b)(a2)
.option pop
a:
  addi a0, a0, 1

.section .sdata,"aw"
foo:
  .word 0
  .space 4091
bar:
  .byte 0
far:
  .word 0

#--- lds
SECTIONS {
  .text : {*(.text) }
  .sdata 0x200000 : { }
}
