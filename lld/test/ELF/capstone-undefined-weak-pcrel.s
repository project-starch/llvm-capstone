## C-56. A Capstone image runs at a load base other than its link address and
## is not relocated, so a pc-relative pair cannot form the value 0: an
## undefined weak symbol's address came out as the load delta and
## `if (&weak)` was true. On Capstone lld resolves a %pcrel_hi/%pcrel_lo pair
## that names an undefined weak symbol to the absolute value (0 plus the
## addend) and rewrites the auipc as a lui with the same register. A call
## (R_RISCV_CALL) is unchanged, and so is a defined symbol. RISC-V keeps its
## behaviour.

# RUN: llvm-mc -filetype=obj -triple=capstone64 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=capstone64 -mattr=+relax %s -o %t.relax.o
# RUN: ld.lld %t.relax.o -o %t.relax
# RUN: llvm-objdump -d --no-show-raw-insn %t.relax | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t.rv.o
# RUN: ld.lld %t.rv.o -o %t.rv
# RUN: llvm-objdump -d --no-show-raw-insn %t.rv | FileCheck --check-prefix=RV %s

# CHECK-LABEL: <_start>:
# CHECK-NEXT:    lui a0, 0x0
# CHECK-NEXT:    mv a0, a0
# CHECK:         lui a1, 0x0
# CHECK-NEXT:    addi a1, a1, 0x8
## A defined symbol stays pc-relative.
# CHECK:         auipc a2, 0x0
# CHECK-NEXT:    addi a2, a2, {{.*}}
## A call to an undefined weak symbol is not rewritten (it resolves to itself,
## as on RISC-V: auipc+jalr, or jal once relaxed).
# CHECK-NOT:     lui
# CHECK:         ret

## RISC-V: the pair stays pc-relative (0 - place).
# RV-LABEL:    <_start>:
# RV-NEXT:       auipc a0, 0xfffef
# RV-NEXT:       addi a0, a0, -0x120

.weak undef
.globl _start, defined
_start:
1: auipc a0, %pcrel_hi(undef)
   addi a0, a0, %pcrel_lo(1b)
2: auipc a1, %pcrel_hi(undef + 8)
   addi a1, a1, %pcrel_lo(2b)
3: auipc a2, %pcrel_hi(defined)
   addi a2, a2, %pcrel_lo(3b)
   call undef
defined:
   ret
