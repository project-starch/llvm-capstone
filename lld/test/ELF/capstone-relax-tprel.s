## Local-exec TLS relaxation on Capstone (C-47). The thread pointer is a
## capability and the sequence has no %tprel_add: the integer addi forms the
## offset and cincoffset adds tp. When hi20 is zero, lld removes the lui and
## must rebase the addi on x0, not on tp; rebasing it on tp adds tp twice and
## the access lands out of bounds (the CPython _PyThreadState_Attach fault).

# RUN: llvm-mc -filetype=obj -triple=capstone64 -mattr=+relax %s -o %t.o
# RUN: ld.lld %t.o -o %t.relax
# RUN: llvm-objdump -d --no-show-raw-insn %t.relax | FileCheck --check-prefix=RELAX %s
# RUN: ld.lld --no-relax %t.o -o %t.norelax
# RUN: llvm-objdump -d --no-show-raw-insn %t.norelax | FileCheck --check-prefix=NORELAX %s

# RELAX-LABEL: <bump>:
# RELAX-NOT:     lui
# RELAX-NEXT:    li a0, 0x8
# RELAX-NEXT:    cincoffset a1, tp, a0
# RELAX-NOT:     tp, 0x8

# NORELAX-LABEL: <bump>:
# NORELAX-NEXT:    lui a0, 0x0
# NORELAX-NEXT:    addi a0, a0, 0x8
# NORELAX-NEXT:    cincoffset a1, tp, a0

.globl _start
_start:
  ret

.globl bump
bump:
  lui a0, %tprel_hi(counter)
  addi a0, a0, %tprel_lo(counter)
  cincoffset a1, tp, a0
  ret

.section .tbss,"awT",@nobits
.p2align 3
pad:
  .zero 8
counter:
  .zero 8
