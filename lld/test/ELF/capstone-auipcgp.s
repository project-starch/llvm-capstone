## cap-auipcgp: auipcgp takes auipc's R_RISCV_PCREL_HI20 and its %pcrel_lo
## partner unchanged. Against an undefined symbol there is no null form of
## it (C-56 rewrites only auipc), so lld refuses it.

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=capstone64 -mattr=+cap-auipcgp %t/def.s -o %t/def.o
# RUN: ld.lld %t/def.o -o %t/def
# RUN: llvm-objdump -d --no-show-raw-insn --mattr=+cap-auipcgp %t/def | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=capstone64 -mattr=+cap-auipcgp %t/undef.s -o %t/undef.o
# RUN: not ld.lld %t/undef.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

# CHECK-LABEL: <_start>:
# CHECK-NEXT:    auipcgp a0, 0x{{[0-9a-f]+}}
# CHECK-NEXT:    cincoffsetimm a0, a0, 0x{{[0-9a-f]+}}
# ERR: error: {{.*}}auipcgp against undefined symbol undef

#--- def.s
.globl _start
_start:
.Lpcrel_hi0:
  auipcgp a0, %pcrel_hi(x)
  cincoffsetimm a0, a0, %pcrel_lo(.Lpcrel_hi0)
  ret
.data
.p2align 12
.zero 16
x: .word 1

#--- undef.s
.weak undef
.globl _start
_start:
.Lpcrel_hi0:
  auipcgp a0, %pcrel_hi(undef)
  cincoffsetimm a0, a0, %pcrel_lo(.Lpcrel_hi0)
  ret
