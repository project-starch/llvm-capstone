# cap-auipcgp: auipcgp rd, imm20 is scc rd, gp, (pc + (imm20 << 12)), a U-type
# in custom-1. Its %pcrel_hi pairs with a %pcrel_lo on cincoffsetimm, or on an
# access, exactly as auipc's does. Without the feature custom-1 is not decoded.
# docs/history/21-11-2026_09-30-00_pcrel-capability-derive-census.md
# RUN: llvm-mc -triple=capstone64 -mattr=+cap-auipcgp -show-encoding %s | FileCheck %s
# RUN: llvm-mc -triple=capstone64 -mattr=+cap-auipcgp -filetype=obj %s -o %t.o
# RUN: llvm-objdump -d -r --mattr=+cap-auipcgp %t.o | FileCheck %s --check-prefix=DIS
# RUN: llvm-objdump -d %t.o | FileCheck %s --check-prefix=STD
# RUN: not llvm-mc -triple=capstone64 %s 2>&1 | FileCheck %s --check-prefix=NOFEAT

# CHECK: auipcgp a0, 0
# CHECK-SAME: encoding: [0x2b,0x05,0x00,0x00]
# NOFEAT: error: instruction requires the following: PC-relative capability derive (auipcgp)
auipcgp a0, 0
# CHECK: auipcgp s1, 1048575
# CHECK-SAME: encoding: [0xab,0xf4,0xff,0xff]
auipcgp s1, 0xfffff

# DIS:      auipcgp a2, 0x0
# DIS-NEXT: R_Capstone_PCREL_HI20 sym
# DIS:      cincoffsetimm a2, a2, 0x0
# DIS-NEXT: R_Capstone_PCREL_LO12_I .Lpcrel_hi0
# DIS:      lw a3, 0x0(a2)
# DIS-NEXT: R_Capstone_PCREL_LO12_I .Lpcrel_hi1
# STD:      2b 05 00 00 <unknown>
.Lpcrel_hi0:
auipcgp a2, %pcrel_hi(sym)
cincoffsetimm a2, a2, %pcrel_lo(.Lpcrel_hi0)
.Lpcrel_hi1:
auipcgp a2, %pcrel_hi(sym)
lw a3, %pcrel_lo(.Lpcrel_hi1)(a2)
