# Every Capstone capability instruction, every operand form the assembler
# accepts, and the exact encoding each produces.  Measured 2026-09-04 on the
# branch llvm-mc (the first assembler test this target has had).  Immediates
# are pinned at both ends of their range; registers are given by ABI name and
# by x-number (the x-number forms print as ABI names -- see cap-regnames.s).
#
# The lcc and tighten fields are uimm5 in the encoding and encode 31 even
# though the silicon accepts far less (lcc selectors 0-5; tighten 0-7 -- above
# 7 the RTL raises ILLEGAL_OPERAND_VALUE).  The assembler's job is the field;
# the value question belongs to Sema (Tier 4 of the validation plan).
#
# MUTATION: change the encoding of `ldc a0, 0(a1)` below to
# [0x5b,0xb5,0x05,0x01] -> the CHECK fails (performed 2026-09-04).
#
# RUN: llvm-mc -triple capstone64 -show-encoding %s | FileCheck %s

# CHECK: ldc a0, 0(a1)  # encoding: [0x5b,0xb5,0x05,0x00]
ldc a0, 0(a1)
# CHECK: ldc a0, -2048(a1)  # encoding: [0x5b,0xb5,0x05,0x80]
ldc a0, -2048(a1)
# CHECK: ldc a0, 2047(a1)  # encoding: [0x5b,0xb5,0xf5,0x7f]
ldc a0, 2047(a1)
# CHECK: ldc a0, 8(a1)  # encoding: [0x5b,0xb5,0x85,0x00]
ldc x10, 8(x11)
# CHECK: ldc s0, 16(sp)  # encoding: [0x5b,0x34,0x01,0x01]
ldc s0, 16(sp)

# CHECK: stc a0, 0(a1)  # encoding: [0x5b,0xc0,0xa5,0x00]
stc a0, 0(a1)
# CHECK: stc a0, -2048(a1)  # encoding: [0x5b,0xc0,0xa5,0x80]
stc a0, -2048(a1)
# CHECK: stc a0, 2047(a1)  # encoding: [0xdb,0xcf,0xa5,0x7e]
stc a0, 2047(a1)
# CHECK: stc a0, 8(a1)  # encoding: [0x5b,0xc4,0xa5,0x00]
stc x10, 8(x11)
# CHECK: stc ra, 32(sp)  # encoding: [0x5b,0x40,0x11,0x02]
stc ra, 32(sp)

# CHECK: cincoffset a0, a1, a2  # encoding: [0x5b,0x95,0xc5,0x18]
cincoffset a0, a1, a2
# CHECK: cincoffset a0, gp, a1  # encoding: [0x5b,0x95,0xb1,0x18]
cincoffset a0, gp, a1

# CHECK: cincoffsetimm a0, a1, 0  # encoding: [0x5b,0xa5,0x05,0x00]
cincoffsetimm a0, a1, 0
# CHECK: cincoffsetimm a0, a1, -2048  # encoding: [0x5b,0xa5,0x05,0x80]
cincoffsetimm a0, a1, -2048
# CHECK: cincoffsetimm a0, a1, 2047  # encoding: [0x5b,0xa5,0xf5,0x7f]
cincoffsetimm a0, a1, 2047
# CHECK: cincoffsetimm sp, sp, -16  # encoding: [0x5b,0x21,0x01,0xff]
cincoffsetimm sp, sp, -16

# CHECK: lcc a0, a1, 0  # encoding: [0x5b,0x95,0x05,0x08]
lcc a0, a1, 0
# CHECK: lcc a0, a1, 1  # encoding: [0x5b,0x95,0x15,0x08]
lcc a0, a1, 1
# CHECK: lcc a0, a1, 2  # encoding: [0x5b,0x95,0x25,0x08]
lcc a0, a1, 2
# CHECK: lcc a0, a1, 5  # encoding: [0x5b,0x95,0x55,0x08]
lcc a0, a1, 5
# CHECK: lcc a0, a1, 31  # encoding: [0x5b,0x95,0xf5,0x09]
lcc a0, a1, 31

# CHECK: shrink a0, a1, a2  # encoding: [0x5b,0x95,0xc5,0x02]
shrink a0, a1, a2

# CHECK: movc a0, a1  # encoding: [0x5b,0x95,0x05,0x14]
movc a0, a1
# CHECK: movc a0, zero  # encoding: [0x5b,0x15,0x00,0x14]
movc x10, x0
# CHECK: movc s0, sp  # encoding: [0x5b,0x14,0x01,0x14]
movc s0, sp

# CHECK: tighten a0, a1, 0  # encoding: [0x5b,0x95,0x05,0x04]
tighten a0, a1, 0
# CHECK: tighten a0, a1, 7  # encoding: [0x5b,0x95,0x75,0x04]
tighten a0, a1, 7
# CHECK: tighten a0, a1, 31  # encoding: [0x5b,0x95,0xf5,0x05]
tighten a0, a1, 31

# CHECK: scc a0, a1, a2  # encoding: [0x5b,0x95,0xc5,0x0a]
scc a0, a1, a2
# CHECK: init a0, a1, a2  # encoding: [0x5b,0x95,0xc5,0x12]
init a0, a1, a2

# CHECK: delin a0  # encoding: [0x5b,0x15,0x00,0x06]
delin a0
# CHECK: delin t6  # encoding: [0xdb,0x1f,0x00,0x06]
delin x31

# CHECK: mrev a0, a1  # encoding: [0x5b,0x95,0x05,0x10]
mrev a0, a1
# CHECK: mrev a1, a1  # encoding: [0xdb,0x95,0x05,0x10]
mrev a1, a1

# CHECK: seal a0, a1  # encoding: [0x5b,0x95,0x05,0x0e]
seal a0, a1
# CHECK: drop a0  # encoding: [0x5b,0x10,0x05,0x16]
drop a0
# CHECK: revoke a0  # encoding: [0x5b,0x10,0x05,0x00]
revoke a0

# CHECK: cjalr ra, 0(a0)  # encoding: [0xdb,0x50,0x05,0x00]
cjalr ra, 0(a0)
# CHECK: cjalr zero, 0(ra)  # encoding: [0x5b,0xd0,0x00,0x00]
cjalr zero, 0(ra)
# CHECK: cjalr ra, -2048(a0)  # encoding: [0xdb,0x50,0x05,0x80]
cjalr ra, -2048(a0)
# CHECK: cjalr ra, 2047(a0)  # encoding: [0xdb,0x50,0xf5,0x7f]
cjalr ra, 2047(a0)

# capenter's funct7 is 0b0001101, as the RTL and QEMU decoders have it; the table
# said 0b0100010 (byte 0x44) until C-36 was fixed, and that encoding is now pinned
# INVALID in the disassembler suite.
# capenter rs1, rs2 with rd encoded 0 (both implementations fix the destination).
# CHECK: capenter a0, a1  # encoding: [0x5b,0x10,0xb5,0x1a]
capenter a0, a1
# return rd, rs1, rs2: rd is the sealed-return capability, READ from the rd field.
# CHECK: return a0, a1, a2  # encoding: [0x5b,0x95,0xc5,0x42]
return a0, a1, a2

# CHECK: ccsrrw a0, ssp, a1  # encoding: [0x5b,0xf5,0x15,0x01]
ccsrrw a0, ssp, a1
# CHECK: ccsrrw a0, ssp, a1  # encoding: [0x5b,0xf5,0x15,0x01]
ccsrrw a0, 0x011, a1
# CHECK: ccsrrw a0, 0, a1  # encoding: [0x5b,0xf5,0x05,0x00]
ccsrrw a0, 0, a1
# CHECK: ccsrrw a0, 4095, a1  # encoding: [0x5b,0xf5,0xf5,0xff]
ccsrrw a0, 0xfff, a1

# The symbol form of call is PseudoCALL (auipc/jalr); the register form is
# CAP_CALL and does not assemble today -- see cap-call-mnemonic.s.
# CHECK: call foo  # encoding: [0x97'A',A,A,A,0xe7'A',0x80'A',A,A]
call foo
