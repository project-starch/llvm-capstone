# CONTROL for cap-call-mnemonic.s: the symbol form of `call` is PseudoCALL and
# must keep expanding to the auipc/jalr pair with a call_plt fixup.  Any fix
# for the CAP_CALL mnemonic collision (a distinct register-form mnemonic, or
# parser precedence) has to leave this file green -- that is what makes the
# XFAIL in the sibling file a bound on the fix and not just a record of the bug.
#
# Measured 2026-09-04: encoding [0x97'A',A,A,A,0xe7'A',0x80'A',A,A],
# fixup_capstone_call_plt at offset 0, value foo.
#
# MUTATION: change `call foo` to `call a0, a1` -> this file fails to assemble
# today (the collision), so the two files cannot both be green until the
# mnemonic question is settled one way or the other.
#
# The relocation itself is NOT checked here: both llvm-objdump -r and
# llvm-readobj -r print it as "Unknown foo" on the current tree -- the ELF
# relocation-type name table for Capstone objects is incomplete (measured
# 2026-09-04; see reloc-names.s, which pins the name that should appear and is
# XFAIL until it does).  A control must be green, so this file asserts only the
# expansion, which is correct today.
#
# RUN: llvm-mc -triple capstone64 -show-encoding %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -triple capstone64 -filetype=obj %s | llvm-objdump -d -r - | FileCheck %s --check-prefix=OBJ

# ASM: call foo
# ASM-SAME: encoding: [0x97'A',A,A,A,0xe7'A',0x80'A',A,A]
# ASM-NEXT: fixup A - offset: 0, value: foo, kind: fixup_capstone_call_plt
# OBJ: auipc ra, 0x0
# OBJ: jalr ra
call foo
