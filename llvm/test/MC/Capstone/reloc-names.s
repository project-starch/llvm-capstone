# Relocations in Capstone objects must be NAMED by the binary-format layer.
#
# Measured 2026-09-04: for `call foo`, both `llvm-readobj -r` and
# `llvm-objdump -d -r` print the relocation as "Unknown foo".  The fixup is
# fixup_capstone_call_plt (cap-call-symbol.s pins that), so the object carries
# a relocation the name table cannot label.  Cause located: ELFRelocs/
# Capstone.def defines R_Capstone_CALL_PLT = 19 and ELF.h includes it for the
# enum, but lib/Object/ELF.cpp's getELFRelocationTypeName switch has no
# `case ELF::EM_CAPSTONE` (RISCV's is at :114-120) and neither does
# getELFRelativeRelocationType -- so EVERY Capstone relocation prints as
# Unknown, not only this one.  Scope: human inspection of objects and the
# planned obj-relocs tests; nothing in the board path reads relocation names
# (the board-run artifact check is `llvm-objdump -d`), so no board verdict is
# affected.  XFAIL until fixed; then lit reports XPASS and the marker comes off.
#
# MUTATION: change `call foo` to `lui a0, 1` (no relocation) -> the CHECK for a
# named relocation fails on an EMPTY relocation list, which shows the check is
# bound to a relocation being present and named, not to any text.
#
# RUN: llvm-mc -triple capstone64 -filetype=obj %s | llvm-readobj -r - | FileCheck %s
# XFAIL: *

# CHECK: Relocations [
# CHECK: .rela.text
# CHECK: R_Capstone_CALL_PLT foo
# CHECK-NOT: Unknown
call foo
