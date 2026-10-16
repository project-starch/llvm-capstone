# The monitor's CSMINT emulation for vp2, as an RTL simulation arm: pre-registration (2026-10-17, before any run)

On vp2, CSMINT traps to M as cause 2, with mtval holding the instruction bits, and the monitor emulates it. The
design was settled with the RTL and runtime lanes (`../README.md`, "Monitor item for vp2"). `va-csmint-emu.S` is that
emulation, written as the M-mode handler of a bare arm in the RTL tree's conventions (va-runv6 at capstone-ariane
virtual-profile-2-d 41fa120bf).
- It is placed between `csmint_emulate:` and `csmint_done` in the arm. The monitor port carries it.
- The RTL lane runs it in its simulation on the D4 tree. The board lane does not run anything on the RTL lane's
  worktrees.
- QEMU's `helper_csmint` (capstone-qemu `target/riscv/capstone_table.c:218-283` at d35af47b163c) is the reference for
  every step and every refusal. The emulation checks in QEMU's order, and refuses before any write.

## The emulation, as written

| step | what it does | refusal |
|---|---|---|
| 0 | decode: from mtval; if mtval is 0, from the instruction at mepc, read under MPRV and MXR | — |
| 1 | check the descriptor's alignment | `desc & 7` → 4, tval desc |
| 2 | read `{base, end, perms}` with three plain `ld` under `mstatus.MPRV`, MPP = S from the trap | — |
| 3 | check the slot's alignment | `slot & 15` → 6, tval slot |
| 4 | walk the slot VA in software (Sv39, tables read physically through the RAM capability). It needs V, then R, W, A and D on the leaf, U only with `mstatus.SUM`, and an aligned superpage. It never sets A or D (the RTL lane's rule). | 15, tval slot |
| 5 | check the descriptor | base >= end or perms > 15 → 29; not representable → 29 |
| 6 | mint the LINEAR capability with CAPCREATE, CAPTYPE, CAPBOUND and CAPPERM | — |
| 7 | open the table window: `csrsi csupctl 8; csnode; csrci csupctl 8` and nothing else | 30 or 29 from CSNODE itself |
| 8 | STC the capability to the slot's physical address, through the RAM capability | — |
| 9 | rd = the ancestor id; mepc += 4 | — |

- **Representable** means a length below 4 KiB, or a granule-aligned base and a granule-multiple length, with granule
  2^(msb(len)-9). This is `capstone_repr_ok`, validated against the encoder oracle by `repr-cross-check.py`.
- **CSNODE's own refusals** arrive as M faults inside the window. The handler then clears the bit, restores the saved
  mepc and mstatus, and reports the cause.
- **The routine returns** a status (0 for done, otherwise the cause S must receive) and a value (the ancestor id, or
  the tval). Here the arm prints both and skips the instruction. The monitor injects the cause to S instead.

## Pre-registered readings (CAPPRINT, in order)

The table is RECYCLE, capacity 64, next-unused 3, record 2 valid, as va-runv6. Addresses are symbolic. The numbers
in brackets are this folder's own build (`link.ld` below: `.text.init` at 0x80000000, `.tohost`, `.text` and `.data`
each page-aligned). The RTL lane's build must recompute them from `llvm-nm`, because a different link script moves
them. That is what vb-trans's three positions did on vp1.

| row | what | readings |
|---|---|---|
| 1 | D1 {dbuf, dbuf+256, RW} into slotA | 0, 4 |
| ecall 1 | next-unused; rec3 w1; rec4 w1 | 5, 0x3000000100000001, 0x3000000000000000 |
| 2 | descriptor misaligned (desc+4) | 4, desc+4 [0x80008004] |
| 3 | slot misaligned (slotB+8) | 6, slotB+8 [0x80009018] |
| 4 | base >= end | 29, 0 |
| 5 | unrepresentable: [dbuf+8, dbuf+8+8192), granule 16 | 29, 0 |
| 6 | perms 16 | 29, 0 |
| 7 | slot in a page whose PTE has no D | 15, slotnod [0x8000a000]; nothing written |
| 8 | second CSMINT on slotA: D4 {dbuf+512, dbuf+768, RO} | 0, 6 (child 5, ancestor 6). QEMU overwrites. |
| ecall 8 | next-unused; rec5 w1; rec6 w1 | 7, 0x3000000180000001, 0x3000000000000000 |
| 9 | next-unused set to 63 (one free identity), D1 into slotB | 30, 0 (CSNODE's own fault) |
| final | slotA read back with LDC: type, base, end, cursor, perms | 1, dbuf+512 [0x8000b200], dbuf+768 [0x8000b300], dbuf+512, 4 |
| final | slotB, slotB+8, slotnod | 0, 0, 0; then pass |

That is 32 values in all. A missing value, or any 0xbad1, 0xbad2 or 0xbad3 (a non-CSMINT illegal instruction, an
operand-convention break, an unknown ecall), is a failure of the arm, not a reading.

**Row 7 and QEMU.** QEMU's own CSMINT would succeed there: its `svadu` is on by default (`target/riscv/cpu.c:1919`),
and its walker then sets D (`cpu_helper.c:1123-1130`). The RTL walker faults on a clear A or D, and the emulation
follows the RTL. On silicon, Linux sets D in its page-fault handler and the retried CSMINT succeeds, so the end state
agrees with QEMU but the fault count does not. The QEMU prediction for vp2 would match the board's fault counts only
with `svadu=false`.

**Not covered here:** a page fault on the descriptor read under MPRV; that trap would arrive in M during the
emulation. The monitor port has to handle it by reporting the load page fault (13) to S, and a later arm should test
it.

## Building it (syntax and layout only; the RTL lane builds its own)

    clang --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
      -fuse-ld=lld -I<capstone-ariane>/verif/tests/custom/capstone -I<riscv-tests env/p and macros/scalar> \
      -T link.ld va-csmint-emu.S -o va-csmint-emu.elf

It assembles with the in-tree clang, and the disassembly is checked:
- the 9 CSMINT words are `0xa0b5165b` (funct7 0x50, rd a2, rs1 a0, rs2 a1);
- the one CSNODE is `0xa80a985b`, with rd a6 and rs1 s5;
- the window is exactly `csrsi 0x7c4,8; csnode; csrci 0x7c4,8`.

## First run, and one change before the second (2026-10-17)

**The first run (the RTL lane, D4 + Q-15)** read 0xbad1 on every row. That is the arm's guard against an illegal
instruction that is not CSMINT, and it fired correctly:
- CSMINT's funct7 0x50 fell into the decoder's empty default arm, so it issued as an unknown op and raised the illegal
  instruction at execute, with **mtval 0**;
- the ruling's "mtval = the bits" is the decoder's illegal path, which the RTL lane is fixing;
- row 1's table read 3 / 0 / 0, consistent with nothing having been minted.

Silicon built today would read mtval 0 the same way, so the emulation now decodes from the instruction at mepc when
mtval is 0. The fallback is exercised only on a decoder that leaves mtval 0. The pre-registered readings are
unchanged.
