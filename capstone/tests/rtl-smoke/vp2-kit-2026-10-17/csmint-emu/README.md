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

## Second run: 5a43dd463da4 on the Q-15 tree, both decoders (the RTL lane, 2026-10-17)

**The readings are identical on the pre-fix decoder (via the mtval-0 fallback) and on the fixed one (mtval carries
the bits), 36 values:**

    2 0 | 3 0 0 | 4 0x80009004 | 6 0x8000a018 | 29 0 | 29 0 | 29 0 | 15 0x8000b000 | 2 0 | 3 0 0 | 2 0 |
    7 0x18 7 0x18 7 0x18 7 0x18 7 | 0 0 0 | pass

The RTL lane's build is one page later: desc 0x80009000, slotA 0x8000a000, slotB 0x8000a010, slotnod 0x8000b000,
dbuf 0x8000c000.
- **Rows 2-7 read exactly as pre-registered:** 4 at desc+4, 6 at slotB+8, 29 three times, 15 at slotnod. So these
  work on the RTL:
  - the three plain `ld` under MPRV from M in capability mode (the first of the three unknowns, now read);
  - both alignment checks;
  - the representability rule;
  - the software walk's A/D refusal.
- **Rows 1, 8 and 9 read status 2, and the table stayed at 3 / 0 / 0:** CSNODE itself raised an illegal instruction.
  - The cause is on the RTL side: a `csupctl` write did not flush the pipeline, so the `csnode` right after
    `csrsi csupctl, 8` decoded under the stale flag. va-runv6 had three instructions between the two.
  - The RTL lane's fix flushes on `csupctl` and `srevroot` writes, as `satp` and `fflags` writes already do.
  - Nothing changes in the emulation.
- **The final read-back's `7 0x18` pattern is downstream of that.** No mint reached the STC, so slotA held zeros. The
  LDC read back an untagged word: LCC x1 gave 7, and selectors 2-5 on an untagged operand raised 2. The arm should
  report this better: an untagged read-back currently produces a cascade of 2s rather than one clear reading.
- **Still unread:** LDC through a NONLIN capability, and CAPBOUND's cursor. They come from the re-run on the flushing
  RTL.

## Third run: validated end to end on the RTL (capstone-ariane virtual-profile-2-d bf3db3175, 2026-10-17)

**The setup.** bf3db3175 flushes the pipeline on `csupctl` and `srevroot` writes, and carries this arm, at
5a43dd463da4, as `verif/tests/custom/capstone/va-csmint-emu.S` with a testlist entry. It runs in 62,309 cycles and
passes. The board lane read the 32 CAPPRINT operands itself, from that run's retirement trace (the RTL lane's
`vp7-wt/.sup/flush/va-csmint-emu.*.log`):

    0 4 | 5 0x3000000100000001 0x3000000000000000 | 4 0x80009004 | 6 0x8000a018 | 29 0 | 29 0 | 29 0 |
    15 0x8000b000 | 0 6 | 7 0x3000000180000001 0x3000000000000000 | 30 0 |
    0 0x8000c200 0x8000c300 0x8000c200 4 | 0 0 0

**31 of the 32 readings are as pre-registered**, with the build's addresses one page later (desc 0x80009000, slotA
0x8000a000, slotB 0x8000a010, slotnod 0x8000b000, dbuf 0x8000c000):
- row 1 mints child 3 and ancestor 4, and the records are as QEMU's `cap_rev_create` then `cap_rev_mrev`;
- rows 2-7 refuse in QEMU's order, with QEMU's causes;
- row 8's re-mint overwrites slotA, as QEMU does;
- row 9's 30 is CSNODE's own fault inside the window, caught by the handler, and nothing is written;
- the read-back gives the LINEAR child [dbuf+512, dbuf+768), with CAPBOUND's cursor at the base and perms RO.

**The 32nd, the read-back's type, reads 0 where I pre-registered 1. The prediction was wrong, not the reading.**
- I took 1 from `asm_insn.h`'s CAP_TYPE_LIN, which is CAPTYPE's INPUT encoding, where NOT_CAP is 0.
- LCC returns the spec's numbering, which has no NOT_CAP. QEMU's `CAP_TYPE_LIN` is 0 (`target/riscv/cap.h:29`), and
  the RTL computes `cap_type - 3'd1` (`capstone_dyn_unit.anvil:403`, "see spec").
- So QEMU and the RTL agree on 0 for LINEAR.

**What this settles.** The monitor's CSMINT emulation is validated on the RTL, in every step and every refusal row:
- the MPRV descriptor reads;
- the software Sv39 walk and its A/D refusal;
- the representability rule;
- the mint with CAPCREATE/CAPBOUND;
- the csupctl.table window, with CSNODE and its own 30;
- the physical STC;
- the decode, from mtval or from the instruction at mepc.

All three unknowns are read: MPRV `ld` works, LDC through a NONLIN capability returns the tagged LINEAR, and
CAPBOUND leaves the cursor at the base. The RTL needed two changes along the way: an unknown custom encoding is now
illegal at decode, with mtval carrying the bits, and the pipeline now flushes on csupctl and srevroot writes.

**Still to do:**
- port the routine into the board monitor (capstone-sbi), where its caller injects the status into S instead of
  printing it;
- handle a page fault on the MPRV descriptor read, which this arm does not create.

## va-csmint-mon.S: the monitor's path, pre-registered (2026-10-17, before any run)

**Why a second arm.** The runtime lane's conditions for the port (`docs/plans/2026-10-17-vp2-monitor-csmint-port.md`)
require an arm that exercises the real path:
1. S issues CSMINT, which traps with cause 2.
2. An M trap entry builds the MONITOR's frame: S's x[n] at frame + 8n, and S's sp parked in CSCRATCH, as capstone-sbi's
   `_cap_trap_entry` does.
3. It calls `csmint_dispatch`.
4. Either rd is written into the frame and the frame restored, or the refusal is injected into S. S's own stvec
   handler records scause and stval and steps over the instruction.
5. S reads rd itself, and M reads the slot back physically.

The arm also has to link the SAME routine the monitor ships, so `va-csmint-mon.S` `#include`s `capstone_csmint.S`
from the capstone-sbi tree. It does not carry a copy.

**The pinned source:**
- caplifive-sbi (capstone-sbi), branch `vp2/csmint-emu`, commit **868dd5f**;
- `capstone_csmint.S` sha256 **77c5f13019379cdfff2f290647331fdd279b7e69115a51386c77fecd4bf7650f**;
- build with `-I<capstone-sbi at 868dd5f>`, and check the file's sha256 before building. A different hash is a
  different routine.

**What it adds, beyond the routine:**
- a descriptor in a page without A (13): the monitor walks the descriptor in software, and never uses MPRV;
- an sp operand (2, tval the instruction): the monitor keeps S's sp in CSCRATCH;
- CSNODE's defence. A free head of 0 with a free count of 1 passes the routine's pre-check (64 - 63 + 1 = 2), and
  CSNODE refuses the header as corrupt (29) inside the window. The window's trap vector catches it, the cause is
  injected into S, and nothing is written.

**The window's vector.** On the RTL the trap vector is mtvec in both modes (the RTL lane, bf3db3175). The routine
copies the vector capability with `CCSRRW(s8, ctvec, x0)`, points mtvec at `csm_window_trap` for the window, and
restores cursor, metadata and tag with one CCSRRW.

**Readings (CAPPRINT, in order: S's a2, a5, a6 per row).** Addresses are this folder's build with `link.ld`. The RTL
lane's build reads them from `llvm-nm`.

| row | what | a2 (rd), a5 (scause), a6 (stval) |
|---|---|---|
| 1 | D1 into slotA | 4, -1, -1 |
| 2 | descriptor misaligned | -1, 4, desc+4 [0x80008004] |
| 3 | slot misaligned | -1, 6, slotB+8 [0x80009018] |
| 4 | base >= end | -1, 29, 0 |
| 5 | unrepresentable | -1, 29, 0 |
| 6 | perms > 15 | -1, 29, 0 |
| 7 | slot PTE without D | -1, 15, slotnod [0x8000a000] |
| 8 | descriptor page without A | -1, 13, descnoa [0x8000c000] |
| 9 | rs1 = sp | -1, 2, 0xa0b1165b |
| 10 | D4 into slotA | 6, -1, -1 |
| 11 | next-unused 63 (the pre-check) | -1, 30, 0 |
| 12 | corrupt free head (the defence) | -1, 29, 0 |
| final | slotA: type, base, end, cursor, perms | 0, dbuf+512 [0x8000b200], dbuf+768 [0x8000b300], dbuf+512, 4 |
| final | slotB; slotnod; next-unused; the free word | 0, 0; 0; 63; 0x100000000 |

That is 47 values, then pass. Any 0xbad1 or 0xbad2 is a failure of the arm.

**Not covered by either arm** (the plan records it):
- the monitor's own `_cap_trap_entry` around the hook;
- the mtval-0 fetch on the fixed decoder.
vp2 exercises the first. The fetch was exercised by va-csmint-emu on the pre-fix decoder; the monitor's version of it
is a software-walked fetch, which this arm does not reach.

### va-csmint-mon, first run (the RTL lane, bf3db3175, capstone_csmint.S at 868dd5f with its sha256 checked)

**The build's addresses:** desc 0x8000a000, slotA 0x8000b000, slotB 0x8000b010, slotnod 0x8000c000, dbuf 0x8000d000,
descnoa 0x8000e000, frame 0x8000f000.

**Rows 1-10 read exactly as pre-registered, 30 values:**

    4 -1 -1 | -1 4 desc+4 | -1 6 slotB+8 | -1 29 0 | -1 29 0 | -1 29 0 | -1 15 slotnod | -1 13 descnoa |
    -1 2 0xa0b1165b | 6 -1 -1

These all work on the RTL:
- the frame-building entry;
- `call` and `ret` into the included routine in capability mode;
- `CCSRRW(s8, ctvec, x0)` on a pair only `csrw mtvec` had set;
- the window on the stub;
- the injection into S's own stvec handler;
- the MPRV-free software walk of the descriptor's page (13);
- the sp-operand decode (2, with the bits).

That is two mints through the whole monitor path.

**Row 11 never produced a reading, and the failure is the arm's, not the RTL's or the routine's.** The arm kept its own
table capability (s8) and RAM capability (s3) in registers across exits, but the return path restores S's registers.
So m_next63's store went through S's s8, a scalar, and raised 24. The arm's unexpected-trap path then cascaded until
the cycle limit, and rows 12 and the finals were not reached.

**Fixed:** M re-mints both capabilities at the top of its ecall handling. The readings are unchanged. The re-run is
requested.

### va-csmint-mon, second run: the monitor's whole CSMINT path validated (the RTL lane, bf3db3175, arm a1f7b5163e7d)

The routine is capstone-sbi 868dd5f, with its sha256 checked. The run passes after 144,709 cycles. The board lane read
the CAPPRINT operands from the run's own retirement trace (`vp8-wt/.sup/mon2/va-csmint-mon.*.log`):

    4 -1 -1 | -1 4 0x8000a004 | -1 6 0x8000b018 | -1 29 0 | -1 29 0 | -1 29 0 | -1 15 0x8000c000 |
    -1 13 0x8000e000 | -1 2 0xa0b1165b | 6 -1 -1 | -1 30 0 | -1 29 0 |
    0 0x8000d200 0x8000d300 0x8000d200 4 | 0 0 | 0 | 0x3f | 0x100000000

**Every value is as pre-registered.** There are 46 of them, not 47: 12 rows of 3 plus 10 finals is 46. The "47" in
the pre-registration and in the arm's header was a miscount of the table, which itself lists 46.

What this validates on the RTL, beyond the routine arm:
- the monitor-shaped frame entry, `call` into the included routine, the rd write-back and the frame restore;
- the injection of every refusal into S's own stvec handler: 4, 6, 29, 15, 13, 2 and 30;
- the routine's free-identity pre-check (row 11's 30, before the window);
- **the defence (row 12).** A corrupt free head passes the pre-check, CSNODE refuses it with 29 inside the window,
  the window's trap vector lands it on the stub, and it reaches S as a 29. The table is unwritten: next-unused is
  still 63 and the free word still 0x100000000.
- two mints through the whole path, with the second LINEAR child in slotA and its cursor at the base.

The RTL lane carries the arm, and a pinned copy of the routine (its sha in the testlist comment), as a regression for
the decode, the flush and the window.

**Not validated here:**
- the monitor's own `_cap_trap_entry` around the hook (capstone-sbi cb91d7d, behind `CAPSTONE_VP2_CSMINT`). It
  assembles, and with the flag off it is byte-identical to 234ee66; vp2 exercises it;
- the mtval-0 fetch path on a decoder that leaves mtval 0.

## va-csmint-entry.S: the path through the monitor's OWN trap entry, pre-registered (2026-10-17, before any run)

**Why a third arm.** The runtime lane's next step: "only vp2 exercises it" should not stay true of
`_cap_trap_entry`. This arm `#include`s capstone-sbi's `sbi_capstone.S` itself, with `CAPSTONE_TARGET_FPGA` and
`CAPSTONE_VP2_CSMINT` defined, and that file includes `capstone_csmint.S`. mtvec points at its `_cap_trap_entry`, so
the shipped entry runs:
- the CSCRATCH swap and the GPR save;
- `ldc(gp)`;
- `movc(a0, sp)` and the call to `save_smode_context`;
- the ecall dispatch and its CEPC + 4;
- `_handle_non_ecall`'s CSMINT hook;
- `_handle_branch_1` and `return_to_sumode`.
The disassembly shows the hook as written: `li t1, 2; bne; movc; cincoffsetimm 16; jal csmint_dispatch; bnez ->
_handle_branch_1`.

**Pinned: caplifive-sbi `vp2/csmint-emu` cb91d7d.** Check each file's sha256 before building:

| file | sha256 |
|---|---|
| `sbi_capstone.S` | 1cdc41d16551a168617fa1702d37bf8c52c847c744ca4160efac04246123ba5a |
| `sbi_capstone.h` | 2c6f6bcab4ce180ee642382228ad52fb171bdb3840370bfad3f794cdba5dded1 |
| `capstone_target.h` | 66b051f5f23fb024eef7086286a065cb3e6f24f340705b90f854129a86c31259 |
| `capstone_csmint.S` | 77c5f13019379cdfff2f290647331fdd279b7e69115a51386c77fecd4bf7650f |

Build with `-I<capstone-sbi at cb91d7d>`. The include comes last in the arm's text, because `sbi_capstone.h` redefines
LDC, STC (with the operand order swapped), MOVC and CCSRRW for the monitor's own code.

**What the arm supplies in place of monitor state, line by line. This is what only the vp2 boot proves.**
- The four C functions the entry calls are asm stubs:
  - `save_smode_context` returns;
  - `handle_trap_ecall` is the arm's row reporter plus the three table edits;
  - `handle_exception` prints 0xbad2 and the cause, then spins, so no CSMINT may fall through to it;
  - `handle_interrupt` prints 0xbad3, then spins.
  The monitor's real C versions run only on vp2. That includes `handle_exception`'s `time` emulation after the hook's
  fall-through, and whatever `save_smode_context` records.
- **The stack.** CSCRATCH holds a NONLIN capability over a 4 KiB stack, cursor at the top, with a capability in the
  `ldc(gp)` slot at frame - 16.
  - The monitor's stack capability comes from `SPLIT` at init (`sbi_capstone_init.S`) and `dom_init`. Its type there
    is not reproduced here.
  - The entry's own `movc(a0, sp)` followed by further use of sp, which the hook repeats, already runs on silicon at
    every boot of the current monitor.
- **The monitor's gp** is that slot's capability here. The monitor's real gp, and the C code that uses it, run only
  on vp2.

**Readings:** exactly va-csmint-mon's 46 values in the same order, then pass. They are printed by the
`handle_trap_ecall` stub from S's live a2, a5 and a6. On every ecall the entry overwrites S's a0 and a1 with the SBI
error/value pair, and S reloads them. Any 0xbad2 or 0xbad3 is a failure.
