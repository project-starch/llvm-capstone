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

### va-csmint-entry, run: the shipped `_cap_trap_entry` around the hook, validated (the RTL lane, bf3db3175)

The run was on capstone-ariane virtual-profile-2-d cab4ee891. That commit adds only tests to bf3db3175's RTL. Inputs:
- the arm at 941196aedc65;
- the include directory, a `git archive` export of caplifive-sbi cb91d7d with all four pinned sha256 values checked
  before the build;
- the ELF, sha256 b47aba619b6b3899092d496523aee06018ff53ca8e20e6104a14eeb5a3c46fa1.

The run passes after 149,861 cycles. The board lane read the readings from the run's own retirement trace
(`capprint-retired.py`: 46 retired prints, 46 raw, no speculative leftovers):

    4 -1 -1 | -1 4 0x80009004 | -1 6 0x8000a018 | -1 29 0 | -1 29 0 | -1 29 0 | -1 15 0x8000b000 |
    -1 13 0x8000d000 | -1 2 0xa0b1165b | 6 -1 -1 | -1 30 0 | -1 29 0 |
    0 0x8000c200 0x8000c300 0x8000c200 4 | 0 0 | 0 | 63 | 0x100000000

**Against the pre-registration:**
- **39 values are identical** to va-csmint-mon's.
- **The seven address readings are not, literally.** Each sits exactly 0x1000 below va-csmint-mon's, at the same
  offset from the same symbol (diffed by script over both runs' retired prints):

  | reading | symbol + offset |
  |---|---|
  | row 2's stval | `desc` + 4 |
  | row 3's | `slotA` + 0x18 |
  | row 7's | `slotnod` |
  | row 8's | `descnoa` |
  | the three final addresses | `dbuf` + 0x200, + 0x300 and + 0x200 |

- **Why they moved:** this build's data symbols lie 0x1000 lower (`desc` 0x80009000 against 0x8000a000 in that arm's
  build, both from the ELFs the runs used).
- **The miss is the pre-registration's.** It predicted absolute addresses that depend on the link layout, and should
  have stated them symbol-relative. Symbol-relative, all 46 match.

**The path, counted from the retirement trace:**
- **Entries:** `_cap_trap_entry` retires 27 times: 12 CSMINTs and 15 ecalls.
- **The hook:** `_handle_non_ecall` 12, `csmint_dispatch` 12 and `_handle_branch_1` 12. So every CSMINT took the hook,
  and none fell through to `handle_exception`, which retires 0 times. `handle_interrupt` also retires 0 times.
- **Refusals and the window:**
  - `csm_inject` retires 10 times, for the ten refusal rows. Rows 1 and 10 mint.
  - `csm_window_trap` retires once, for row 12's CSNODE defence. Its 29 is the run's one logged exception
    (ILLEGAL_OPERAND_VALUE).
- **The rest of the entry:** `save_smode_context` 27 and `handle_trap_ecall` 15. `return_to_sumode` retires 26 times:
  every entry but the last ecall, which ends the run.

**Validated:** the monitor's own `_cap_trap_entry` at cb91d7d with `CAPSTONE_VP2_CSMINT`, run end to end on the RTL.
That covers:
- the CSCRATCH swap;
- the GPR save and restore;
- `ldc(gp)`;
- the hook;
- the rd write-back through `_handle_branch_1`;
- `return_to_sumode`.

**Still vp2-only:** what the stubs stand in for, as listed in the pre-registration:
- the C bodies of `save_smode_context`, `handle_trap_ecall`, `handle_exception` and `handle_interrupt`;
- the monitor's stack capability from `SPLIT` and `dom_init`, in place of the arm's NONLIN one;
- the monitor's gp.

## va-csmint-ft.S: the hook's FALL-THROUGH, pre-registered (2026-10-17, before any run)

**The gap.** The hook's other branch has never run, in simulation or on silicon:
- `csmint_dispatch` returns 0 ("not ours");
- the entry re-reads mcause and calls `handle_exception`;
- the entry writes the returned value to x[rd], with rd from mtval[11:7], and advances mepc.

In va-csmint-entry every illegal instruction was a CSMINT, and the `handle_exception` stub retired 0 times. On the
board this branch carries every Linux `rdtime`, which the FPGA monitor emulates in `handle_exception`. A defect there
would hang vp2's first boot, directly after a flash.

**The arm.** Same setup, pins and includes as va-csmint-entry: caplifive-sbi cb91d7d with the four sha256s above. The
stubs differ:
- `handle_exception` prints its cause argument and mtval, then returns `0x7e50 | mtval[23:20]` in place of the time
  emulation;
- `handle_trap_ecall` with a7 = 30 prints S's s6, s7, s9 and a3. Those are sentinels 0x5106, 0x5109 and 0x5a03, plus
  the target register s7 (x23).

S's four rows:
- **A:** `csrr s7, time`, 0xc0102bf3. The RTL has no time CSR (no `CSR_TIME` arm in `csr_regfile.sv`), so the read
  is illegal. The commit stage puts the instruction bits in tval (`commit_stage.sv:970`).
- **B:** `csrr s7, mhartid`, 0xf1402bf3: an M-only CSR read from S, a privilege violation.
- **C:** 0xb0001bdb, a near-miss. It has CSMINT's opcode and funct3 with funct7 0x58, one bit from 0x50.
  - 0x58 is unassigned, so it takes the decoder's `default: illegal_instr` arm.
  - The custom-2 funct3-1 case assigns 0x00-0x0d, 0x20-0x22, 0x24 and 0x51-0x54 (`decoder.sv:1209-1391`). 0x51 is
    CDERIVE, which the first draft used before this check.
  - Masked, it reads 0xb000105b against CSMINT's 0xa000105b, so the decode compare must reject it.
- **D:** va-csmint-entry's row 1, the real CSMINT, run after the three fall-throughs.

**Predicted readings, 26 values then pass.** Addresses are stated symbol-relative this time:

| row | the `handle_exception` stub | S's registers |
|---|---|---|
| A | `2 0xc0102bf3` | `0x5106 0x7e51 0x5109 0x5a03` |
| B | `2 0xf1402bf3` | `0x5106 0x7e54 0x5109 0x5a03` |
| C | `2 0xb0001bdb` | `0x5106 0x7e50 0x5109 0x5a03` |

- Row D prints `4 -1 -1`, as va-csmint-entry's row 1: the same table state, since nothing before it writes the table.
- The finals read slotA back: `0`, `dbuf`, `dbuf + 0x100`, `dbuf`, `6`. That is LINEAR in lcc's numbering, the base,
  the end, the cursor and RW; va-csmint-entry's RO read 4.

**Predicted path counts, from the retirement trace:**
- `_cap_trap_entry` 9: four illegal instructions and five ecalls;
- `csmint_dispatch` 4;
- `handle_exception` 3;
- `handle_trap_ecall` 5;
- `csm_inject` 0;
- `csm_window_trap` 0.

**What a failure would say:**
- a wrong s7 with the stub's prints present means the write-back;
- a changed sentinel means the frame;
- a stub print missing for A, B or C means the hook claimed a non-CSMINT, which is the decode compare;
- a hang after the stub returns means the return path.

### va-csmint-ft, run: the fall-through validated, every reading and every path count as pre-registered (the board lane, bf3db3175)

The board lane ran it on a private copy of the RTL lane's run tree (`sup-runs/csm-ft`, virtual-profile-2-d cab4ee891,
RTL bf3db3175), with the RTL lane's runner (`run-sup-ladder.sh`, S12_MEM_DELAY=12, SUP_TRACE).
- The includes are the same `git archive` export of cb91d7d, with the four sha256s re-checked.
- The arm is at d2458229ee1e. The ELF's sha256 is b5aa021e744d113e8761ee73559cf736d6f1e1cfaec6ed2f3f3a6a310e8023d3.
- The run passes after 89,653 cycles.
- `capprint-retired.py` reads 26 retired prints and 26 raw.
- A script compares them with the pre-registration: 0 mismatches, `dbuf` = 0x8000c000 from the ELF.

    2 0xc0102bf3 | 0x5106 0x7e51 0x5109 0x5a03 |
    2 0xf1402bf3 | 0x5106 0x7e54 0x5109 0x5a03 |
    2 0xb0001bdb | 0x5106 0x7e50 0x5109 0x5a03 |
    4 -1 -1 | 0 0x8000c000 0x8000c100 0x8000c000 6

**Path counts, all as predicted:**
- `_cap_trap_entry` 9;
- `csmint_dispatch` 4;
- `handle_exception` 3;
- `handle_trap_ecall` 5;
- `csm_inject` 0;
- `csm_window_trap` 0.

**Validated:** the branch every Linux `rdtime` takes on a flag-on monitor:
- the hook calls `csmint_dispatch`, which says not ours for a `time` read, an M-only CSR read, and a one-bit near-miss
  of CSMINT's encoding;
- the entry passes cause 2 to `handle_exception`;
- the returned value lands in x[rd], from mtval[11:7];
- mepc advances, and S resumes with its other registers intact.

A real CSMINT after three fall-throughs still mints.

**Still vp2-only:** `handle_exception`'s C body, the real time emulation, on the monitor's own stack and gp.

## Boot BL on silicon, pre-registered (2026-10-17, before the boot): the flag-on monitor under Linux, one variable from boot P

**What it settles.** This is the last vp2 firmware gate the runtime lane asked for. Inside boot P's proven
configuration it runs the monitor that vp2 will carry: capstone-sbi cb91d7d, built with `CAPSTONE_VP2_CSMINT`. So
every Linux `rdtime` takes va-csmint-ft's fall-through on silicon. It runs on caplifive_sublet_d93a8d66b, resident.

**One variable.** The cleanest comparison against boot P (`b0-silicon-delegated-runtime.md`, Boot P, 2026-10-16):
- **Unchanged from boot P:** the wrapper (caplifive-opensbi 1269c77), the private Image ae58cf97765442da (all 19
  manifest files re-checked in its decompressed initramfs), boot P's three supervised-call defines, and its ten
  rungs in its order.
- **The monitor:** cb91d7d. Its tree differs from boot P's ebe0a25 only by `capstone_csmint.S` and the hook:
  ebe0a25's tree is 234ee66's (0b255381), and 234ee66..cb91d7d touches those two files only.
- **Firmware with the flag:** 26bc4b23c520d9d6. The build uses `build-fpga-fw.sh`, whose wrapper passes the defines
  to the assembler (`platform/fpga/ariane/objects.mk:18`).
- **Control, flag off:** the same build from cb91d7d **reproduces boot P's firmware byte for byte,
  1b44a465d96753b2**. So the flag, with what it compiles in, is the only difference.

**No CSMINT can execute in this boot, which the static scan shows.** The scan (`scan-csmint.py`) finds every word
with CSMINT's encoding at 2-byte steps. Positive control: 1 hit in va-csmint-ft's ELF and 12 in va-csmint-entry's.
- **The Image:** 6 hits, all inside the gzip-compressed initramfs span [0x60e158, 0x9ffb9a). That span starts at
  System.map's `__initramfs_start`, past the kernel's text (to 0x3837b4) and init text (to 0x42c8e0).
- **The decompressed initramfs:** 124 files, 2 hits, neither an instruction:
  - libm.so.6's hit lies in `.rodata`;
  - mruby.dom's, at 0xccfbe, is 2 mod 4, and mruby.dom is RVC-free: its e_flags are 0x0, and its disassembly has
    394,463 four-byte instructions, no two-byte one and no address that is not 4-aligned. So that offset cannot be an
    instruction at all. It straddles the `stc` at 0xccfbc and the `sw` at 0xccfc0.
- **This is a property of BL's image, not of the monitor.** Any other image set under a flag-on monitor would need
  the same scan.

**What the hook does in this boot, stated before it runs:**
- **Every illegal-instruction trap from S reaches `csmint_dispatch`, and returns "not ours".** This includes each
  Linux `rdtime`.
- **The fast path.** A CSR exception's mtval is the instruction bits on d93a8d66b too: its `commit_stage.sv:865` copies
  the instruction into tval, and its csr_regfile has no `CSR_TIME` arm. The classic monitor's rd write-back already
  relies on mtval[11:7]. So each such trap costs about 15 more instructions and no page walk.
- **Nothing counts these on silicon.** The evidence is the boot and the rungs, compared with boot P.

**A deliberate CSMINT on this bitstream, written down and NOT run:**
- **How it traps.** d93a8d66b's custom-2 funct7 case ends in `default: ;` (decoder.sv:1343), so CSMINT issues to the
  FLU and traps illegal with tval 0. That is the behaviour the vp2 decoder's 2026-10-17 comment replaced.
- **The decode.** The hook then fetches it through the mtval-0 path: a software walk of mepc.
- **A refusal found before the table check** (alignment, a fault on the descriptor or the slot, a bad descriptor) is
  injected into S as on vp2.
- **A valid one** reaches `csrr 0x5C1` (`capstone_csmint.S:189`). d93a8d66b has no srevroot, so that read traps
  illegal in M, into `_cap_trap_entry`, which is not reentrant: **an expected wedge**. vp1 has no srevroot either (it
  is item A, vp2's).
- **So the flag-on monitor belongs on vp2.** On an earlier bitstream it is safe only while nothing issues CSMINT,
  which is this boot's case.
- **Why not run it:** a reading on a bitstream vp2 code never runs on says nothing about vp2, and it would end the boot.

**Predictions** (boot P's readings, which this boot must repeat):
- **Retvals:** b0-stats 0, b0-hello 0, b5-hello 0, b5-mruby-e-stats 0, b5-mruby-stdin-stats 0, b5-mruby-bench 0,
  r54-bounds 64, cw-pool 0, b6-sublet 0, b0-stats2 0.
- **mruby's hashes:** 53c234e5e8472b6a, 5476dba1f4bd59f1, 8d1539bc79e2b630.
- **No EXCT, EXCS, RGNF or ILLX.** ILLX is the monitor's report for an illegal instruction other than a time read.
- **Bench times:** each benchmark within 0.5% of boot P's 138315, 45514, 2458, 30708 and 81907 ms. The B5 pair
  repeated boot P within 0.20%, and the hook's cost on a time read cannot move a benchmark measurably.
- **Exploratory:** the launch ticks, against boot P's (domain 22.93 s cold, regions 9.12 s).

**What a failure would say:**
- **Linux does not boot, or a rung differs:** the flag-on monitor is at fault, because the flag is the only variable.
  The first suspect is the fall-through under Linux's own traps, which va-csmint-ft covered only in simulation.
- **An ILLX:** an illegal instruction that is neither CSMINT nor a time read. The hook should have passed it through
  unchanged.

### Boot BL, result (2026-10-17 09:37-09:50, caplifive_sublet_d93a8d66b, firmware 26bc4b23c520d9d6): every reading as boot P's

The board script checked the firmware hash and the shared tree's pins before the boot. The watchdog was live (52
lines). `compare-bootp.py` compares the run with boot P's record. Its positive control, a copy of boot P with one retval
and one hash altered, reads DIFFER on both.
- **Retvals:** IDENTICAL on all ten rungs. b0-stats 0, b0-hello 0, b5-hello 0, b5-mruby-e-stats 0,
  b5-mruby-stdin-stats 0, b5-mruby-bench 0, r54-bounds 64, cw-pool 0, b6-sublet 0, b0-stats2 0.
- **mruby's hashes:** 53c234e5e8472b6a, 5476dba1f4bd59f1, 8d1539bc79e2b630, the same three.
- **No EXCT, EXCS, RGNF or ILLX.**
- **Bench times against boot P,** all within the pre-registered 0.5 %:

  | benchmark | boot BL, ms | change |
  |---|---|---|
  | mandel_term | 138,351 | +0.03 % |
  | fib | 45,524 | +0.02 % |
  | so_lists | 2,466 | +0.33 % |
  | so_mandelbrot | 30,683 | -0.08 % |
  | ao_render | 81,931 | +0.03 % |

- **Launch ticks (exploratory):**
  - the cold mruby launch's domain stage is 23.24 s against 22.93 s, and its regions stage 9.09 s against 9.12 s;
  - the other three launches are within 0.03 s per stage.

**What this establishes:** the vp2 monitor build, capstone-sbi cb91d7d with `CAPSTONE_VP2_CSMINT`, boots Linux on
silicon and runs the process ABI's whole B-series, with every reading as on the flag-off build of the same tree. Every
Linux `rdtime` crossed the hook's fall-through, va-csmint-ft's path, and Linux's clocksource advanced normally.

**Still vp2-only:** a CSMINT that the hook handles. Here none could be issued (the static scan above), and on this
bitstream a valid one would wedge (`csrr` srevroot).

## Descriptor v7 on the combined monitor, capstone-sbi vp2/tier1-monitor fba7812 (2026-10-18)

The runtime lane's semantics ruling (Q-23) replaced the three-word descriptor with six words:
- w0 base, w1 end, w2 perms (bits 0-3) with bit 8 = NONLIN, w3 the cursor (base <= cursor <= end);
- w4 a second slot (0 = none; LINEAR with a second slot is refused with 29);
- w5 reserved, which must be 0.

The order of refusals is: the access-shaped refusals (6, 15, 13) before the semantic one (29), and w4 is checked like
the first slot. The routine is capstone-sbi vp2/csmint-v7 (729a340, then 67ed67f). 67ed67f emulates only a CSMINT
from S: the claim-auditor found the routine emulating one from U, and QEMU refuses CSMINT at PRV_U at decode.
fba7812 merges it with the S-trap injection (vp2/os-trap-inject f8bfde5).

Pinned:
- sbi_capstone.S 7b23b6a67540fd45cee4fa2bd66f5cdd7098b9c5208e1b0ce244367a5d520365
- capstone_csmint.S 4600d04680158edf90396cc303cbb5aff4e2a286c9a0f302058ecd41232d6fbf
- sbi_capstone.h 2c6f6bcab4ce180ee642382228ad52fb171bdb3840370bfad3f794cdba5dded1
- sbi_capstone.c 2ff807a6db2bce15127fe349f4fb0ef508c1ebfdcea62cb25b2612bd2184d7aa (not linked by these arms)

- **va-csmint-mon v7** is the RTL lane's, in capstone-ariane virtual-profile-2-c. 2639bd078 re-pre-registered it with
  81 readings: 18 rows of three, then a readback of 27. It read all 81 on 729a340's routine. 7b8c0d979 moves its pin to
  fba7812, where 0 of 81 values differ (SUCCESS after 211,099 cycles).
- **va-csmint-entry v7** is this folder's `va-csmint-entry.S`, sha256 602a0bf1e99d869f…, written by the RTL lane. It
  applies mon's v7 rewrite to the v6 file (7b1155f93d15):
  - six-word descriptors at a 64-byte stride;
  - rows 11-16;
  - the readback of slotC, slotD and slotE, with slotB's page carved into four slots.

  Its prediction is mon's 81 readings at its own layout, the same rule the v6 run used. The prediction is in the
  file's header; the file was not committed before the run. On bf3db3175 it read 81 values, SUCCESS after 215,008
  cycles (`results/rtl-bf3db3175-v7-entry.result-lines.txt`). 64 values equal mon's. The other 17 are address-valued,
  and each sits exactly 0x1000 below mon's, at this arm's layout. The file here is the bytes the run compiled: the RTL
  lane's run script copied it in just before the ladder, and its sha256 and an unchanged mtime match. The executed
  trace agrees too: the row offsets 64..640 and the report sequence 1-16, 21, 17, 22, 18, 99.
- **va-osinject revision 2** (`../osinject/`) adds the U-origin CSMINT row on the same files: 23 of 23 as predicted.


## Boot BLb, pre-registered (2026-10-18, before the boot): capstone-bootstrap's capstone-sbi pointer bumped to dafe4f2, BL's control on the resident vp2

**What it settles.** caplifive-opensbi's capstone-bootstrap points at capstone-sbi 234ee66. Bumping it to dafe4f2
brings in three commits:
- c3293b3: the CSMINT hook, behind CAPSTONE_VP2_CSMINT;
- 83966bc: the S-trap injection, which is FPGA-only and compiled into every FPGA build;
- dafe4f2: the v7 routine, behind the flag.

With the flag off, the change against boot P's monitor (ebe0a25, whose tree is 234ee66's) is the injection and its
domain-call counter. The runtime lane asked for BL's control before any physical-line record cites the bumped
firmware: one boot on that firmware, flag off, with BL's image set, repeating boot P's readings within BL's bands.

**The firmware (arm B):** fw_payload 2a0850a406d1229c, gate 25712 of 32768 B.
- **Wrapper:** 1269c77, boot P's and BL's wrapper. capstone-bootstrap's head 3ab5655 differs from it only by the
  capstone-sbi pointer, so this is the bumped wrapper's tree.
- **Monitor:** capstone-sbi dafe4f2. Its four files were checked in the build's extracted source: sbi_capstone.c
  c8172e5abc09687e…, sbi_capstone.S 7b23b6a67540fd45…, capstone_csmint.S 4600d04680158edf…, sbi_capstone.h
  2c6f6bcab4ce180e….
- **Defines and Image:** boot P's three supervised-call defines, without CAPSTONE_VP2_CSMINT, and BL's Image
  ae58cf97765442da.
- **Symbols:** `_inject_s_trap` is in the ELF and `csmint_dispatch` is not.

**A second variable, and how the design handles it.** P and BL ran on caplifive_sublet_d93a8d66b. The resident
bitstream is vp2 (caplifive_vp2_bf3db3175), and putting d93a8d66b back is a reflash, which is the lead's call. I found
no vp2 record of the B-series rungs:
- vp2's sessions ran the bare arms and the corpus;
- Tier 1/2 kept only b0-stats and b0-hello, dropping the rest for image size.
- **Arm B runs first:** BL's ten rungs in BL's order, P's oracles, on the bumped firmware.
  - If B reads boot P within BL's bands, the control passes. Neither the firmware nor the bitstream moved any reading.
- **Arm A runs next only if B differs:** boot P's own firmware, 1b44a465d96753b2, on vp2. It is the arm that differs
  from B by the firmware alone.
  - B is then judged against A within the same bands, and A against P says what the bitstream changes.
  - A firmware difference shows as B ≠ A; a bitstream one as A ≠ P with B = A.
- **The d93a8d66b reading of the bumped firmware** waits for the next residency of that bitstream.

**Predictions for B** (boot P's readings, BL's bands):
- **Retvals:** b0-stats 0, b0-hello 0, b5-hello 0, b5-mruby-e-stats 0, b5-mruby-stdin-stats 0, b5-mruby-bench 0,
  r54-bounds 64, cw-pool 0, b6-sublet 0, b0-stats2 0.
- **mruby's hashes:** 53c234e5e8472b6a, 5476dba1f4bd59f1, 8d1539bc79e2b630.
- **Bench times within 0.5 %** of boot P's: 138315, 45514, 2458, 30708 and 81907 ms.
- **No EXCT, EXCS, RGNF or ILLX, and no INJS, NODM or INJL.** An INJS would be a trap from S that boot P's monitor
  could not have survived. P did not wedge, so none is expected, and one would be a reading in its own right.
- **Exploratory:** the launch ticks, against P's.

**What a difference would say** (read no further than the first failure; arm A then runs):
- **Linux does not boot, or a B-series rung differs on both arms:** vp2 differs from d93a8d66b for this image set,
  not the firmware.
- **B differs and A does not:** the bump. The first suspect is the counter or the injection on a path P's monitor
  handled differently.

**The scripts** (`bootblb/`), checked before the boot:
- **judge-blb.py** applies BL's bands mechanically. Against boot P's run-scoped transcript, BL's own run reads PASS.
  These each read DIFFER: BL's altered-copy control (a retval and a hash), so_lists moved 1.71 %, and an INJS line.
  An empty transcript reads NO DATA.
  - The comparison must use the run-scoped `board-bootp.txt`, never the raw capture. A first try took
    `board-bootp-raw.txt` by `ls | head -1`, and BL then read DIFFER against its own committed IDENTICAL. The scripts
    pin the path, and P's transcript by sha256/16 5fb970fff3b3ea24.
- **board-bootblb.sh B|A <fw hash>** STOPs on any of these:
  - a wrong firmware hash;
  - a driver other than the lane's 4ae259caae50b35d;
  - a changed P transcript;
  - a shared tree not at df0e9a8 / 93440d5;
  - an existing output directory.
- **A stubbed dry run** read each branch as written: B passing, B differing, no transcript, a wrong firmware, and
  arm A with B the same, B different, or B absent.

### Boot BLb, result (2026-10-18 01:24-01:51): DIFFER, as pre-registered. One reading, so_lists, is 0.57 % faster on the bumped firmware

Result lines: `bootblb/results-bootblb.txt`.
- **Arm B** (bumped firmware 2a0850a406d1229c) against boot P: DIFFER.
  - All ten retvals and all three mruby hashes are identical, and there is no monitor tag of any kind.
  - Four of the five bench times are inside the band: mandel_term +0.00 %, fib +0.32 %, so_mandelbrot −0.07 % and
    ao_render −0.06 %.
  - **so_lists reads 2444 ms against 2458, −0.57 %, outside the 0.5 % band.**
- **Arm A** (boot P's own firmware 1b44a465d96753b2, on vp2) against boot P: PASS.
  - Everything is inside the band, so_lists exactly 2458. The bitstream, vp2 against d93a8d66b, moves none of
    these readings.
- **B against A:** DIFFER, on so_lists alone (−0.57 %); fib reads +0.37 %. The launch ticks match P's within 0.3 s
  per stage.

**The pre-registered reading is "B differs and A does not: the bump",** so the control has not passed and the pointer
bump does not land on it. That rule cannot separate a real effect from the draw-to-draw spread of the shortest
benchmark: 14 ms on 2.4 s, in the faster direction, where BL's own so_lists read +0.33 % against P.

## Boot BLb repeat, pre-registered (2026-10-18, after B and A, before B2 and A2): one more draw per arm, alternating

**Design.** Arm B2 is the bumped firmware 2a0850a406d1229c, then arm A2 is boot P's 1b44a465d96753b2. Everything else
is as in B and A: the vp2 bitstream, BL's Image and ten rungs, the judge and the bands. Run order B, A, B2, A2.

**Readings, written down before the draws:**
- **The 0.57 % was the spread, not the bump:** B2's so_lists is within 0.5 % of A's 2458, i.e. 2446-2470 ms, and B2
  against A2 reads PASS. The control then passes on B2, and the pointer lands, citing all four arms.
- **The bump moved so_lists:** B2 reads 2445 ms or less again, and A2 stays within the band of 2458.
  - That is a real effect of dafe4f2 on the firmware, however small, faster rather than slower. It is reported, and
    it needs a mechanism before the bump lands. The candidate is a monitor code or data layout change on the
    rdtime trap path, which every Linux time read takes.
- **Anything else** (a retval, a hash, a tag, another time outside the band) is read like arm B: no further than its
  first difference.
- **A2 reading outside the band of P** would mean the spread is wider than BL's bands assumed. The 0.5 % band is
  then too tight for so_lists, and that is reported as such, not used to pass B.
