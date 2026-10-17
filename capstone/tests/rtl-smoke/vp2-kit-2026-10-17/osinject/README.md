# va-osinject: the monitor injects an OS-origin trap into S (pre-registered 2026-10-17, before any run)

**Why.** vp2 Tier 1 boot 1 (`../board-tier12/results/board-vp2-tier12-1.result-lines.txt`) wedged on a cause-24 trap
from S. The kernel module's `ldc` used an integer base, which the RTL refuses in every mode (Q-23). The monitor's
`handle_exception` then ran `fault_return_from_domain` with no domain active, `return_to_sumode` never undid the
CSCRATCH swap, and the next trap's `cincoffsetimm(sp)` at `_cap_trap_entry+4` faulted on S's integer sp forever.
medeleg cannot route 24..30 to S on this RTL (Q-24), so the monitor must deliver the kernel's own faults.

**The fix under test.** capstone-sbi `vp2/os-trap-inject` db23ffd (46c440c, then db23ffd), on capstone-bootstrap
c3293b3:
- for a trap from S or U (mstatus.MPP != M), `handle_exception` keeps its report, adds INJS and returns
  `CAPSTONE_INJECT_S` (-2) at the EXCX, ILLX and no-CPMP-region sites;
- the entry branches on -2 to `_inject_s_trap`: csm_inject's sequence fed from mcause/mtval/mepc, SPP from MPP,
  SPIE = SIE, SIE = 0, mepc = stvec, MPP := S. It leaves through `_handle_branch_1` and `return_to_sumode`;
- `fault_return_from_domain` stops with NODM when no domain call is outstanding (`dom_calls_active`, bracketing all
  eight compiled domain calls).

**The predicate is MPP, from the RTL** (the RTL lane, csr_regfile.sv at bf3db3175). A classic domain runs at M (the
switcher never changes priv_lvl), so its trap reads MPP = M, the kernel's MPP = S, and a U task's MPP = U. Only the
monitor's own state separates a domain's M trap from the monitor's own.

**Pinned (check each sha256 before building):** capstone-sbi db23ffd
- `sbi_capstone.S` 7b23b6a67540fd45cee4fa2bd66f5cdd7098b9c5208e1b0ce244367a5d520365
- `capstone_csmint.S` 77c5f13019379cdfff2f290647331fdd279b7e69115a51386c77fecd4bf7650f
- `sbi_capstone.h` 2c6f6bcab4ce180ee642382228ad52fb171bdb3840370bfad3f794cdba5dded1

Build as `va-csmint-entry` (`-I<capstone-sbi at db23ffd>`, the csmint-emu folder's `link.ld`), on bf3db3175's RTL.

## Revision 2 (2026-10-17, after the first run, before its re-run): row 3 and the combined monitor

The claim-auditor found that capstone_csmint.S emulated a CSMINT from U (pre-existing since c3293b3): a U process
could have had the monitor mint into kernel pages. The routine now emulates only a CSMINT from S (vp2/csmint-v7
67ed67f, the runtime lane's rule: QEMU refuses CSMINT at PRV_U at decode).
- **Row 3** is added: from U, `CSMINT(a2, a0 = slotB, a1 = descB)` at label `csmint_u`. The routine declines it, the
  entry's handle_exception stub injects the illegal instruction into S, and S then reads slotB's two words.
- **The arm now includes the combined monitor**, capstone-sbi vp2/tier1-monitor fba7812: the injection with its
  review fixes (vp2/os-trap-inject f8bfde5) plus the gated v7 routine. The include is sbi_capstone.S; the C files are
  not in this arm. Pinned:
  - `sbi_capstone.S` 7b23b6a67540fd45cee4fa2bd66f5cdd7098b9c5208e1b0ce244367a5d520365 (unchanged since db23ffd)
  - `capstone_csmint.S` 4600d04680158edf90396cc303cbb5aff4e2a286c9a0f302058ecd41232d6fbf
  - `sbi_capstone.h` 2c6f6bcab4ce180ee642382228ad52fb171bdb3840370bfad3f794cdba5dded1
- **Rows 1 and 2 are unchanged** in source and prediction, apart from row 2's return point moving into s9. The first
  run (results/rtl-bf3db3175.result-lines.txt) was on db23ffd's files.

| row | scause | stval | SPP | sepc | t2 | s5 | s6 | slotB w0 | slotB w1 |
|---|---|---|---|---|---|---|---|---|---|
| 3 (U CSMINT) | 0x2 | 0xa0b5165b (the instruction) | 0 | `csmint_u` | 0x7777 | 0x5555 | 0x6666 | 0 | 0 |

0xa0b5165b is `.insn r 0x5B, 0x1, 0x50, a2, a0, a1`, computed from the encoding. bf3db3175 makes an unknown custom
encoding illegal at decode with mtval = the instruction, which the routine's own mtval decode relies on.
- **What the ungated routine would read instead:** it would have emulated the CSMINT, read descB (all zero), and
  refused base >= end with 29 through csm_inject, which forces SPP = 1. So row 3 reads 29 and SPP 1 without the gate,
  and 2 and SPP 0 with it.

### Revision 2, result (the RTL lane, 2026-10-17, on fba7812's files): 23 of 23 as predicted

`results/rtl-bf3db3175-rev2.result-lines.txt`, read from the run's retirement trace.
- Rows 1 and 2 read as in the first run, at this build's addresses.
- Row 3 reads scause 2, stval 0xa0b5165b, SPP 0 and sepc `csmint_u`, the sentinels back, and slotB's words 0 and 0.
  So the routine declines a CSMINT from U, and the entry injects it into S as an illegal instruction, minting nothing.
- The control ends at the cycle limit with no reading: its handle_exception stub returns -1 1,392 times, and the ldc
  at ldc_s traps 1,393 times.

## What the arm runs

The shipped trap entry (`#include "sbi_capstone.S"`, CAPSTONE_TARGET_FPGA and CAPSTONE_VP2_CSMINT), with mtvec at
`_cap_trap_entry`. The C functions the entry calls are asm stubs, as in va-csmint-entry. `handle_exception`'s stub is
the patched C's decision for this arm's traps: MPP S or U returns -2.

1. **Row 1, S origin.** S executes `ldc t0, 0(a0)` with a0 = slotA's address, an integer, at label `ldc_s`. S's
   handler `s_trap` records scause, stval, sstatus.SPP and sepc, steps over the ldc, and `sret`s. S then `ecall`s:
   this is the SECOND trap after the injection, and it reaches the `handle_trap_ecall` stub only if the entry
   swapped CSCRATCH back (on boot 1 it could not).
2. **Row 2, U origin.** S `sret`s to U, whose page is marked U. U executes the same ldc at `ldc_u`. The trap goes to
   M, is injected with SPP = 0, and `s_trap` returns to S at `s_after_u`, which reports.

Each row prints seven values (`CAPPRINT`), then the arm passes:
scause, stval, SPP, sepc, t2, s5, s6. t2 is a register `_inject_s_trap` clobbers, set to 0x7777 before the trap; s5
and s6 are callee-saved sentinels. All three must come back from the frame.

## Predicted readings (addresses from `nm` on the built ELF, before the run)

| row | scause | stval | SPP | sepc | t2 | s5 | s6 |
|---|---|---|---|---|---|---|---|
| 1 (S) | 0x18 | `slotA` | 1 | `ldc_s` | 0x7777 | 0x5555 | 0x6666 |
| 2 (U) | 0x18 | `slotA` | 0 | `ldc_u` | 0x7777 | 0x5555 | 0x6666 |

Then the pass marker. 14 values in all.
- **stval = the faulting base's integer value.** That is what boot 1's MTVL read (its low bits were the module's slot
  pointer). If the RTL writes 0 instead, that one column reads 0 and is reported as such. It is not a failure of the
  injection.
- **Row 2's scause is 24 if U may execute LDC.** decoder.sv:1399-1404 has no privilege gate for LDC, and the DYN
  unit raises 24 on a NOT_CAP rs1 in every mode. If another gate refuses capability instructions in U, row 2 reads 2
  (illegal instruction): the csmint hook declines it and the same injection carries it, still with SPP = 0. That is
  what this row tests.

**Control (`-DOSINJECT_CONTROL`)**: the stub returns -1, resuming without injecting. The ldc re-traps forever,
`s_trap` never runs, and no row prints. The run ends at the cycle limit without the pass marker; per the
testharness note, compare the rows, not the exit status. This shows that row 1 cannot be satisfied without the
injection.

## What this arm does not cover, and where it is covered

- **The C half:** the MPP test, the three sentinel sites, the guard and the counter. They are checked in the
  generated monitor of the firmware built from db23ffd (fw_payload `d1a376ffe932…`, gate PASS):
  - handle_exception returns -2 at three sites;
  - INJS reports in handle_exception (2) and swap_cpmp (1), the latter after an MPP != 3 test;
  - NODM in fault_return_from_domain;
  - the counter at gp+208: 8 increments and 8 decrements around the 8 compiled domain calls, 1 test, 1 carve; 34
    references in all.
- **The two-sided half:** a classic domain's fault must still return CAPSTONE_DOMAIN_FAULT_RETVAL to its caller. No
  sim arm links the real monitor C. That half goes to the board, as a control rung before vm-contract in the next
  Tier 1 boot on the patched firmware: a classic domain that faults must return its fault code, as it does today.
