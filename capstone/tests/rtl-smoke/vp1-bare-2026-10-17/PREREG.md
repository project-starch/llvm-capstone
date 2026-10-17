# vp1's first session: item B's translation and Q-17's check order as bare JTAG arms (pre-registered 2026-10-17, before any board run)

The first thing to run on `caplifive_vp1_9ed9a2594.bit` after the physical regression (the S-16 bare set, the Sublet
arms and one corpus set, which are the d93a8d66b images as they are and must read identical). Built by `build.sh` from
the RTL tree's sources (capstone-ariane `virtual-profile` 7234b6d43, tests only on top of the RTL of 9ed9a2594; copied
into `tests/` and checked equal at build time) with the bare M-mode wrapper of `sup-bare-2026-10-03-ladder` (env/) and
the node-aware header of `sublet-bare-2026-10-15` (inc/, copied). Runner: `sup-bare-2026-10-03-ladder/run_sup_bare.py`
(`SUP_IMG` = the image, `SUP_REC_ADDR` from the `.rec`, `FPGA_BITSTREAM` = the resident name); the report is the UART's
`SR <count>` / `SV <value>` lines. No SDK, no monitor, no Linux: the test's own monitor builds Sv39 tables in its data,
programs one all-permitting PMP entry for the page-table walker (entry 0, NAPOT all ones, RWX), sets satp and
`csupctl.virt`, and arms supervised domains with `CSSUPERVISE`.

| image | sha256 | board_rec | SR | readings (SV, in order), from the simulation of the committed tree |
|---|---|---|---|---|
| `images/vb-trans.bin` | `a3fa54d0e6f54f16618c16b9680c30567cc2c07efa7fffa49c0cf320eaeac7f1` | 0x80011800 | 36 | 0x0 0x1 0x0 0x28 0x0 0x77 0x1234 0x0 0x1 0x0 0x5a5a 0x0 0x0 0x5 0xd 0x0 0x0 0x1 0x99 0x0 0x5 0xf 0x0 0x0 0x0 0x5 0xf 0x0 0x1 0x0 0x5 0x1c 0x0 0x0 0x1 0x0 |
| `images/vb-trans-off.bin` | `79e670cc77352ee4486ec31f8b4681cae0200e4c2fdd781f252b5d51c2d653d1` | 0x80011800 | 13 | 0x0 0x1 0x0 0x28 0x0 0x77 0x1234 0x0 0x1 0x0 0x0 0x5a5a 0x0 |
| `images/q17-order.bin` | `ee2adf9b373e502838bc9739b61b6c325c5ca64411d22137bb0fbcd20ba6c66a` | 0x80004000 | 3 | 0x19 0x19 0x2 |

The vectors are the monitor-side CAPPRINT operands in retirement order on `virtual-profile` (run `sub`, 2026-10-17,
RTL 9ed9a2594); the three in-domain CAPPRINTs of arms 4 and 6 compile out under the wrapper (`BOARD_TEST_CH`), so the
board vector of `vb-trans` is 36 values against the simulation's 39. What each reading means is in `tests/vb-trans.S`'s
header, arm by arm: arm 0 identity under translation (1, then 0x77 and 0x1234 read through the translated domain);
arm 1 the store through the remapped page landing at the PHYSICAL page (0x5a5a, then 0 at the identity location); arm 2
the load page fault (csupstatus 5, csupcause 0xd, csupepc at the load, csuptval at the page) resumed to read 0x99; arm 3
the store page fault (5, 0xf, 0, 0); arm 4 the consuming ldc refused on the read-only page (5, 0xf, 0) and the resumed
ldc returning after clearing its source (1); arm 5 the bounds fault under translation (5, 0x1c, 0); arm 6 the
non-consuming ldc from a read-only page completing (1); the trap handler's mcause 0 last. `vb-trans-off` is the same
source with the flag off (csupctl 0, satp bare): arms 0 and 1 only, arm 1's two readings inverted (0 then 0x5a5a).
`q17-order`: 0x19 (a cderive from a dead child with a window outside it: the liveness query first), 0x19, trap count 2.

**The monitor that boots on vp1 is flag-OFF, by construction (2026-10-17).** The CSMINT hook arrives only in
caplifive-sbi cb91d7d, behind `CAPSTONE_VP2_CSMINT`. vp1's first session boots no monitor that has it:
- the bare arms above run with no monitor at all;
- corpus set 7's firmware 1d2bbbf37f63f435 is wrapper 1269c77 + monitor ebe0a25, whose tree has neither
  `capstone_csmint.S` nor the hook.

**Boot BL's flag-on firmware (26bc4b23c520d9d6, `vp2-kit-2026-10-17/csmint-emu/README.md`) is vp2-only. It must never
boot on vp1.** vp1 has no srevroot (that is item A, vp2's). Under that monitor a valid CSMINT reaches `csrr 0x5C1`
(`capstone_csmint.S:189`), traps illegal in M into the non-reentrant `_cap_trap_entry`, and wedges, as on d93a8d66b.

## Positive controls on the resident bitstream (the same images on caplifive_sublet_d93a8d66b)

| image | SR | readings (SV, in order), from the simulation of capstone-sublet b7e3287e4 with the same sources |
|---|---|---|
| `images/vb-trans.bin` on caplifive_sublet_d93a8d66b (positive control) | 36 | 0x0 0x1 0x0 0x28 0x0 0x77 0x1234 0x0 0x1 0x0 0x0 0x5a5a 0x0 0x1 0x0 0x14 0xffffffff7fff9000 0x5 0x99 0x0 0x3 0x8000000000000010 0x4 0xffffffff7fff806f 0x0 0x1 0x0 0x18 0x5 0x0 0x5 0x1c 0x0 0x0 0x1 0x0 |
| `images/q17-order.bin` on caplifive_sublet_d93a8d66b (positive control) | 3 | 0x1d 0x19 0x2 |

On d93a8d66b `csupctl` has no bit 1 and nothing translates: arm 1 inverts (0 then 0x5a5a), arm 2's load reads the
physical page (csupstatus 1, no fault, then the resume CALL re-enters and the domain returns: 5 is the stale status of
the second CALL's escape, 0x99 the word it stored), arm 3's store lands (1), arm 4 reads the parked word's low half
(the stale-status readings 5/1/0/0x18), arms 5 and 6 as on vp1; `q17-order` reads 0x1d first (the window test before
the liveness query, Q-17's defect) against 0x19 on vp1. A bitstream that reads vp1's vector on arm 1 translates; one
that reads the control's does not. `vb-trans-off` reads its 13 values on both bitstreams (the flag off is the physical
profile, and the 95-test sweep is identical between the two trees).

**Read on the board 2026-10-17 (caplifive_sublet_d93a8d66b, one power cycle per image, hashes checked):** `vb-trans-off`
13 of 13 and `q17-order` 0x1d 0x19 0x2 as above; `vb-trans` SR 36 with 33 values equal to the control vector and three
that differ from it by the IMAGE'S LINK LAYOUT, not by the silicon. Each of the three subtracts a link address from a
CSR whose value on the board equals the simulation's: position 16 (0-based) is csuptval - nomap with csuptval 0, and
`nomap` is 0x80007000 in the control simulation's ELF but 0x80008000 in the board image (`build.sh` rebuilt, sha256
identical; `llvm-nm`), so the board reads 0xffffffff7fff8000; position 23 is csuptval - ropage with csuptval 0x6f on
both, and `ropage` is 0x80008000 against 0x80009000, so the board reads 0xffffffff7fff706f; position 27 is csupepc -
fault4 where csupepc holds the pc of arm 4's RETURN (no escape on the resident bitstream), and the in-domain CAPPRINT of
arm 4 compiles out under `BOARD_TEST_CH`, so the RETURN is 0x14 past `fault4` on the board (0x80001d98 - 0x80001d84)
against 0x18 in the simulation. The raw csupepc values are the control trace's `csrr 0xfc2` results (run `vb7ctl`),
which place the stale pc at the RETURN of each returning arm. So the control vector for the board image is the row
above with those three positions replaced by 0xffffffff7fff8000, 0xffffffff7fff706f and 0x14, and the resident
bitstream read it 36 of 36. The vp1 vector has no layout-dependent entry: its arms 2 to 4 fault for real, so the epc
and tval readings are 0 by construction, and the arm-0 reading 0x28 is the same code in both builds (it matched).

**Flash authorisation (2026-10-17, the lead's own word in the board lane's session, naming this bitstream):** reflash
with `caplifive_vp1_9ed9a2594.bit` once:
- it seals;
- its preflight passes on the sealed sha256;
- ST100's overnight pair on d93a8d66b has finished.
The first session is this file's: the bare arms, the physical regression and corpus set 7, with the flag-off monitor.
It stops and reports on any reading that differs from d93a8d66b. The earlier approval named 4b958dc3f; this one
supersedes it for 9ed9a2594.

**Added before the run (2026-10-17, while vp1's chain waited on ST100): `vb-trans-plain`, run LAST among vp1's arms.**
- **What it tests:** R-56's clear-refusal gate (`ldc_store_ok`). It does a consuming `ldc` from a page writable from
  the start. vp1's one router-left combinational loop breaks through that gate (the synth and RTL lanes' risk list).
  vb-trans arms 4 and 6 exercise the same gate both ways.
- **The image:** `images/vb-trans-plain.bin`, sha256 997bc656df3be645…, board_rec 0x80009800. It is built by `build.sh`
  from capstone-ariane 1bcef8712's `vb-trans-plain.S` without `VB_TABLE`.
- **That file does not change vp1's other images:** its flag-off build of the edited `vb-trans.S` reproduces vb-trans
  and vb-trans-off's hashes above exactly.
- **The board vector is from the RTL lane's simulation of THIS image on 9ed9a2594** (37,040 retired): SR 5, values
  `0 1 0x66 0 0`.
  - 0: armed.
  - 1: csupstatus; the domain returned after the admitted consuming `ldc`.
  - 0x66: stored through the loaded LINEAR capability.
  - 0: the parked word, cleared.
  - 0: mcause.
  - The header's simulation vector is `0 0 1 0x66 0 0`; the in-domain CAPPRINT compiles out on the board.
- **The corpus set 7 gate now counts four arms as pre-registered.** There is no d93a8d66b reading of this arm.

**Amended before the run (2026-10-17): s1gap's place in the stop rule.** s1gap's cycle counts vary run to run on
d93a8d66b itself. It has three runs, 2026-10-16:
- s1gap-close's k = 3 gap read 0x1d, 0x22, 0x1d;
- s1gap-far's k = 1 revoke read 0x45, 0x42, 0x43, its k = 2 gap 0xfbc, 0xfbc, 0xfbd, and its k = 4 revoke 0x47, 0x4d,
  0x47.

So an exact match against the d93a8d66b record would stop the session on noise; the session's own script already said
"a cycle-count difference is reported, not a defect". The stop rule therefore applies to s1gap's functional readings:
the four revoke causes, base validity, traps, GAP and the count, and every reading of its control arm. These must be
identical. The eight cycle counts (positions 0, 1, 3, 4, 6, 7, 9, 10, the test's own layout in
s1-revoke-gap-2026-10-14/PREREG.md) are reported. The comparator is `cmp-s1gap.py`, tested both ways:
- d93a8d66b's two repeats pass against its record;
- a changed GAP reading fails;
- a missing arm fails;
- an empty reference is an error.

The S-16 set (98 readings identical across bitstreams) and the Sublet floor keep the exact rule.

**Amended at 20:25, before the retry (2026-10-17): an image that never started is re-run once.**

At 20:03 the session's S-16 arm s16st-esc-n8 never ran its image. The JTAG `load_image` timed out ("[riscv.cpu] Timed
out after 120s waiting for busy to go low") before `continue`, so the arm produced no reading, and S-16 read 17 of 18
arms identical to d93a8d66b. The stop rule is about readings that differ, and this arm has none. Two sources say how to
treat it:
- the same event on 36a641e0b (2026-10-03) was re-run once with the same image
  (`sup-bare-2026-10-03-ladder/results/board-36a641e0b.result-lines.txt:7-9`);
- the board-run skill classifies JTAG errors as infra, to retry.

The rule:
- An arm whose run log has no `] continue` line and no reading is re-run ONCE, with the same image and record address.
- An arm that started is never re-run: its reading stands.
- A second non-start is a missing reading, and stops the session.
- More than two non-starts in one set point at the board, not an arm: hold for the lead.

`continue-vp1.sh` applies it, and acts only if that non-start is the session's only failure (every other gate line
passed). It then:
- writes a new SEQ section carrying the session's other gate lines verbatim beside the new S-16 comparison;
- applies after-flash.sh's gate to that section, unchanged;
- runs corpus set 7.

It is stricter than the armed chain in one respect: vp2's chain starts only if set 7 is also identical to d93a8d66b
(the lead: "if vp1 as expected, go with vp2"). Dry run, with the runner, set 7 and the chain stubbed:
`results/dryrun-continuation.txt`. Each of its eight branches produced the predicted action.

Found on the way, and fixed before it could fire: vp2's chain waited for a free board with a `ps` pattern that matched
its own parent (rerun-then-vp2.sh, under vp1-session/), and that wait had no bound. A clean vp1 end would have left the
board idle. The chain now calls `../vp2-bare-2026-10-17/board-free-wait.sh`, which ignores the caller's ancestors and is
bounded at 6 h. It was tested two-sided, including against the live session. The earlier chain dry run
(`../vp2-bare-2026-10-17/results/dryrun-chain.txt`) had stubbed exactly that wait.
