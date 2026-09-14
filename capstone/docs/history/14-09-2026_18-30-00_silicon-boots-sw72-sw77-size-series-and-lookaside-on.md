# Silicon boots sw72–sw77 (2026-09-14): the speedtest1 size series closes at 1.18, the lookaside-ON pair reads 1.19, the allocator cells re-base on the #3 module, and the entry watchdog is proven live

Bitstream `caplifive_r30r31_1bfff7776`, CVA6 at 25 MHz, cycles from `mcycle`. Every row is cited by
image hash from the run-scoped log; a `k800 = 4` control opens every boot; the expected reading of
every arm was written in the driver header before launch. Sources of record: measurements doc §4g
(QEMU rows), §7o–§7q (the boots); ISSUES.md R-12, R-33, M-1, M-7; the state doc's per-boot rules.

## 1. Whole-benchmark ratios on silicon (domain / native, cycles)

| workload | lookaside pool | native cycles | domain cycles | ratio | boot |
|---|---|---|---|---|---|
| `main --size 1` | off | | | 1.2195 | sw63 (§7m) |
| `main --size 20` | off | 54,214,856,567 | 64,732,455,367 | 1.1940 | sw68 (§7o) |
| `main --size 20` | **on, both arms** | 53,142,976,993 | 63,437,512,052 | **1.1937 → 1.19** | sw77 (§7q) |
| `main --size 100` (speedtest1's default) | off | 337,235,381,252 | 398,572,346,349 | **1.1819 → 1.18** | sw73 (§7p) |

Both size-100 arms verified at `23674002 573a4409`, both size-20 ON arms at `3807866 2738af78`,
`DROPPED 0` throughout. Pre-registered before the boots: 1.18 (band 1.14–1.24) and 1.19 (band
1.15–1.24); both landed on the pre-registered value. The size-100 pair took 3.75 h + 4.43 h.

Decomposition, QEMU `-icount` instruction counts of the same binaries × the board's CPI:

| pair | instruction ratio | CPI native | CPI domain | CPI ratio | product |
|---|---|---|---|---|---|
| size 100, off (sw73) | 1.2383 | 3.746 | 3.575 | 0.954 | 1.182 |
| size 20, off (sw68) | 1.2547 | 3.831 | 3.646 | 0.952 | 1.194 |
| size 20, on (sw77) | 1.2528 | 3.795 | 3.616 | 0.953 | 1.194 |

The domain retires 24–25 % more instructions and runs them at a 4.6 % lower CPI; the ratio falls
slowly with size (1.2195 → 1.1940 → 1.1819) as the fixed boundary work amortises. SQLite's lookaside
pool — the configuration SQLite ships — takes 1.98 % off the native arm and 2.00 % off the domain arm
and moves the ratio by 0.0003; the native `--stats` arm reports 25,122 successful lookasides on
silicon. The "every silicon row is lookaside OFF" caveat is now a measured non-effect. Seven testsets
at size 1 (sw63): main 1.2195, star 1.2514, parsenumber 1.2677, orm 1.1843, fp 1.2435, cte 1.1654,
rtree 1.1879.

## 2. The allocator cells re-based on the #3 module (sw74, sw74b, sw75; `--size 1`, oracle `112006 38bb59fd`)

Under the #3 loader/module the non-representable Sublet arena request (1,419,584 bytes) is rounded up
to 1,421,312 (`ALEN:0015B000`), so the Sublet cell reports `HEAP 911104` where sw61 had 910,008 and
the sw60/sw61 rows needed re-basing.

| cell | image | HEAP | silicon cycles, #3 module | previous silicon | Δ | QEMU icount, #3 module |
|---|---|---|---|---|---|---|
| ⑥ Sublet, lookaside on | `ceeded2533a74bce` | 911,104 | **2,794,183,730** | 2,797,516,229 (sw61) | −0.12 % | 690,051,663 (archived 690,505,703) |
| ⑤ memsys5, lookaside on, 2 MiB static heap | `e6ee5255c896aa21` | 2,097,152 | **2,551,483,818** | 2,551,506,640 (sw60) | −0.0009 % | 678,572,868 (= archived) |
| ② native, lookaside on | `d95dd98c0de73c68` | 2,097,152 | **2,108,202,651** | 2,107,533,496 (sw60, same recipe) | +0.03 % | 535,283,834 |

Re-based Sublet-over-memsys5 on silicon: **⑥/⑤ = 1.0951** (was 1.0964), still a two-geometry
"configuration" comparison (a 1.42 MiB rounded arena against a 2 MiB static heap). The pool is live in
both domain cells: 25,010 successful lookasides for ⑤ (sw75 arm 4, its second run, 2,552,134,789
cycles) and for ⑥ (sw74b, as the boot's first run). ⑤ is unmoved by the module change and by the
stack-declaration knob (the loaded bytes are the same; the block is order 10 either way).

### 2a. The whole allocator matrix — native × capability, three allocator rows (`main --size 1`)

"Native" is the same SQLite amalgam and speedtest1, compiled by the same clang at the same `-O` for
`riscv64-unknown-elf` (rv64imac/lp64, no capability flags), run as an ordinary Linux user process on
the same core and bitstream: plain RISC-V on the same silicon, no domain, no monitor, no hostcall
boundary. "Capability" is the same source for `capstone64` with the gp-captable ABI, loaded into a
capability domain by the kernel module and entered through the monitor. QEMU columns are `-icount`
instruction counts; silicon columns are `mcycle` cycles; ratios are capability / native.

| allocator | native, QEMU | capability, QEMU | ratio | native, silicon | capability, silicon | ratio |
|---|---:|---:|---:|---:|---:|---:|
| memsys5, pool off (① / ④) | 545,623,496 | 692,983,497 | 1.2701 | 2,176,757,779 (sw59) | 2,639,069,185 (sw59) | 1.2124 |
| memsys5 + lookaside (② / ⑤) | 535,283,834 | 678,572,868 | 1.2677 | 2,108,202,651 (sw74) | 2,551,483,818 (sw75) | 1.2103 |
| Sublet, lookaside on (⑥) | none by construction | 690,051,663 | — | none | 2,794,183,730 (sw74) | — |

The native Sublet cell cannot exist: every Sublet primitive is a capability instruction (opcode
`0x5b`), so there is no non-capability build of it. Sublet is compared with the arm it replaces,
capability + lookaside: ⑥/⑤ = 1.0169 on QEMU and **1.0951** on silicon; against the unprotected
native build, ⑥/② = 1.2891 on QEMU and 1.3254 on silicon. The Sublet control with the pool
forced off (⑥′) is 705,997,303 instructions on QEMU. All three rows are pinned to size 1 by Sublet's
rev-node budget (43,355 nodes minted per run of 65,532); the memsys5 rows continue to sizes 20 and
100 in the §7 image family (128 MiB region arena) as the table in §1 — pool off at 1, 20, 100 and
pool on at 20.

## 3. R-33: the rounded arena's reclaim, on silicon

The revoke-reshare probe on the 1,419,584 request — the request sw60/sw62 wedged on — now runs to
`RR/done` with `released pool rc=1`, RCLM 0 → 1 and none of RCPR/RCSH/RCRE, three times (sw74 arm 2's
teardown, sw74b's probe, sw75 arm 5 at the representable 4,194,304 as the control, where only the
tables region rounds, `ALEN:001AB800`). Boot sw72 gave the rounding log its positive and negative
control in one boot (`not-representable-lines=1` at 1,419,584, 0 at 4,194,304). sw60's 1,728-byte
RCSH "shortfall" was the bounds re-encoding gap already filed under R-33, and the fill was complete
then too. Attributing the clean reclaim to the rounding is inferred, not measured: the settling arm
(monitor 4274268 + a pre-rounding module + the 1,419,584 request, predicted `RCRE:000006C0`) is owed.

## 4. The entry watchdog's live positive control (sw76)

sw64's image `23da3b126a304585` stalls deterministically at share3 (`SHA5`, no `SHA6`). The watchdog
fired on the live board — `ENTRY-STALL 781s  last share marker=SHA5:00000001, no SHA6 for 421s ->
Aborting runner` — the runner was terminated (rc=130), the board released, and the run-scoped log
carries exactly one boot banner: no reset. Every earlier "the watchdog would have caught sw64" was a
replay; this is the first live firing. M-1 (the trap vector in measurement images) stays open.

## 5. Two per-boot limits, each found by losing one arm

| limit | signature | arms lost | classification |
|---|---|---|---|
| **one Sublet workload per boot** | the SAME image's second run enters, then wedges; `rev_node_head = 0xFFFF` — one run mints split 5,481 + mrev 37,874 = 43,355 of the 65,532 nodes | sw74 arm 4, sw74b arm 4 | R-12's budget, not a defect |
| **one REGION_ARENA workload per boot** | the SAME image's second run stops at `SQ: C2/mkarena`: a second 128 MiB `create_region` from a 256 MiB CMA area that still holds the first (the host never releases its arena); trap log 0x83, head 289 | sw77 arm 6 | not R-12, not M-7; host-side, N = 1 |

memsys5 cells repeat freely within a boot (sw75 arm 4; sw73 ran size 100 in one boot), so a
`--stats` positive check for a domain image goes on QEMU or on a memsys5 cell, never as a second
Sublet or REGION_ARENA run.

## 6. Instruments changed today (each with a positive and negative test)

* Runner: a never-entered arm is reported from its transcript's last share tag and host marker; the
  R-16 wording only when the transcript ends at SHA5 (sw77 arm 6 had been printed as R-16 from fixed
  text). The S-15 share-trap summary line was keyed by the bare domain path and could never fire for
  a selector-bearing arm; keyed by label now. Host refusals (`0x5117BADn`) decode before the S-07
  decoders.
* `arena-mismatch-gate.py` refuses a non-representable arena size before a boot is spent.
* `precommit-scan.sh`: a missing or empty FPGA-console file BLOCKS instead of silently skipping the
  URL check; the loud override is `PRECOMMIT_SCAN_NO_URL_FILE=1`.
* Eight private-struct hosts grown to the #3 module's ioctl struct and pair-proven on QEMU.

## 7. Owed, and decisions for the lead

Owed, unscheduled: the R-33 settling arm (one short boot); the size-100 instret image (~3 h); the
seven testsets at size 20 (~9 h, seven boots); R-33's bottom-truncation arm (shape not yet written).

Decisions: the paper proposal (`docs/plans/2026-09-14-paper-evaluation-update-proposal.md`, now with
the lookaside-ON row; no edit to the paper); the M-1 trap-vector default; closing llvm #14 and #18 on
GitHub (landed by content); the status of llvm #2 and #3 (August, on no branch); capstone-qemu #3
lands as a plain merge when its rebase arrives. The full PR table (27 llvm merges, 3 buildroot, 1
qemu) is §1 of `docs/plans/2026-09-14-next-silicon-experiments.md`.
