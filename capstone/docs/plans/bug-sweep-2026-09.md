# Bug sweep, September 2026 — every recorded issue re-tested against the current toolchain and silicon

**Status: IN PROGRESS, started 2026-09-05.** Plan and decisions: the compiler lane's approved plan
(the four lead decisions: this lane applies registry edits with the capstone/board lane notified
per batch; the full ~15-boot board set; compiler-side bugs fixed inside the sweep; the remaining
default-ABI `-fno-jump-tables` pins retired here).

## Method

- A verdict needs an instrument that can fire. `GONE` is recorded only when the same reproducer
  fires on the artifact it was written against (an older compiler build, a frozen image, or a
  mutation) and stays silent on the current one. Otherwise the verdict is `UNTESTABLE`, with what
  was tried.
- Verdicts: `STILL PRESENT` · `GONE` · `UNTESTABLE`. Each row names the command, rung or boot that
  produced it and the compiler library hash or bitstream it ran on.
- Root-caused hardware or compiler bugs get a `claim-auditor` pass before `GONE` is written into
  the registry (the weakest link is always "the repro no longer fires for a reason unrelated to
  the fix").
- Board rules as in `capstone/tests/board-results/`: control `k800` first, at most three unknowns
  per boot, an expected wedge last, distinct entry VAs, preflight GO, result lines only.

Compilers used: NEW = `compiler-validation-plan` at the sweep's head (lib hash per row); OLD = the
main checkout's build of 2026-09-04 20:18 (pre-cycle-1: crashes on C-20), used only to show an
instrument live.

## Verdict table

Columns: id · registry · old status · instrument · verdict · evidence · action.

| id | registry | old status | instrument | verdict | evidence | action |
|---|---|---|---|---|---|---|
| (filled per bucket) | | | | | | |

## Bucket log

### B1 — compile-only (no lock)

(pending)

### B2 — QEMU

(pending)

### B3 — RTL simulation at the flashed commit 5097eb166

(pending)

### B4 — board

(pending)

### B5 — registry edits applied

(pending)

### B6 — fixes

(pending)

## Controls for the sweep itself (recorded before any GONE)

A sweep that finds everything fixed must be able to find something not fixed. Three entries known
open were classified first, with the same instruments the sweep uses:

| id | instrument | verdict |
|---|---|---|
| C-32 | `llvm-lit` on `c32-movc-untagged-live.ll` (XFAIL) on lib ae821a017089 | STILL PRESENT (expected failure) |
| F-02 / F-03 | `llc -mtriple=capstone64 -mattr=+m -O2` on `findings/F0{2,3}-*/reduced.ll` | STILL PRESENT (DAGCombiner `visitLOAD`/`visitSTORE` asserts, rc 134) |
| Q-01 | `run-sqlite-memory.sh` (B2, pending) | expected STILL PRESENT |

### B1 — compile-only, first pass (2026-09-05 12:38; NEW lib ae821a017089, OLD lib 1cf7143fabee = the main checkout's 2026-09-04 build, post-c128)

| id | registry | old status | instrument | verdict | evidence | action |
|---|---|---|---|---|---|---|
| C-20 | ISSUES (prose only), compiler-repros | filed, not fixed | `C20-*/run.sh` | GONE, instrument live | OLD: `VERDICT: C-20 PRESENT -- __builtin_ctz crashes the backend`; NEW: `ABSENT -- the file compiles`; lit `c20-cttz.ll` | archive folder with a FIXED banner; entry needs writing (C-20 has no ISSUES entry) |
| C-21 | compiler-repros | filed 19-08 | `C21-*/run.sh` | pending: ABSENT on NEW and on OLD (OLD is post-c128, so the instrument was not shown live) | pre-c128 compiler build in progress | decide after the pre-c128 run |
| C-22 | compiler-repros | filed 19-08 | `C22-*/run.sh` | pending, same | same | same |
| C-23 | compiler-repros | filed 19-08 | `C23-*/run.sh` (has a positive control, exit 2 if the detector is broken; it was 0) | pending, same | same | same |
| C-2 | ISSUES | OPEN (partially widened) | rv8 qsort/miniz build at -O1 and -O2 through their build scripts | builds (rc 0 ×4); GONE pending the pre-c128 instrument | `/tmp/capstone/sweep/b1/rv8-*.log` | close with the pre-c128 evidence; lit `i128-logical-mixed-extend.ll` passes |
| C-11 | ISSUES | FIXED 2026-07-28 | `readelf -sW fw_jump.elf \| grep -c fw_fdt_bin` | GONE (0) | b1 log | archive |
| H-01 | compiler-repros | FIXED | `src/repro.sh` | GONE (rc 0) | b1 log | index README with FIXED |
| F-02 | fuzz findings | open | `llc -O2` on `reduced.ll` | STILL PRESENT | `DAGCombiner.cpp` `NewLoad.getNode() == N` assert, rc 134; OLD crashes earlier with the C-39 signature (pre-fix) | **fix** (B6) |
| F-03 | fuzz findings | open | same, store twin | STILL PRESENT | `NewStore.getNode() == N`, rc 134 | **fix** (B6) |
| C-32 | XFAIL | open | lit XFAIL | STILL PRESENT | expected failure on ae821a017089 | **fix** (cycle 3, B6) |
| S-12 workaround pass (W-01) | XFAIL `s12-movc-ldc-rename.mir` | inert on c128 | lit XFAIL | STILL PRESENT (inert) | expected failure | retire the pass, flag and test (S-12 fixed in RTL, 6/6 clean) |
| C-26, C-31, C-40, C-28, C-39, C-41 | lit pins | fixed in cycles 1–2 | `ptr-diff-signed.ll`, `c31-*.ll`, `c40-*.ll`, `tail-call.ll`, `fuzz-f01-*.ll`, `pseudo-expansion-roundtrip.ll` | GONE | each pin was red on the commit before its fix (recorded in the fix commits; `MUTATION:` headers) and passes now | entries for C-28..C-41 need writing in ISSUES.md |

### B2 — QEMU, slice a (2026-09-05 12:38–12:46; lib ae821a017089; QEMU 5dc356547d7f)

| id | registry | old status | instrument | verdict | evidence | action |
|---|---|---|---|---|---|---|
| Q-01 (control) | ISSUES | RESOLVED on the reference, nightly row red | `run-sqlite-memory.sh` | STILL PRESENT | QEMU stopped before the shell prompt right after the loader printed the domain segment (entry 0x10000, 265307 B); rc 1 | the sweep's first control fired; root-cause in B2 (kernel-module order-11 request) |
| C-16 | ISSUES | FIXED 2026-08-02 | `run-ladder-qemu.sh strarray` -O0 | GONE | `__CAPSTONE_LADDER_STRARRAY_PASSED__ (retval = 420)` = oracle | archive |
| C-4a | ISSUES | FIXED 2026-07-28 | `run-ladder-qemu.sh rv8_sha512` -O0 | GONE | retval 1390718314 = oracle | archive |
| C-12 | ISSUES | FIXED 2026-07-28 | `run-ladder-qemu.sh beebs_crc32big` with `DOMAIN_WINDOW=32k` | GONE | retval 1703161001 = oracle | archive |
| C-4b | ISSUES | FIXED 2026-07-28 | `run-ladder-qemu.sh beebs_crc32big` with `LADDER_NO_RO_COPY=0` | pending (first run rc 75 = boot infra flake; rerun in slice b) | | |
| I-3 | ISSUES | FIXED 2026-07-28 | `run-domain-smoke.py --domain-loader capstone-diag.user accum_probe.dom` | pending (first run: my share dir lacked the loader; rerun in slice b) | | |

RTL batch 1 at 5097eb166 (arm readings from the `.iss` logs): `linear-clear-audit` — arm 0
(MOVC of a linear source) prints NOT_CAP so the instrument sees a clear; arm 1 (CINCOFFSET,
NONLIN) unchanged, type 2; arm 2 (CINCOFFSET, LINEAR) prints NOT_CAP: **cincoffset consumes its
linear source on the flashed RTL** (R-21's cincoffset half GONE at 5097eb166; INIT is R-25's test).
`excode-base-audit`: two exceptions raised as designed (UNEXPECTED_OPERAND, INVALID_CAPABILITY)
— the +1 encoder offset is read from the .iss codes in the B3 table. `ldc-perm-check`: raises
UNEXPECTED_OPERAND on the write-through-read-only-cap arm = R-23's fix present. `movc-zero-
self-clobber`: SUCCESS 1715 cycles (R-19 still does not reproduce in simulation, as recorded).
`untagged-ldc-stc-128` / `-fixup`: SUCCESS (S-06 fix present at 5097eb166).

### B2 — QEMU, slice b (12:46; lib ae821a017089)

| id | registry | old status | instrument | verdict | evidence | action |
|---|---|---|---|---|---|---|
| C-4b | ISSUES | FIXED 2026-07-28 | `run-ladder-qemu.sh beebs_crc32big` with `LADDER_NO_RO_COPY=0` (rerun after a boot flake) | GONE | retval 1703161001 = oracle | archive |
| I-3 | ISSUES | FIXED 2026-07-28 | `run-domain-smoke.py --domain-loader capstone-diag.user accum_probe.dom` | (reading below) | `/tmp/capstone/sweep/b2/i3.log` | |

### B3 — RTL simulation at 5097eb166, batch 1 (12:42–12:47; Verilator, detached worktree, `.iss` logs under `/tmp/capstone/sweep/ariane-5097eb166/verif/sim/out_*`)

| id | registry | old status | instrument | verdict | evidence | action |
|---|---|---|---|---|---|---|
| R-21 (cincoffset/scc half) | ISSUES | OPEN, spec violation, confirmed 2026-08-11 | `linear-clear-audit` (arm 0 instrument control NOT_CAP, arm 1 NONLIN control unchanged, arm 2 LINEAR source) | GONE for cincoffset at 5097eb166 | arm 2 prints NOT_CAP (Reg[19] = 0), arm 0 shows the instrument sees a clear; 545 cycles, tohost 0 | status → "cincoffset/scc consume since the s12 lineage; INIT per R-25"; hand-off |
| R-23 | ISSUES | FIXED IN RTL 2026-08-12 | `ldc-perm-check` | GONE (fix present) | UNEXPECTED_OPERAND raised on the read-only-cap write arm, tohost 0 | archive |
| R-24 | ISSUES | OPEN, spec violation | `excode-base-audit` | STILL PRESENT (by design of the RTL; hand-off) | the two arms raise UNEXPECTED_OPERAND / INVALID_CAPABILITY; the +1 base is what the package documents; unchanged at 5097eb166 | keep OPEN, RTL lane |
| R-19 (sim half) | ISSUES / fpga-repros | OPEN, silicon-only | `movc-zero-self-clobber` | UNTESTABLE in sim (as recorded: never reproduced in Verilator) | SUCCESS 1715 cycles, 0 exceptions | board batch 6 |
| R-20 (sim) | ISSUES / fpga-repros | FIXED IN SILICON | `r20-stc-ld-x10` | GONE (fix present) | SUCCESS 775 cycles, 0 exceptions; the fix commit f623c48a1 is in 5097eb166 by content | archive package (resolved-but-retained as acceptance test) |
| S-06 (sim) | ISSUES / fpga-repros | FIXED in silicon | `untagged-ldc-stc-128`, `untagged-ldc-stc-fixup` | GONE | SUCCESS 537 / 847 cycles | archive package (retained as acceptance test) |
| S-07 / S-10 residual | ISSUES / fpga-repros S09, S10 | fixed in RTL, "not synthesised" | `s07-wbuf-forward-residual` (+ `-ctl`), polarity: 16 traps = residual NOT observed, 0 traps = LIVE, control first | residual NOT observed at 5097eb166 (17 UNEXPECTED_OPERAND = 16 legs + the positive control) in both arms | `.iss` counts | verify the fix is in 5097eb166 by content, then status → synthesised/flashed |

### B3 — RTL simulation at 5097eb166, batches 2–3 (12:48–12:53)

| id | registry | old status | instrument | verdict | evidence | action |
|---|---|---|---|---|---|---|
| R-25 | ISSUES | OPEN, source-verified, "directed test not yet written" | **new** `init-rs1-ne-rd.S` (arm 0: MOVC of a linear source → NOT_CAP, instrument control; arm 1: `INIT(a6, a5, 0)` with a5 UNINIT, rd ≠ rs1; arm 2: `INIT(s3, s3, 0)`) | **STILL PRESENT** | arm 0: Reg[17] = 0 (cleared), Reg[18] type 1; arm 1: the SOURCE a5 (Reg[15]) prints `Type : 1`, cursor 0x80003200, node 2 — identical to rd a6 (Reg[16]): two linear capabilities from one INIT; arm 2: Reg[19] type 1. 443 cycles, 0 exceptions, tohost 0 | hand-off to the RTL lane with the test (`/tmp/capstone/sweep/new-tests/init-rs1-ne-rd.S`, to be committed on their branch); registry: "confirmed by directed test on 5097eb166" |
| S-10b (tag route) | ISSUES | "tag route open", tests `s10b-storebuf-*.S` | UNTESTABLE here | no file matching `s10b`/`storebuf` exists in the RTL lane's checkout; the tests named in the entry live elsewhere (branch or worktree not present) | ask the RTL lane where the tests are |
| C-19 / C-31 sim half | ISSUES | RESOLVED 2026-08-26 (compiler reads a pointer's address with a plain move) | `alu-write-clears-shadow.S` copied from the RTL lane's tree (absent in the 5097eb166 testlist); its detector is CINCOFFSET, which traps UNEXPECTED_OPERAND on a capability-typed rs2, with a positive control placed LAST | **PASS at 5097eb166** (the compiler change is safe on the flashed RTL) | arms A, C and D print (Reg[12] cycle 344, Reg[13] cycle 355, Reg[17] cycle 362 — the rd == rs1, zero-gap shape the compiler emits), the run's only exception is the positive control's at cycle 365, after every measuring arm; the 2000013-cycle SUCCESS is the trap vector, by the test's design. First read as "traps and times out" — an instrument-reading slip: this test is MEANT to end in a trap, so the verdict is the ORDER of prints and exception, not their presence | none (stays RESOLVED; matches the RTL lane's 2026-09-05 reading on their branch) |

Presence by content in 5097eb166 (`git log e1b3db6ba..5097eb166`): the S-07 fix (5c5f4e3a7, "forbid granule co-residency in the write buffer"), the S-10 fix merged for synthesis (3d3ed1502 / 4fee13b2d), the S-07 instrument commits, and the S-12 fixes are all in the flashed RTL — so the registry's "S-10 … none synthesised into a flashed bitstream yet" (08-21) is out of date for route 1: `s07-wbuf-forward-residual` reports the residual NOT observed on 5097eb166 (16 legs trapped + the positive control). Whether the store-buffer tag route (S-10b) is in cannot be said from here (no test file); `page_offset_matches_o` is present in `store_buffer.sv` at :274-294.

### B1 close-out — the 08-19 compiler (421445f12447, the parent of C-23's filing commit; own build dir, llc+clang+lld) (12:54–12:59)

| id | on 08-19 | on pre-c128 (3cb3e621f21c) | on current (ae821a017089) | verdict |
|---|---|---|---|---|
| C-20 | **PRESENT** (`__builtin_ctz` crashes the backend, rc 1) | crash | ABSENT | GONE, instrument live on two older builds |
| C-21 | **PRESENT** (rc 1) | PRESENT | ABSENT | GONE, instrument live |
| C-22 | ABSENT ("the condition is still there") | **PRESENT** | ABSENT | GONE, instrument live on pre-c128 |
| C-23 | ABSENT (positive control fires: `control_returns_b` reads a2; "both functions read the high half") | untestable (`__int128` rejected in C) | ABSENT | **UNTESTABLE for liveness**: `run.sh`/`halves.c` are unchanged since filing (a merge-resolution commit only), and the parent of the filing commit does not exhibit the defect, so the build it was filed against is unrecorded; the current compiler is clean and the detector's positive control fires. Closed as "not reproducible on any recorded compiler; the c128 carrier makes the described mechanism (truncate-compute-reextend) inapplicable" — that is the honest line, not "fixed" |
| C-2 rv8 | build reached the link (clang + lld ran; the script then failed for lack of `llvm-readobj` in that build dir, rc 127 — not a compiler verdict) | pending: pre-c128 rerun with lld and the binutils built | qsort/miniz -O1/-O2 build | miniz half GONE (frontend crash shown live on pre-c128); qsort half pending |

### B4 staging status (13:05)

Staged under `/tmp/capstone/sweep/b4/{overlay,target,markers}` with QEMU-verified C13 markers unless noted; all at distinct VAs:

| rung | VA | level | QEMU | oracle | for |
|---|---|---|---|---|---|
| beebs_expint | 0x60000 | -O1 | PASS | host | R-8 |
| beebs_janne | 0x50000 | -O1 | PASS (generated `_app.c`) | host | R-6 |
| beebs_ns / nskeys / nsflat | 0x80000 / 0x90000 / 0xa0000 | -O1 | PASS | host | R-9 |
| beebs_nssmall | 0xb0000 (32k window) | -O1 | PASS | host | R-9 |
| gpw2 / gpn2 | 0xc0000 / 0xd0000 | -O0 | PASS | host | C-14 |
| k1200 / r14lp | 0x120000 / 0x130000 | -O0 | PASS | host | control refresh, R-14 |
| s06copy / s06aggcap / s06aggwide | 0xe0000 / 0xf0000 / 0x100000 | -O0 | PASS | host | S-06 acceptance |
| rawhazard5 / 6 / 7 | 0x20000 / 0x30000 / 0x40000 | -O0 | diag loader, dbg0..dbg5 = 5 (rh6), dbg0..3 = 5 (rh7) | host | R-1 |
| expint_diag | 0x70000 | -O0 | diag loader, dbg7 = 3883 (the correct value; the board read 2 in R-8) | none (reading) | R-8 |
| tagr / tagf | 0x140000 / 0x160000 | -O0, interp glue | diag loader, retval 1017 on QEMU (sideband is silicon-only) | 0 | M-1 |
| crc32_shrinkoff / crc32_shrinkon | 0x180000 / 0x170000 | -O2 | PASS / PASS | host | W-06/W-07 — **but both arms carry 1 shrink (the glue's): crc32 is not a rung where the flags change code; scanning BEEBS rungs for one that does** |
| frozen R-18 c8fix, rmB, c8, gzl, gz0, sn0; R-19 fdp0fix, fdpO1, fdpraw, fdp0 | 0x30000 / 0x60000 / 0xf0000 (collide pairwise: three boots) | as sent | **cannot run under QEMU**: the sent images fault OOB under the QEMU ladder loader (cause 7 at pc …42c, a store through a stale data capability) — they were built for the board's delivery path; markers record rc=1 honestly and SHA256SUMS verified | package expectation | R-18, R-19 |

C-2 close-out (13:07, pre-c128 build with lld and binutils): rv8 **miniz** -O1 and -O2 crash the pre-c128 clang (`Constants.cpp:2220 ConstantExpr::getCast` assertion) and build on the current compiler; rv8 **qsort** -O1/-O2 build on BOTH, so the qsort half of the entry is not shown live on any recorded compiler (the pre-c128 revision 3cb3e621f21c may already postdate the qsort shape). Verdict for C-2: **GONE** — instrument live on miniz, and the lit pin `i128-logical-mixed-extend.ll` covers the shape by construction.

### B3 — S-10b at 5097eb166 (13:11; tests from `origin/s10-merge-candidate`, run in the detached worktree)

| id | registry | old status | instrument | verdict | evidence | action |
|---|---|---|---|---|---|---|
| S-10b (store-buffer tag route) | ISSUES | "tag route open"; fix c867dfcbb on `origin/s10b-fix` / `s10-merge-candidate`, UNSYNTHESIZABLE (DRC LUTLP-1, 69-LUT loop) per the RTL lane | `s10b-storebuf-primed.S` (inverted polarity: a trap per leg is correct; the positive control runs first) | **STILL PRESENT** on the resident bitstream | SUCCESS 756 cycles (real RVTEST_PASS), 114 retired; positive control fired (one UNEXPECTED_OPERAND at cycle 403, the only handler entry); all 8 RESID legs retired; final readback `8 - trap_count = 8` → **0 traps / 8 legs**: every leg got a live capability over scrubbed memory. Arms reachable: control + legs 1–8, none unreachable | registry line: STILL PRESENT, fix exists but cannot be built into a bitstream; RTL lane owns |
| S-10b residual variant | — | — | `s10b-storebuf-residual.S` | uninformative by design | 9 handler entries = control + 8 legs trapping, trap_count == 8: "the condition was not created", which the header says is the reading on both fixed and pre-fix RTL | none |

### B4 — board batches as planned (written BEFORE any boot; predictions are the go/no-go input)

Every boot: `k800` first (0x10000, the control), at most three unknowns, the one expected to wedge or fault LAST, distinct VAs, preflight GO, result lines into `tests/board-results/`. Overlay = the sweep set staged under `/tmp/capstone/sweep/b4`.

| boot | rungs in order (VA) | IDs | prediction if the entry is STILL PRESENT | prediction if GONE |
|---|---|---|---|---|
| 1 | k800, beebs_janne -O1 (0x50000), beebs_expint -O1 (0x60000), expint_diag (0x70000) | R-6, R-8 | janne/expint return a wrong value; expint_diag dbg7 ≠ 3883 (the board read 2) | oracles 3-way agree; dbg7 = 3883 |
| 2 | k800, beebs_nskeys (0x90000), beebs_nsflat (0xa0000), beebs_ns (0x80000) LAST | R-9 | beebs_ns wedges (the original signature) | all four return |
| 3 | k800, rawhazard5/6/7 (0x20000/0x30000/0x40000) | R-1 | some dbg slots ≠ 5 (the pre-reflash RAW-hazard signature) | every dbg slot 5 |
| 4 | k800, gpw2 (0xc0000), gpn2use1 (0x1b0000), gpn2 (0xd0000) LAST | C-14, C-15 | gpn2 wedges; gpn2use1 = C-15's acceptance (links now; a fault would be new) | all return |
| 5 | k800, s06copy (0xe0000), s06aggcap (0xf0000), s06aggwide (0x100000) | S-06 acceptance on s12fix | — (fixed in RTL; a fault here is a bitstream regression) | all return |
| 6 | k800, k1200 (0x120000), r14lp (0x130000), beebs_nssmall (0xb0000) | control refresh, R-14, R-9 | — | all return, refresh known-good-controls |
| 7 | k800, crc32_shrinkoff0 (0x180000), crc32_shrinkon0 (0x170000) LAST | W-06/W-07 (-O0: OFF 1 shrink, ON 32) | shrinkon faults or miscomputes → the pins stay (silicon debt) | AGREE → retire the shrink pins |
| 8 | k800, tagr (0x140000), tagf (0x160000) LAST (deliberate fault) | M-1 | the fault loops forever, boot lost after tagf | tagf returns 0 |
| 9 | k800, c8fix (0x30000), rmB (0x60000), c8 (0xf0000) — frozen R-18 | R-18 | c8 = 567 | c8 = 576 |
| 10 | k800, gzl (0x60000), gz0 (0x30000), sn0 (0xf0000) — frozen R-18 | R-18 movc-zero arms | gz0 victim 9 damaged | gz0 = 576 |
| 11 | k800, fdp0fix (0x30000), fdpO1 (0xf0000), fdpraw (0x60000) — frozen R-19 | R-19 | fdpraw = 0x08000A31 | fdpraw = 2609 |
| 12 | k800, accum_probe (0x1c0000), accum2_probe (0x1d0000) | I-4 | probes return zeros | host values |
| 13+ | SQLite arms, one unknown per boot: sm0/sm at stage 164 (S-04), SQLITE_STATIC_BUILTINS stages 30–34 (R-15), uc/dp0 pair (R-17), the cycle-1 compiler's -O1 image (S-13) | S-04, R-15, R-17, S-13 | S-04 PRESENT expected on silicon; others unknown | — |

Frozen images (boots 9–11) cannot be QEMU-verified (board delivery path); their markers record the SHA256SUMS match and the QEMU fault honestly. Boots 1–8 and 12 are session 1; 9–12 session 2; 13+ session 3.

### B2 / Part A close-out (13:20)

| id | registry | old status | instrument | verdict | evidence | action |
|---|---|---|---|---|---|---|
| Q-01 | ISSUES | RESOLVED 08-20 (reference exists) / UPDATE 09-04: the nightly's only red row | `run-sqlite-memory.sh`, full console captured (second run; the first run's "QEMU stopped before the prompt" was a different failure and is not counted) | **STILL PRESENT**, same signature as the entry | `SQ: obs=18446744073709551615`, `create_dom failed`, rc 1; the memory-arm image is `LOAD filesz 0x265307 memsz 0x365bb0` (3.56 MB in memory) vs the silicon arm's 0x150818 (1.38 MB), both at -O0; the module doubles the request (code_len + max(code_len, 64 KiB)) so the memory arm asks for ~7 MB = order 11 > MAX_ORDER | fix candidate (B6): shrink the memory arm's image (its 1 MB of .bss and the extra code the extended workload pulls in) or move the row to `run-sqlite-silicon.sh`; the module (board lane's submodule) is not the place |
| Q-02 (d)/(e) | ISSUES | FIXED 09-04, "the nightly gap is still OPEN" | `qemu_staleness_guard` extracted verbatim from `run-nightly.sh` and run against a synthetic tree | **GONE** (gap closed, gate proven to fire) | positive control (binary older than a source file): prints `STALE QEMU … Every QEMU row below is suspect`, sets OVERALL_OK=0; negative control (binary newer): silent, rc 0 | registry: close (d)/(e) with the control line |
| C-15 | ISSUES | FIX WRITTEN, NOT YET BUILT | source + lit + the rung it was found on | **GONE** (fix in every build since 2026-07-30, pinned) | `isGpCaptableGlobal` at CapstoneISelDAGToDAG.cpp:118 (present in the pre-c128 and 08-19 trees too); lit `compiler-used-capability.ll` green (`llvm.compiler.used` in addrspace(200), CHECK-NOT on the symbol) — note the pin's RUN lines are default-ABI only, so a gp-captable arm is worth adding; gpn1use0 rung staged (QEMU 1463068797 = host); gpn2use1 (the rung the bug was found on) staging | registry: FIXED; add the gp-captable RUN arm in B6; board boot 4 carries gpn2use1 |
| W-17 pairs on the committed compiler | CLASSIFICATION | rows were on 57b5c5846ec3 | `run-workaround-pair.sh W-17 {coremark,rv8} -O2` on lib ae821a017089 | recorded | coremark AGREE-PASS (1 image differing), rv8 7/7 AGREE-PASS (2 differing: dhrystone, miniz); rows appended after the 13:10 marker in `workarounds/results/2026-09-05.tsv` | Part A record commit |
| W-06/W-07 board pair | CLASSIFICATION | silicon-debt candidates | `beebs_crc32` -O0 OFF (silicon config, 1 shrink) vs ON (32 shrinks), both QEMU PASS 1703161001; at -O2 every rung tried is byte-identical under the flags (nothing address-taken survives inlining; gp-captable globals need no shrink) | staged for boot 7 | shas e63f1876ac31 / fa180e09631f | board |

