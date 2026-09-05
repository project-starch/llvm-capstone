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
| S-10b (tag route) | ISSUES | "tag route open", tests `s10b-storebuf-*.S` | ~~UNTESTABLE here~~ **SUPERSEDED 13:11**: the RTL lane located the tests on `origin/s10-merge-candidate`; run at 5097eb166 below → STILL PRESENT | no file matching `s10b`/`storebuf` exists in the RTL lane's checkout; the tests named in the entry live elsewhere (branch or worktree not present) | ask the RTL lane where the tests are |
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
| 1 | k800 (the image's control is the 0x20000 build, sha 589ceee3853c6092), beebs_janne -O1 (0x50000), beebs_expint -O1 (0x60000), expint_diag (0x70000) | R-6, R-8 | janne/expint return a wrong value; expint_diag dbg7 ≠ 3883 (the board read 2) | oracles 3-way agree; dbg7 = 3883 |
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

### B2 — Q-03 reproduced (13:17–13:23, `run-domain-batch.py`, the camp-cycle2 manifest played twice = 34 items, QEMU 5dc356547d7f)

31 RET / 3 WEDGE. `cs7-O0` WEDGED at BOTH its positions (12 and 29) while `cs7-O2` returned 505522532 at 13 and 30; `cs2-O2` returned 599932085 at position 5 and WEDGED at position 22 (after the cs7-O0 wedge-and-reboot at 12 … so position 22 is item 10 of its boot). Every other image agreed with itself at both positions. So two effects are on the table: cs7-O0 looks image-bound on this run (2/2 wedges; the entry's batch C had it RET at position 1), cs2-O2 position-bound (1/2). Manifest, logs and results under `/tmp/capstone/sweep/b2/q03/` (to be committed under `tests/fuzz/findings/F04-*` with the root cause). Root-cause step next (B6): bisect the boot state — the wedge follows a reboot in both cs2 cases? — by replaying cs7-O0 alone as item 1, then after N returning items, then after a FAULT.

Auditor residue (a) closed after the merge: of 799 `runtime-qemu` + authority sources, 6 emit a jump table in the default ABI (beebs_cover, beebs_duff, gp_diag halves) and all 6 are silicon-ladder sources, where `SILICON_FLAGS` makes the backend refuse tables (checked with the script's own flags: 0 `.LJTI`). Consequence recorded: with the residual pins, the only nightly carriers of the new lowering are CoreMark -O2 and two RV8 benchmarks — the B2 pin retirements (BEEBS shared script, rv8 aes/primes, default-ABI SQLite) are what widens that.

### B6 — Q-01 fixed (13:35)

`run-sqlite-memory.sh` builds the amalgamation at -O1 with the 256 KB arena (default ABI kept, glue/libc/VFS still -O0). Image `LOAD filesz 0x12bf17 memsz 0x16c7c0` (1.49 MB: .text 1.15 MB, .bss 258 KB) against 3.56 MB before; doubled it fits order 10; all five markers reached, rc 0. The nightly's only red row goes green on its next run. -O0 SQLite coverage stays with the SLT twins.

### B4 — session 1 results (board, bitstream caplifive_s12fix_5097eb166, firmware sha256 aea4fcadf670… rebaked without tagr/tagf → see the results file for the sha in force)

| boot | rungs | readings | verdict |
|---|---|---|---|
| sw01 | k800=4 (control OK); beebs_janne 484656629; beebs_expint 2021290181; beebs_nssmall 2711842293 | all three = host oracle | R-6 GONE on silicon (janne -O1 correct), R-8 GONE (expint -O1 correct; the dbg reading follows in the diag boot), R-9 nssmall OK || sw02 | k800=4; beebs_nskeys 3914083333; beebs_nsflat 1184999093; beebs_ns 1184999093 (LAST, the rung that hung pre-reflash) | all three = host oracle; beebs_ns RETURNED | R-9 GONE on silicon (explained by C-13's fix, as the entry predicted) || sw04 | k800=4; gpw2 3983810698; gpn2use1 1463068797; gpn2 3976364985 (LAST, the wedge-expected rung) | all three = host oracle; gpn2 RETURNED | C-14 GONE on silicon (the movc-scalar-copy fix is in); C-15 acceptance on silicon PASS (gpn2use1 links and returns the host value) |

### B6 — F-02 / F-03 fixed (13:45)

Root cause, from gdb at the assert: `N` = `load<(load (s8), addrspace 200), anyext from i8> t55, FrameIndex:c128<0>` (pointer info with address space 200 and no value — `MachinePointerInfo::getUnknownStack` is patched to `MachinePointerInfo(AllocaAS)`), `NewLoad` = `load<(load (s8) from %fixed-stack.0, align 4)> …` — a NEW node, because once the element pointer folded to a bare FrameIndex, `SelectionDAG::getLoad` re-inferred a fixed-stack pointer info whose address space came from `TargetMachine::getAddressSpaceForPseudoSourceKind`, 0 by default; the CSE key includes the address space, so the "same node back" assumption at `DAGCombiner.cpp:20314` broke. Fix, two halves: `CapstoneTargetMachine::getAddressSpaceForPseudoSourceKind` returns the alloca address space for Stack/FixedStack/ConstantPool/JumpTable/GOT (AMDGPU does the same for its private/constant spaces) — and, because that alone moved the mismatch to loads created with `MachinePointerInfo()` (address space 0, re-inferred to 200: `vararg.ll` asserted the other way round), `InferPointerInfo` in `SelectionDAG.cpp` now gives a value-less address-space-0 pointer info on a capability pointer the capability address space before inferring (shared code; drift manifest regenerated, `shared-patches-present.test` green). Red first: `fuzz-f02-f03-vector-elt-stack-temp.ll` failed on the unfixed llc (the assert), green after; Capstone lit green (see the build log line in the results). F-02/F-03 lines dropped from `known-signatures.txt`; the finding folders marked FIXED.| sw05 | k800=4; s06copy 32; s06aggcap 15; s06aggwide 255 | all three = host oracle | S-06 acceptance on s12fix (5097eb166) PASS: no bitstream regression of the untagged ldc/stc fix |

### B6 — C-32 scoped, not fixed (13:55): the spec sides with the RTL

`capstone-spec/parts/cap-man-insn.adoc` (MOVC): "If `x[rs1]` is not a non-linear capability (i.e., `type != 1`), write `cnull` to `x[rs1]`." A NOT_CAP source is "not a non-linear capability", so the RTL nulling an untagged source (rtl-oracle 2026-09-04, `capstone_flu_unit.anvil:13-26`) is CONFORMANT, and **QEMU is the divergent side** (`op_helper.c:580-585` nulls only a tagged non-copyable source) — which is why every QEMU run of the C-32 shape passes. Two consequences: (1) QEMU is a permissive oracle for any copy of an integer-bridged pointer that stays live — worth a QEMU-side entry (board lane's submodule) so QEMU stops hiding the class; (2) the compiler fix is real and not small: a value produced by `inttoptr` lives in a capability register untagged, and every register-allocator copy of it is a `movc` that destroys the source on silicon. Candidate designs, for the lead: keep bridged integers in integer registers and re-bridge at each use (a rematerializable bridge pseudo, so no c128 copy of an untagged value is ever needed); or a register class that distinguishes a bridged integer so `copyPhysReg` can pick `mv`. Either touches the ABI of integer-bridged pointers; parked as a decision item, XFAIL pin `c32-movc-untagged-live.ll` stays.

### B6 — Q-03 ROOT-CAUSED (13:58): the QEMU monitor stand-in spins on an exactly-matching free region

Bisect: manifest A (cs7-O0 alone as item 1) → RET, so not image-bound; manifest B → WEDGE at positions 8, 12, 22 hitting `cs2-O0` twice (an image that never wedged in the 34-item batch) and `cs7-O0` once, while `cs7-O0` after 1, 3 and 11 returning items RET. The wedged items' loader output stops after `Loadable size = …` — **no `Created domain ID`**, then `[CAPSTONE] Print = Scalar(0x1234)`; returning items print `Created domain ID` and never print 0x1234. So the wedge is inside `create_dom`, before the domain runs, and 0x1234 is printed at one site, `split_out_cap`, which `caplifive-buildroot` carries twice — `package/capstone-sbi-domain/capstone-sbi/sbi_capstone.c:243-248` (→ `sbi.dom`) and `components/opensbi/lib/sbi/capstone-sbi/sbi_capstone.c:246-247` (→ `fw_jump.elf`, the LIVE create path: the module's DOM_CREATE is an SBI ecall, `modcapstone/module/capstone.c:122`; the board lane caught this before rebuilding, else the positive control would have read as 'port did not fix it'): when the requested `[base, base+len)` EXACTLY equals an existing free region, `// matching region. We don't support this for now` → `C_PRINT(0x1234); while(1);`. Whether an allocation matches a leftover region exactly depends on the buddy allocator's block choice after earlier carves (the module never frees a domain's pages, so the free-region list is the boot's memory minus carved blocks, fragmenting into remainders that a later same-order block can equal) — position- and size-history-dependent, any image: Q-03 exactly. **The board firmware already fixed this case** (`caplifive-system/sw/buildroot/components/opensbi/lib/sbi/capstone-sbi/sbi_capstone.c:534-540`: "allocator happened to hand back an exactly-matching region — a layout coincidence, not a capability error. That hang was misattributed to silicon during the 2026-08-01 investigation … `region` is already `mem_l` … All that remains is to shrink the pool"), and the `caplifive-system` package copy of the stand-in (`:234`) still spins silently. Fix = port the firmware's handling into the QEMU stand-in (board lane's submodule); confirmation batches E (same image ×4) and F (two sizes alternating) queued behind the validation suites.| sw06 | k800=4; k1200 4; r14lp 4; gpn2use0 1463068797 | all = oracle | control refresh: k1200 and r14lp (the R-14/R-16 acceptance rungs) still pass on s12fix; known-good-controls can drop its STALE banner |
| sw07 | k800=4; crc32_shrinkoff0 1703161001 (silicon config, 1 shrink); crc32_shrinkon0 1703161001 LAST (32 shrinks, -O0) | both = oracle | W-06/W-07: the shrink flags are NOT silicon debt on this rung — 32 SHRINKs execute correctly on s12fix. One rung; retirement of the silicon-config pins goes through an SQLite silicon arm built with shrink ON (session 2) before the pins move |

Q-03 confirmation batches (13:57–14:00): E (the same image four times) 4/4 RET; F (two sizes alternating, six items) 6/6 RET. So same-size repeats do not trigger the exact fit — consistent with the module never freeing a domain's pages (nothing is reused; an exact fit needs a later block to equal a carve REMAINDER). The site is identified by its unique signature (the 0x1234 print with no `Created domain ID`), not by these batches; a determinism test (manifest B replayed verbatim, and with two items swapped) is queued to tell geometry-bound from timing-bound.

### B6 — validation of the F-02/F-03 compiler change (QEMU, new libs CapstoneCodeGen 0e20b3edd1f0 / SelectionDAG b0cb06db1263)

RV8 -O0 7/7 PASS, RV8 -O2 7/7 PASS, CoreMark -O2 PASS (13:50–13:59); BEEBS -O0 and -O2 twins running, then the agreement gate.| swd1 | k800=4; rawhazard6 dbg0..5 = 5; rawhazard7 dbg0..3 = 5 (retval 48879 = the probe marker); rawhazard5 VOID (VA collision with the control, see below; it printed dbg0..4 = 5, recorded as an observation only) | rawhazard6/7: every dbg slot identical to the QEMU reading of the same bytes | R-1 NOT REPRODUCED on the two valid probes (header unchanged; two probes on one bitstream do not retire an entry); rawhazard5 to be relinked and rerun. **Staging slip recorded**: rawhazard5 was linked at 0x20000, the VA of the control build in the image; the preflight had no entry-VA check (the rule lived in the board-run skill's prose) and the boot went out. It did not stall, which is NOT a verdict on R-3. The check is now C15 in `preflight-board-run.sh` (the board lane's commit 84129116f659 on dev; my parallel C16 draft, negative-tested on the same pair, was dropped in favour of theirs); rawhazard5 gets relinked at a distinct VA before any further boot || swd2 | k800=4; expint_diag retval 3883; accum_probe 100; accum2_probe 3883 | expint_diag = 3883 = QEMU = the correct value (the board read 2 in R-8's day); accum_probe = 100 = its HOST oracle (QEMU through the diag loader printed 3883 — the two loaders do not report the same slot, so QEMU-diag vs board-lpc is not a comparison); accum2_probe has no host oracle, 3883 is a reading | R-8 GONE on silicon by the diagnostic too; I-4 ("probes returning zeros"): both probes now deliver non-zero results and accum_probe delivers the host value — the zero-region symptom is GONE; the diag-loader vs lpc slot semantics are the residual to write down, not a defect |

Boots 9–11 (frozen R-18/R-19 images) — entry-contract control, written before the boots: both packages ran on `caplifive_65536_nodes.bit` with the interp entry glue (`DOMAIN_GLUE=interp`), i.e. an August firmware; today's stack is s12fix (5097eb166) with the current monitor. Whether an August interp-glue image still enters today is not assumed: each boot's FIRST frozen image is the package's own correct-on-any-silicon arm (`c8fix` → 576, `gzl` → 576, `fdp0fix` → 2609). If that arm stalls or misreads, the boot is UNTESTABLE for the package (entry contract changed), not evidence about R-18/R-19; only if it returns its value do the defect arms (`c8`, `gz0`, `fdpraw`) carry a verdict. VAs 0x30000/0x60000/0xf0000 against the 0x20000 control, C15 will check them.

Q-03 determinism (14:15–14:18, manifest B replayed verbatim on the old stand-in): WEDGE at exactly the same positions 8, 12, 22 (`fill02-b5`, `cs7-O0-after5`, `fill10-b11`). Deterministic for a given sequence → allocation-geometry-bound, not timing — as the exact-fit mechanism predicts (the buddy allocator's block choice is a function of the carve history alone). The positive control for the fix is the same manifest returning all 24 on the rebuilt stand-in.

### B4 — session 2 (frozen R-18/R-19 images, one bake per boot because the packages share entry VAs and C15 scans every staged .dom)

| boot | rungs | readings | verdict |
|---|---|---|---|
| sw09 | k800=4; c8fix 0x04090240; rmB 0x04090240; c8 0x04090240 (LAST, the defect arm) | the package's table: `0x04090240` (p=64, k=9, qc=576) is CORRECT, `0x04090237` (qc=567) is the R-18 reading; the entry-contract control (c8fix) returned its value, so the August interp-glue images enter today's firmware | **R-18 scalar-store clobber arm GONE on 5097eb166**: `c8` reads the correct packed word. (My oracle files held the unpacked field 576, so the driver's MISMATCH tags on this boot are a scoring artefact; the rows carry the packed oracle.) || sw10 | k800=4; gzl 0x00090240; gz0 0x00090240 (the `movc a0, zero; sw` victim arm); sn0 1000576 | gzl/gz0 carry k=9 intact with qc=576 — the package's "victim 9 damaged" signature is absent; sn0 = 1000576 = the package's correct value | **R-18 movc-zero arm GONE on 5097eb166** (gz0 reads exactly what gzl, its clean twin, reads); the sn0 lower-half arm correct as expected || sw11 | k800=4; fdp0fix 2609; fdpO1 2609; fdpraw 2609 (LAST, the victim-slot arm) | the package expects fdpraw = 0x08000A31 (= 0x08000000 + 2609) on defective silicon and 2609 clean; fdp0fix (the entry-contract control) and fdpO1 read their clean values | **R-19 victim-slot arm GONE on 5097eb166**: `fdpraw` reads 2609 || sw12 | k800=4; fdp0 2609 (alone: its VA collides with fdp0fix) | the README expects `0x08000A31` on defective silicon for this -O0 `movc a0, zero; sw` initialised arm; raw line `RESULT fdp0 retval=2609 cycles=26727` | **R-19 NOT REPRODUCED on 5097eb166 on either arm** (fdpraw 2609, fdp0 2609). "Not reproduced", not "gone": frozen August images on today's firmware, one clean boot each against fifteen defective ones recorded in the R-18 README |

Validation chain on the S-12-removed compiler (lib 50818be7cbe6, 14:43–14:45): Capstone lit CodeGen 89, MC 11, MC/Disassembler 2 — all pass; RISCV 2257 tests, 2 failures (`emutls.ll`, `rvv/debug-info-rvv-dbg-value.mir`) — MEASURED pre-existing: both fail identically on the unchanged 08-19 build (421445f12447, its own build dir with FileCheck/llvm-config built for the purpose, 15:27), so the shared-code change adds no RISCV failure; coverage gate 0 gaps; byte identity of the S-12 removal: a gp-captable rung (beebs_crc32 -O2) and two default-ABI objects (cap-control-flow -O2, intrinsics -O0) IDENTICAL before and after the deletion.

### B6 — Q-03 positive control on the ported stand-in (14:45–14:48; fw_jump 7cfcd014b2a8, sbi.dom cc2320e2a3d0, both monitor copies patched by the board lane, tail-slot exact fit handled as in the firmware)

Manifest B verbatim: 21 RET, 3 WEDGE at the SAME positions (8 `fill02-b5`, 12 `cs7-O0-after5`, 22 `fill10-b11`); every wedged item prints `0x1235`, `0x8`, `0xa` and no `Created domain ID`; zero `0x1234` prints in the batch log. Manifest F: 6/6 RET. Reading: the old spin is gone; the live case on the QEMU path is the MIDDLE-slot exact fit (the request matches slot 8 of region_n = 10, each time), which the port — like the firmware — only reports and stops on. Q-03 is NOT FIXED, it is narrowed to that case with the slot numbers in hand. Proposed fix shape (sent to the board lane): leave slot i empty (clear its CPMP mapping, mark it empty, skip empty slots in the search) instead of moving a LINEAR capability between slots; positive control = the same manifest at 24/24 with zero 0x1235 prints. Note for the board: the firmware carries the same limitation (`CAPSTONE_SPLIT_EXACT_FIT` tail only), which the preflight has been budgeting around as C7's "~5th create_dom" ceiling.| sw13 | k800=4; tagr 1017; tagf 1017 (LAST, the deliberate fault) | encoding 1000 + 10·type_after_cap_store + type_after_integer_overwrite (tagr uses 7 as a placeholder, no second query): tagf's second `lcc` type query read **7**, not a live type (1 LIN / 2 NONLIN) — the plain integer store cleared the capability. The package's 2026-08-04 expectation was a TRAP because `lcc` on an untagged value faulted then; today QEMU through the diag loader also returns 1017, so the type query no longer faults on NOT_CAP on either side | RTL-store-user-metadata (a code-level observation, not a defect entry): invariant RE-CONFIRMED on 5097eb166, and the software-visible instance it pointed at, R-19, is not reproduced (fdpraw/fdp0 = 2609); **M-1 not exercised** (fix known: `INTERP_EXTRA_CFLAGS=-DINTERP_DOMAIN_MTVEC=1`, unverified for want of a faulting rung) — no fault occurred, so "does a domain fault loop forever" still needs a rung that actually faults (e.g. an out-of-bounds load); the entry stays OPEN with that note |

Compiler-change validation closed (14:48–15:12, lib CapstoneCodeGen 50818be7cbe6 after the S-12 removal): BEEBS -O2 twin 81/81 PASS (2 infra retries); agreement gates `compare-twins.py`: beebs O0/O2 81/81 AGREE-PASS, rv8 O0/O2 7/7 AGREE-PASS. With RV8 -O0/-O2 7/7, CoreMark -O2 and BEEBS -O0 81/81 earlier, the F-02/F-03 fix and the S-12 retirement are validated on lit (Capstone 89/11/2, RISCV control) and the QEMU corpus; committed and offered for the ff.

Q-03 after the board lane's audit (15:15, ISSUES.md 40cb0d6501bb on dev): the compaction design is REFUTED — region ids escape to the guest as array indices (`create_region` returns `region_n-1`; the kernel module caches, looks up and mmaps by that index, with no pop path), so moving a slot retargets a live guest id; and the LANDED tail-slot port (`drop_exact_fit_tail`, fw_jump 7cfcd014b2a8 / sbi.dom cc2320e2a3d0) is the first shrink of `region_n` reachable from the module and inherits a weaker form of the same hazard (a later `create_region` reuses an id the module already holds). So the port is "landed, positive control fired, module-desync hazard recorded" — not verified, not fixed; manifests B/F did not exercise the hazard. The correct fix is the hole with a `region_live[]` sentinel across ~11 consumer sites, to go through a plan and a second audit before code (board lane). The board firmware fixed the same site the same way on 2026-08-01, so the hazard likely exists on silicon firmware too — being checked by the board lane. Also retracted by that audit: the claim that an inline array assignment in the nested branch fails under both capstone-c pipelines (it compiles cleanly under both).

dev fast-forwarded to 24ee7dd3013c and pushed (15:14); the board lane's audit commit sits on top.

### B2 — W-15 residue: pin retirement, prediction written BEFORE the run (15:25)

`retire-pins.py` removes `-fno-jump-tables` from the BEEBS shared script and the 44 per-benchmark scripts, rv8 aes/primes, the default-ABI SQLite build and the core-init-state repro (dated note once per file, never on a continued line); left: the two plain-riscv64 baseline halves, the yield probe, the three frozen fpga-repros copies. Validation on the committed compiler (lib 50818be7cbe6): RV8 -O0/-O2 twins + gate, BEEBS -O0/-O2 twins + gate, the sqlite-memory row. Attribution: the 83 + 84 BEEBS twin images from the pre-retirement runs are hashed (`/tmp/capstone/sweep/beebs-O{0,2}-doms-before.sha`); **prediction: only the benchmarks whose compiled code contains a dense `switch` (a `.LJTI` label in their assembly) change image, every other image is byte-identical** — a failure in an unchanged image is not the pin's. The auditor's source scan found tables only in `beebs_cover` and `beebs_duff` among the ladder halves; the full BEEBS sources may add a few (to be listed from the assembly after the run).

Pin retirement, as it actually ran (15:34): `retire-pins.py` edited the BEEBS shared script + 44 per-benchmark scripts and rv8 aes/primes (46 files), then its whole-file guard against "a backslash-continued line followed by a comment" tripped on a PRE-EXISTING pattern elsewhere in `build-sqlite-capstone.sh` and aborted before that file and the core-init-state repro were edited; both were then edited by hand (the array element dropped; the repro's own-line `-fno-jump-tables \` dropped and the `-fno-jump-tables -O0 \` token stripped, note at the top, `bash -n` clean) BEFORE the chain's sqlite-memory step, and after the rv8/BEEBS twins had already picked up the 46 edits. Files still containing the string are comment-only (the cycle-3 retirements).

### B4 — session 2d (15:38–; rawhazard5 relinked, the SQLite arms)

| boot | rungs | readings | verdict |
|---|---|---|---|
| sw17 | k800=4; rawhazard5 relinked at 0x1e0000 (sha 4b7d0c7852ea7098), alone | `DEBUG rawhazard5 dbg0=5 dbg1=5 dbg2=5 dbg3=5 dbg4=5 dbg5..=0`, retval 48879 — identical to the QEMU reading of the same bytes; no VA collision this time (C15 GO) | all three R-1 probes now read as the reference model on 5097eb166 → an R-1 status change (GONE) is a separate line naming all three, proposed in batch 4 |

Q-03, second audit (15:47, dev 8a4628d827dd): the landed tail drop is LATENT, not a live hazard — after every REGION_CREATE the module's count equals the monitor's and its own region is the tail, so the drop only removes a region that is either invisible to the module or the module's own, which no image can exact-fit (fresh pages, never freed). "Likely on the board firmware too" is withdrawn on the same grounds. Three facts for citing the Q-03 batches: the drop is silent on both platforms (so "it never fired" is unsupported anywhere); manifests B/F could not exercise a module hazard at all (`capstone-test.user` has `create_region` commented out, `userspace/capstone-test.c:34`, so the module's table was empty for all 24 items); the module's only desync detector ("Region ID reuse detected") has never been shown to fire. Fix plan: `docs/plans/q03-region-hole-sentinel.md` (hole + `region_live[]`, index == id preserved, holes never reused, tail special case removed, a 0x1236 print per hole; predicted reading manifest B 24/24 with exactly three 0x1236 prints at items 8/12/22). Two module defects filed from the audit: M-2 (`probe_regions` copies a 96-entry board pool into a 64-entry array, unbounded), M-3 (every Capstone ecall returns error = 0, the module's failure paths are dead code).| sw14 | k800=4; sqm1 (-O0 SLT domain, memcpy optnone ON = silicon default, sha 1ff3686fe7763f48) on q_two.test | `SQ: G/enter`, `SQ: H/return`, `SLT-SUMMARY records=2 stmt_pass=1 stmt_fail=0 query_pass=1 query_fail=0 oom=0 completed=1` | the first -O0 SQLite domain run on the board today passes (the SLT board arms so far were -O1/-O2). NOT an S-04 arm: at -O0 the attribute changes nothing (see the retraction under sw15) || sw15 | k800=4; sqm0 (-O0 SLT domain, memcpy optnone OFF, sha 0a0489454fd63371 — the pair differs from sqm1) on q_two.test | `SQ: G/enter`, `SQ: H/return`, `SLT-SUMMARY records=2 stmt_pass=1 query_pass=1 oom=0 completed=1` — `sqlite3_open` succeeded, no SQLITE_NOMEM | **RETRACTED 16:45 (board lane's check): this pair does NOT test S-04.** S-04 blames the -O1 memcpy form (`sm0`, seven `sb` from the -O1 tail loop that do not stick); at -O0 the optnone attribute changes nothing, so both arms carried the WORKING form and were expected to pass on any bitstream. Correct line: -O0 SQLite domains pass on 5097eb166 with and without the attribute (expected); the blamed -O1 form with BEEBS_MEMCPY_OPTNONE=0 was not run — built and booted next as the real arm |

Pin retirement, twins so far (16:05): RV8 -O0 7/7, RV8 -O2 7/7; BEEBS -O0 80/81 — the one failure, `aha-mont64`, is attributed BEFORE the rerun: its image is byte-identical to the pre-retirement one that passed at 14:15 (sha 23a1c502862f, `jr` count 0: no jump table in it), and the failure is `QEMU stopped before the shell prompt` on the loader's `cp` (a pexpect timeout at the guest shell — the boot-login infra-flake class the harness retries elsewhere), so it is not the pin's; two solo reruns are queued behind the chain as the positive confirmation. BEEBS -O2 running.| sw16 | k800 + sqshr (-O1 SLT, shrink ON, 3533 SHRINKs, sha 227063c4626fac5a) | **VOID BOOT**: the driver's early-halt control timed out and the shell never answered ("core left halted by the early halt control"), aborted before k800 or any domain ran — infra, no verdict | rerun queued as sw19 (same bake-per-arm); sqbase (-O1 OFF twin, 0 SHRINKs) runs as sw18 in between, so the pair exists either way || sw18 | k800=4; sqbase (-O1 SLT, silicon config, 0 SHRINKs, compiler 50818be7cbe6, sha 680ebe68987badce) on q_two.test | `SQ: G/enter`, `SQ: H/return`, `SLT-SUMMARY records=2 stmt_pass=1 query_pass=1 completed=1` | the -O1 SQLite silicon domain built by the COMMITTED compiler (F-02/F-03 fix + S-12 removal) passes on s12fix — the OFF twin for sqshr, and a silicon datapoint for the compiler change beyond the ladder rungs || sw19 | k800=4; sqshr rerun (-O1 SLT, shrink ON, 3533 SHRINKs, sha 227063c4626fac5a) on q_two.test | `SQ: G/enter`, `SQ: H/return`, `SLT-SUMMARY records=2 stmt_pass=1 query_pass=1 completed=1`; its OFF twin sqbase (0 SHRINKs, same compiler) passed in sw18 | **W-06/W-07 not silicon debt on the SQLite -O1 shape either** (two shapes now: beebs_crc32 -O0 with 32 SHRINKs, SQLite -O1 with 3533). Moving the silicon-config default is the lead's call — ready to decide |

### B2 — pin retirement validated (15:33–16:28, compiler 50818be7cbe6)

RV8 -O0 7/7, RV8 -O2 7/7, gate 7/7 AGREE-PASS. BEEBS -O0 80/81, BEEBS -O2 81/81, gate 80/81 (the one O0-only failure is `aha-mont64`, below). sqlite-memory row (default ABI, now with jump tables) PASS, both markers. **Attribution against the prediction:** -O0: 11 images changed (cover, dtoa, duff, lcdnum, mergesort, miniz, picojpeg, qrduino, trio-snprintf, trio-sscanf, wikisort), every one now carrying ≥ 1 `jr` (a table); 70 unchanged. -O2: 12 changed (dtoa, duff, lcdnum, levenshtein, mergesort, miniz, picojpeg, qrduino, slre, trio-snprintf, trio-sscanf, wikisort); nine carry a `jr`, three (dtoa, lcdnum, levenshtein) changed WITHOUT an indirect jump — expected: the `no-jump-tables` function attribute also gates SimplifyCFG's switch-to-lookup-table transform, so at -O2 those switches became data lookup tables (no `jr`); all twelve pass. The prediction ("only switch-carrying benchmarks change") holds with that refinement; 69/70 unchanged images are exactly the ones without a dense switch. **aha-mont64 -O0**: image byte-identical to the pre-retirement one (23a1c502862f, `jr` 0) — the retirement cannot be its cause; its failure was `QEMU stopped before the shell prompt` on the loader's `cp` (a guest-shell stall). My first two solo reruns were VOID by harness misuse (an `OUT_DIR` override left the host binary out of the share dir: `cp: can't stat /mnt/host/beebs_aha-mont64_host.user`); proper solo reruns with the runner's defaults are queued behind the board lane's lock window.

aha-mont64 -O0, solo reruns with the runner's defaults (16:30–16:35, behind the board lane's lock window): `__BEEBS_AHA_MONT64_PASSED__` twice, image 23a1c502862f (the same bytes). The one BEEBS -O0 failure in the pin-retirement twin was a guest-shell stall, not the pin: **BEEBS -O0 81/81 with the pins retired**, and the -O0/-O2 gate's single O0-only entry is closed by these reruns.

R-1 status-change evidence (16:40, board lane's requirement: the probes were rebuilt with today's compiler and the July images have no recorded sha, so the hazard shape is read off the new bytes): rawhazard5 (0x1e0000 image) arm C, the positive control (computed index `[j-1]` plus an extra store): the loop stores `sw a0, 0x0(a1)` at 0x1e027c through `a1 = cincoffset a1, a0, a1` of the array capability (`ldc a0, 0x0(gp)`), then the loop head re-derives the same address into a different register — `ldc a0, 0x0(gp); cincoffset a0, a0, a1; lw a0, 0x0(a0)` at 0x1e021c–0x1e0224 — and reloads the bytes just stored, with only the counter's `ld`/`sd` through s0 between. The store-through-one-register / load-through-another shape is intact at -O0 in all three images — rawhazard6 has the same computed-index structure (13 `cincoffset aN, aN, aN` index derivations, 17 `ldc … (gp)` re-derivations feeding its `lw`/`sw`); rawhazard7 is the literal-index probe, so its stores and reloads go through registers freshly loaded by `ldc … (gp)` with immediate offsets (7 `ldc (gp)`, only 2 `cincoffset` triples), which is the same different-register reload of the same bytes without the index arithmetic; corrected 17:25 from an earlier "same structure" that had only been checked on rawhazard6. The probes read 5 = correct on 5097eb166. R-1 → GONE, attribution (forwarding fix vs S-12 change) left open.| sw20 | k800 + sqm0o1 (-O1 SLT, SQLITE_MEMCPY_OPTNONE=0 — the memcpy form S-04 blames; sha b4b453f90cd7edb4, differs from sqbase; q_two passes under QEMU as S-04 always did) | **VOID BOOT (infra)**: the console server refused the socket.io reconnects and the GDB action timed out before any domain ran (24 min of retries) | rerun as sw21 on the same bake || sw21 | rerun of sw20 | preflight GO, then socket.io ConnectionError on the first connect: the console HOST does not resolve from this machine since ~16:45 (board lane confirms: getent empty, HTTPS probes time out — DNS/tunnel, not the board; no server-side GDB session to clear) | the S-04 arm's boot is PENDING on the console; a five-minute probe loop (`retry-s04.sh`, up to an hour) boots it as sw22 if the console returns; the image (sqm0o1) stays baked in the board overlay |

Q-03 closed by the board lane (17:10): the hole + `region_live[]` sentinel monitor (fw_jump ca218db97d53 / sbi.dom ba3a8dca38e3) is what every QEMU suite boots from now on; manifest B 24/24 twice with an identical hole sequence (items 8 → i=8/n=10, then 9/11, 11/13), F 6/6, the five REV_TRANSFERRED probes and the module-consistency loader green; Q-05 filed for a QEMU-fidelity question found on the way (a host observer reading transferred pages through QEMU's tagged duplicate). Capstone-C does not short-circuit `||`/`&&` (recorded in HOW-TO-RUN-ON-QEMU.md).

### Close-out (17:25)

- Branch head 668edd125d4a; dev fast-forwarded to it locally by the board lane (content gate CLEAN) and pushed when github.com resolves again (DNS on this machine failed at ~16:45 together with the FPGA console host).
- The `retire-pins.py` guard that aborted the script fired on a PRE-EXISTING pattern in `build-sqlite-capstone.sh:52`: a `\`-continued `sed -e` expression whose next line starts with `#define` inside the single-quoted replacement text — not a shell comment, harmless; the two skipped files were edited by hand and `bash -n` clean; the core-init-state repro was edited and syntax-checked but not run.
- Pending, infrastructure-blocked: the S-04 -O1 arm boot (image `sqm0o1` b4b453f90cd7edb4 baked in the board overlay; `retry-s04.sh` probes the console until ~18:08 and boots it as sw22 if the host resolves — any such reading is unreviewed and needs a batch-5 line or a rerun). **18:05: handed to the board lane** (every board run goes through it; the probe loop was stopped before it could fire) — sw22 is the first boot of its phase A and it files the batch-5 line. **19:21: sw22 LANDED (board lane; verified against the raw transcript `/tmp/capstone/slt-campaign/s04o1-raw.txt` lines 27/96/100/101).** Bitstream caplifive_s12fix_5097eb166, fw ec0eb0db2852 packing only k800/lpc/q_two/sqlite_host.user/sqm0o1: control `k800` retval=4 cycles=4435 OK; `sqm0o1` (-O1, SQLITE_MEMCPY_OPTNONE=0, b4b453f90cd7edb4, host a1895d35f768b5d0) `--slt q_two.test`: A/dom-ok … G/enter, H/return, `SLT-SUMMARY records=2 stmt_pass=1 stmt_fail=0 query_pass=1 query_fail=0 oom=0 completed=1`, `M5 oom=0`. No SQLITE_NOMEM at open → **S-04 NOT REPRODUCED on 5097eb166 with the -O1 memcpy form it blames, N=1** (header stays WORKED AROUND; the silicon memcpy default is the lead's decision). Batch-5 registry rows (board-results TSV, S-04 dated block, W-09 board column "not silicon debt on this shape (N=1)") are the board lane's, in its results commit. The boot survived a tailscaled restart mid-run; the earlier sw20/sw21 voids were the relay data path, not the console service.
- **C-44 (filed by the board lane 18:00 as "the retired SQLite `-fno-jump-tables` pin was load-bearing") — REFUTED as a compiler defect 18:20; it is a STALE BINARY.** The faulting SLT image 772ef8e98dd71b0b dispatches its 26 jump tables with `lui a0, 0x161; addi; add; lw a0, 0(a0); jr a0` — the generic BR_JT expansion (integer base) that d99b68d0c80a replaced; today's compiler never emits it (under `-capstone-gp-captable` it refuses jump tables at -O0/-O1/-O2 — `jump-table.ll` jr=0 — and without the flag it forms the table base from gp). The main checkout's `llvm/cmake-build-debug` was built 2026-09-04 20:18, before dev was fast-forwarded: its `llc` does not know `-capstone-gp-captable-jump-tables`, emits a jump table under gp-captable, and `ninja -n llc clang lld` there has 2158 steps pending. Same SLT config rebuilt with today's compiler, pin retired vs pinned via `DOMAIN_EXTRA_DEFS`: byte-identical (1ff3686fe7763f48, jr=0); select1 under QEMU on it: `records=1031 stmt_pass=31 stmt_fail=0 query_pass=1000 query_fail=0 completed=1` = native. A script change and a compiler change that land together are consistent only together. Also learned: `jr` count is not a jump-table detector on new-compiler -O1 images — sqbase/sqm0o1 carry 438 `jr`, all `jr t1` tail calls, zero table forms. Gate added: `capstone/tests/toolchain-fresh.py` (ninja dry-run for `llc clang lld`; STALE iff any pending compile or tablegen step lies outside the VCS-revision chain that every commit legitimately re-triggers; "cannot check" = rc 2; prints the CodeGen `.so` hash+mtime and clang's embedded revision as the identity for result rows), called once per process tree from `capstone-test-env.sh` as a warning, fatal with `CAPSTONE_REQUIRE_FRESH_TOOLCHAIN=1`. Controls: the main checkout's build dir read STALE (2158 steps, `.td`/`.cpp` roots) before the board lane rebuilt it at 18:29 and fresh after; a freshly rebuilt worktree reads fresh; a touched Capstone source reads STALE naming that object; a bogus dir is rc 2. Two definitions were rejected on the way: "any pending step" (a fresh tree shows the revision-chain relinks after every commit) and "a source newer than its object, generated files excluded" (blind mid-rebuild, when the roots are already regenerated `.inc` files). The board lane's SLT campaign images (c01e6b89, 7451fe06) are the 09-04 binary's output and their board rows must carry that identity; the retirement stands.
- Deferred, not dropped: R-15 stages and R-17/S-01 uc/dp0 arms (stage machinery, session 3); M-1's faulting rung (fix known, unverified); C-9; C-17's >64-bit constant arm decision.
- Corrections made this session, all recorded where they happened: C-19's simulation test misread (caught before publication); "retraction" misapplied to R-25 in a message; rawhazard5 staged at the control's VA (C15 now gates it); the R-18 oracle files in unpacked form and the accumulator-probe oracle swap (annotated in the rows); R-1 first written GONE on two probes; the S-04 "not reproduced" line that stood on dev for nine minutes (b5aa78aff7bb); the first aha-mont64 reruns voided by an OUT_DIR override; the rawhazard6/7 "same structure" sentence above.


### Rows for the IDs that had no instrument this sweep (19:45) — closing the table's own criterion

The plan's verification asks for a row per inventory ID. A count against the registry's headings at 19:35 found 33 without one: the record-only, reserved, hand-off and no-instrument items, none of which was run. Each gets its row below with a verdict no stronger than what the record supports. "NOT RUN" means exactly that; none of these is a GONE.

| id | registry | old status | instrument | verdict | evidence | action |
|---|---|---|---|---|---|---|
| C-3 | ISSUES | OPEN | RV8 -O2 twins (default ABI) under QEMU; the board rv8 line | NOT REPRODUCED under QEMU at -O2 (RV8 7/7 ×2 in the pin-retirement validation and the W-17 pair); board line NOT RUN | B2 rows above (W-17 rv8 7/7; retirement RV8 7/7 ×2) | keep OPEN until a board rv8 -O2 line lands (board lane's campaign is the natural place) |
| C-5 | ISSUES | OPEN | none — a design limit (4 KiB code window), not a defect with a reproducer | UNTESTABLE as filed | — | keep as a documented limit |
| C-10 | ISSUES | CLOSED (refuted lead) | none needed | record | — | none |
| C-18 | ISSUES | FIXED 2026-08-11 | its lit pin | lit green at the close-out (Capstone 89 pass); pin NOT re-fired against a pre-fix compiler this sweep | lit run in the branch's validation | none |
| C-24, C-25 | ISSUES | RECORD ONLY | none | record | — | none |
| C-27, C-33, C-36 | ISSUES | FIXED 2026-09-05 (this branch) | their red-first lit tests at fix time | fixed in-branch; lit green at the close-out; not re-fired against the pre-fix compiler this sweep | branch commits + lit | none |
| C-29, C-30, C-35, C-42 | ISSUES | RESERVED | none | record (numbers reserved, nothing filed) | — | none |
| C-34 | ISSUES | CLOSED 2026-09-05 without a fix | none | record | — | none |
| C-37 | ISSUES | OPEN | `llvm-readelf -r` on any Capstone object | STILL PRESENT — 23 "Unknown" relocation names on `jt.o` today (ELF.cpp has no EM_CAPSTONE case) | this session's `jt.o` readelf | keep OPEN; small fix, queue after the sweep |
| C-38 | ISSUES | OPEN (naming decision) | none | UNTESTABLE — a decision, not a defect | — | lead's call |
| C-43 | ISSUES | OPEN, latent class | the cttz table fault (measured 2026-09-05) and areJTsAllowed under gp-captable | STILL PRESENT as a class; its two known instances are mitigated in-branch (cttz arithmetic form, jump tables refused) | `lowerCTTZNoTable`, `areJTsAllowed` | keep OPEN; class-level guard is a separate design item |
| I-01 | ISSUES | RESOLVED 2026-09-04 | negative-tested push gate | record | — | none |
| I-1, I-2 | ISSUES | FIXED (harness) | none this sweep | record | — | none |
| I-02 | ISSUES | OPEN (allocation convention) | none | UNTESTABLE — process, not code; the I-1..I-5 vs I-01..I-03 collision is noted in the registry | — | lead's call on a convention |
| I-03 | ISSUES | OPEN, latent on the board | none this sweep (needs a linker placement that lands the array misaligned) | NOT RUN | — | keep OPEN |
| I-5 | ISSUES | OPEN, cheap fix identified (monitor) | none — monitor | hand-off | — | monitor lane |
| M-4 | ISSUES | OPEN, filed 2026-09-05 (monitor bound check) | none — monitor | hand-off; verified file:line at filing | — | monitor lane |
| R-2 | ISSUES | EXPLAINED by C-13 | the C-13/C-14 compiler-side fix's acceptance on the board | covered: gpw2/gpn2 PASS this sweep (batch 4, session 1) | the C-14 rows above | none |
| R-4, R-5 | ISSUES | OPEN, no reproducer | none — the plan said "write one only if a sweep result points at them"; none did | UNTESTABLE | — | keep OPEN |
| R-7 | ISSUES | CLOSED into R-1 | R-1's three probes on the board this sweep | covered by R-1 GONE (rawhazard5/6/7) | the R-1 rows above | none |
| R-10 | ISSUES | ROOT CAUSE of C-13 (RTL), board-confirmed 07-29 | the compiler-side workaround's acceptance (C-14) | RTL side not re-tested; workaround acceptance PASS this sweep | the C-14 rows above | keep as the RTL root cause; RTL lane owns any RTL fix |
| R-11 | ISSUES | OPEN, not yet hit | `check-repr.py` on the current corpus | NOT RUN this sweep (planned in B1, not executed — no log exists) | — | keep OPEN; run it with the next corpus refresh |
| R-12 | ISSUES | OPEN, will bite at call_dom | a >1024-split counting rung | UNTESTABLE without new work; deferred (see close-out) | — | deferred, not dropped |
| R-13 | ISSUES | OPEN | R-21's `linear-clear-audit` arm 2 (a LINEAR cincoffset source) | GONE at 5097eb166 for the cincoffset case: the source reads NOT_CAP after cincoffset (finding 27); the INIT/TIGHTEN duplication stays under R-21 | the R-21 row above | fold into R-21; registry note is the board lane's |
| R-22 | ISSUES | OPEN, spec violation, not reported | a `stc`-clears-source arm (B3 planned it beside R-24) | NOT RUN — only `excode-base-audit` (R-24) was run | — | keep OPEN; RTL lane hand-off with R-24 |
