# `fpga-testing-dev` rebuilt: eight commits, no instrument, every fix line accounted for, every row measured

**Dates:** built 2026-09-05, measured 2026-09-07 (the first measurement pass was lost to the 5-6 September
host outage; see "What the measurement pass taught"). **Branch:** `fpga-testing-dev-clean` in `capstone-ariane`,
intended to replace `fpga-testing-dev` by a force-push the project lead performs. **Old tip preserved:** tag
`backup/fpga-testing-dev-2026-08-21` (= `e12a0e3e9`), plus `backup/<branch>-2026-09-05` tags on every one of
our branches and an all-refs bundle under `~/dev/llvm-capstone-rebuild/backups/`.

## What the audit found before anything was rebuilt

* `fpga-testing-dev` was the shared base `7e4dc440f` plus 7 curated commits from 2026-08-21; the flashed
  bitstream's branch `s12-fix-for-synthesis` (`5097eb166`) was the same base plus 41 raw commits. The two were
  **RTL-identical except S-10, S-12 and a testbench-only delay knob**; every line of the S-07 on-silicon
  instrument was in both.
* The instrument is unconditional and cost, by the arm-2 tie-off measurement, **1.82 ns WNS, ~1,900 failing
  endpoints, ~750 LUTs** on a design that fails timing on every endpoint.
* Four look-alike side-branch commits were content-identical to lineage commits; nothing unique lived only there.
* The five fix commits reused verbatim contained zero instrument identifiers; `data_rtag_src`, `rd_ctag_src` and
  `cap_wb_displaced` were all introduced by the instrument commit `6f1a597ba`.
* Every stored `rtl-lint.REF.txt` on every branch said `UNOPTFLAT 39` and was measured with the instrument present.
* `run.tcl`'s only change on the old branch was a 28-line comment carrying a wrong number.

## The rebuilt history (above `7e4dc440f`, which is untouched)

| # | commit | provenance | RTL identity check |
|---|---|---|---|
| 1 | lint gate + sweep script + baseline **measured at the base** | scripts from the flashed tip | gate `--update` at `7e4dc440f` |
| 2 | S-06 | cherry-pick `3673f5869` | identical |
| 3 | S-08 | cherry-pick `31bffa77d` | identical |
| 4 | S-07 fix + sim-only assertion + 8 tests | `1dd83bfac` + `813658ded` | write buffer identical modulo one probe tie-off |
| 5 | S-10 + the refill-path comment | `4fee13b2d` hunk, placed by script | 12/12 added code lines identical; probe paragraph rewritten |
| 6 | S-12 + tb knob + 3 tests | `git diff 80843404c 5097eb166` | `issue_read_operands.sv` md5-identical |
| 7 | mtval cursor + tval latch + switch-212 mirror | `fed5c55b7` + tval subset of `39111e119` | `ex_stage.sv` identical; 11 tval lines identical |
| 8 | synthesis tooling | flashed-tip versions | `corev_apu/fpga` byte-identical to upstream |

**Residual against `5097eb166`** (`core/` + `corev_apu/`): 12 files, the S-07 instrument and comment text and nothing
else; mechanically classified, then attacked by an adversarial audit that returned SUPPORTED. Its two corrections
were taken: the switch-212 trap-summary mirror (a live repro folder names it as the UART-safe readout; a build
without it answers `0x00` there, which decodes as "no trap") and the nine-line refill-path comment.

## Measurements (all on this branch, 2026-09-07; records committed with their commits)

* **Simulation identity.** All **88 rows** of the capstone testlist identical, trace hashes included, between the
  rebuilt tip and the flashed `5097eb166`, swept the same day with the same runner.
* **Determinism control.** The base tree's 67 rows identical to the pre-fix baseline measured on RTL-identical
  `013e162fd` on 2026-08-14. The runner was validated against three committed oracle rows first, including a test
  whose `tohost` is not at the default address.
* **Lint, per commit** (gate positive-controlled with a live injected loop, 40 -> 41, FAIL): base 39/713, S-06
  39/717, S-08 39/717, S-07 39/717, S-10 **40**/717, S-12 40/717, tip 40/717 (`UNOPTFLAT`/`UNUSEDSIGNAL`; every other
  counter constant). The baseline file is re-derived exactly twice, at S-06 and S-10.
* **Pre-fix controls on the parent trees** (test copied in, same runner):
  S-06 `s06-lowhalf-zero` FAIL 729 -> SUCCESS 731, `-swap` FAIL 732 -> SUCCESS 734, `s06sec-raw-alias-no-launder`
  FAIL 697 -> SUCCESS 723; S-08 `s06sec-ctx-scalar-roundtrip` FAIL 612 -> SUCCESS 592; S-07 `s07-wbuf-tag-reorder`
  4 -> 1 exceptions with its control pinned at 1; S-10 `s07-wbuf-forward-residual` 9 -> 17 with its control at 17;
  S-12 under `S12_MEM_DELAY=40`: pre-fix (S-10 RTL + the S-12 testbench knob) FAIL, 254 `Exception:` lines at 190,471 cycles (control 1), tip 1 exception at
  156,794 cycles. The 254 is 253 reproducer traps plus the ARM P control's trap at cycle 591, which is the tip's 1; the
  original commit b9dd83249 quoted the same runs as 255 in its table and 254 in its prose, so the trap count is
  convention-dependent and the cycle counts (190,471 / 156,794) are the exact match.
* **Per-commit deltas are all attributable:** S-06 changes six rows by a few cycles and one timeout trace; S-08
  changes the two domain-switch tests; S-07 changes no shared row; S-10 changes the residual pair and one test by
  three cycles; S-12 changes nothing at delay 0; the mtval commit changes exactly one trace hash
  (`s07-ldc-chain-forward`, the cursor now in tval).

## What the measurement pass taught, in the order it cost time

1. **Every docker container on this host shared an 8 GiB cgroup** (`system.slice`), while `free` showed 150 GB.
   Simulations were killed at random for two hours; the same load contributed to the host outage. Resolved by
   running the built model on the host and, since 2026-09-07, by the host's own `docker.slice`
   (`~/bin/logs/AGREED-RESOURCE-RULES.md`).
2. `cva6.py --steps gen` compiles a directed test AND simulates it; `--steps gcc_compile` does neither for
   directed tests. A compile step that returns success and produces no ELF reads as `NOBUILD` rows.
3. `+tohost_addr` is per ELF (`nm | grep -w tohost`); hardcoding the first test's value made seven tests fail at
   cycle 384. The determinism control caught it.
4. The pre-S-12 negative control needs the S-12 commit's testbench knob applied to the pre-S-12 RTL; without it the
   define is inert and the control silently reads delay 0.
5. A positive control for the lint gate must be a *live* loop; a dead one is pruned and the gate stays silent.

## Not on the branch, and where it lives

The S-07 recorders, selftest, gran_match, displacement detector and read-tag-source probe; the strip commit;
intermediate sweep records; everything on `s12-ldc-rolling-*`, `s07-recorder-clear*`, `timing-*`, `s10b-fix`
(unsynthesizable), `s12-fix-noinstr`, `s12-fix-variant-b`; the R-25 fix (its own branch, by decision).
`s12-fix-for-synthesis` and `s12-fix-noinstr` are frozen as the provenance of the two synthesised bitstreams.

## Status

The tip is **new RTL** (the instrument is gone) and is unsynthesised: a candidate, not ready. Pushed 2026-09-07 as
`fpga-testing-dev-clean`, tip `947327f6d` (`chain-v3` plus one corrected S-12 message sentence). Synthesised the same day;
see the addendum at the end. Any board work goes through the board lane. The force-push to `fpga-testing-dev` is the project lead's action.

## Addendum, 2026-09-07 (later the same day): synthesised, censused, NOT usable

One run on the synthesis machine, the arms' container and guard, default flow, ceiling 100 GB:
exit 0, 1h43m22s, synthesis peak 21.1 GB, collector 34.2 GB, load 39/80 at launch.

| build | WNS (ns) | failing / total (CPU clock) | placed LUTs | launch census |
|---|---:|---:|---:|---|
| `947327f6d` this | −11.717 | 97,438 / 174,756 | 170,481 (83.65%) | 97,438 `dom_switcher/req_en_q`, 1 DDR |
| `6f8345fdb` arm 2, tie-off | −13.491 | 99,879 / 173,789 | 168,944 | 99,879 `issue_read_operands` |
| `5097eb166` arm 1, flashed | −15.311 | 101,782 / 174,895 | 169,694 | `dom_switcher` (`_thread_0_event_reg_87`, `_init_0`) |

Best-timed fix-carrying build on record; census verified on both axes; DRC and loop signature
identical to the arms. The pre-registered expectation ("LUTs and WNS between the arms, near arm 2")
missed in one direction: larger than both, better-timed than both.

**Two corrections to earlier wording.** "Instrument-free" was wrong: the S-07 layer is gone, but
upstream's UART instruction tracer (`core/tracer.sv`, synthesised unconditionally since it was added
upstream; 65,562 of the failing endpoints end inside it) and the base's 12 debug-bank apertures
remain. And "inside flow variance" for the LUT differences was an assumption: no commit has ever
been built twice with identical settings on this flow.

**Verdict: NOT usable as a board bitstream**, on the census principle as it was meant rather than as it
was written. The launch register `req_en_q` is the domain-switch busy level; it is constant during a
body, but its rising edge is the very signal that gates commit (`commit_stage.sv:303`, commit
`030378a66`'s fix) and the trap-entry CSRs, `instret`, the scoreboard and the register-file module are
all on its failing cone. Experiment B, run the same afternoon in Verilator, measured both edges: with the commit stage seeing busy
one cycle late, the instruction after a CALL commits (2 of the 2 informative CALL tests; FAIL and TIMEOUT);
with the frontend seeing it late, the callee retires with PCs shifted by +4 (the same two tests, TIMEOUT);
the controller arm is clean. Full argument, the auditor's refutation, the experiment and the things that
would change the verdict are in `ref/bitstream-usability-is-the-census-not-the-slack.md`, 2026-09-07 entry.

**Consequence for the branch.** `fpga-testing-dev-clean` is the correct RTL history and stays; the
force-push over `fpga-testing-dev` is unaffected by the timing result. What it is not, yet, is a
bitstream: flashing it would need the commit-gate change (or an equivalent) and a re-synthesis first.
The resident `caplifive_s12fix_5097eb166.bit` remains the licensed bitstream.

## Addendum 2, 2026-09-07/08: the fix synthesised — pre-registered lines hold, still NOT usable

`ef5a8eaf2` (this branch's tip: `947327f6d` + `dom_switch_active_q` feeding the three consumers) went
through the same machine, container, guard and flow: exit 0, 3h09m under load 75–85, synthesis peak
20.96 GB, artifact `synth-ef5a8eaf2-exit0.tar.gz` (404,402,489 bytes).

| build | WNS (ns) | failing / total (CPU clock) | placed LUTs | launch census |
|---|---:|---:|---:|---|
| `ef5a8eaf2` fix | −12.733 | 101,143 / 174,188 | 170,410 | 101,143 / 101,143 `issue_read_operands` (`lsu_valid_q_reg[0]_rep`) |
| `947327f6d` base | −11.717 | 97,438 / 174,756 | 170,481 | 97,438 `dom_switcher/req_en_q`, 1 DDR |

Pre-registered reading (census doc): (1) launches from `dom_switch_active_q` **0**; (2) the flag's D input
not a failing endpoint; (3) `req_en_q` launches **nothing** that fails. All three hold: the busy-edge hazard
measured in Experiment B has no failing path on this build. Verdict nevertheless **NOT usable**: the worst
launch is now the issue-to-LSU valid, live on every memory instruction, and the endpoint population is the
base build's (tracer 65,562 identical; issue 21,731; ex 8,415; csr 978). Census verified on both axes (101,143 = 101,143);
collector peak 37.9 GB. What that says about the worst-launch
census, what is measured and what is only inferred, the per-checkpoint query that settles it and the
options for the lead are in the census doc's 2026-09-07/08 entry. Board side: nothing to flash; the resident
`5097eb166` stays. **Same night, retracted:** the per-checkpoint query found the resident's own failing endpoints
also failing from a live register (LSU bypass occupancy counter, 101,604 of 101,784, −15.157). The census gate
never separated the flashed build from the ones it rejected; the resident stays on its board record, not on
the census (census doc, RETRACTED 2026-09-08 section).

### Appendix: Experiment B result rows (verdict, cycles, RVFI hash, exceptions)

Tip = the identical `core/` tree's record. Arms delay `dom_switch_busy` by one cycle at C = `commit_stage_i`,
K = `controller_i`, F = `i_frontend`. A row is shown once when all four agree.

```
capenter                       all four: SUCCESS 436 47d4c257d4aeed72 0
call-ctx-save                  all four: TIMEOUT 400013 fded058d33bdc5a1 1
data-sharing                   all four: TIMEOUT 400013 f82dae083835b7ce 1
data-transfer                  all four: SUCCESS 421 b7f8e7813273cfd8 0
csd                            all four: SUCCESS 2090 c650ea0e663dfeeb 0
lcc                            all four: SUCCESS 362 4a6cfffc462c3b0c 0
jalr                           all four: SUCCESS 570 0033c8a0579e5616 0
interrupt                      all four: SUCCESS 560 1cef1be4647cd099 0
interrupt-rv                   all four: SUCCESS 401 08649d9763ec0c82 0
revocation                     tip : SUCCESS 932 f2b5adfeeb07a7f3 0
                               C   : TIMEOUT 400013 9761126089477a06 1   <- DIFF
                               K   : SUCCESS 932 f2b5adfeeb07a7f3 0
                               F   : TIMEOUT 400013 436d581af8709a17 0   <- DIFF
s06sec-ctx-scalar-roundtrip    tip : SUCCESS 592 5d1038b0a5aba41a 0
                               C   : FAIL 573 22f2bd7d7b2afa03 0   <- DIFF
                               K   : SUCCESS 592 5d1038b0a5aba41a 0
                               F   : TIMEOUT 400013 516a20fd751263fa 0   <- DIFF
r20-stc-ld-x10                 all four: SUCCESS 775 406fc4e491c7d5fc 0
ccsrrw                         all four: TIMEOUT 400013 f0332621c5173654 1
cpmp-if-check                  all four: SUCCESS 542 0a91a18774881413 0
cpmp-su-mode                   all four: TIMEOUT 400013 7dea5b550a393f38 0
scalar-store-movc-zero         all four: SUCCESS 415 20f2994b3413b183 0
scalar-store-addi-zero         all four: SUCCESS 12816 47d2b28c838f1f45 0
s07-ldc-chain-forward          all four: SUCCESS 276242 90a93c9151cb5463 1
```

The arm C patch (arms K and F differ only in which instance's port is rewired):

```diff
diff --git a/core/cva6.sv b/core/cva6.sv
index 2e6632884..563f6459c 100644
--- a/core/cva6.sv
+++ b/core/cva6.sv
@@ -808,6 +808,11 @@ module cva6
   logic dom_switch_commit_ack;
   dom_switch_req_t commit_dom_switch_req;
   logic dom_switch_busy;
+  // EXPERIMENT B (2026-09-07): dom_switch_busy captured one cycle late at ONE consumer (commit_stage_i).
+  logic dom_switch_busy_dly_q;
+  always_ff @(posedge clk_i or negedge rst_ni) begin
+    if (!rst_ni) dom_switch_busy_dly_q <= 1'b0; else dom_switch_busy_dly_q <= dom_switch_busy;
+  end
 
   // data req
   logic dom_switch_data_valid;
@@ -1867,7 +1872,7 @@ module cva6
       .dom_switch_valid_o (commit_dom_switch_valid),
       .dom_switch_req_o (commit_dom_switch_req),
       .dom_switch_ack_i (dom_switch_commit_ack),
-      .dom_switch_busy_i (dom_switch_busy),
+      .dom_switch_busy_i (dom_switch_busy_dly_q),
       // Capstone end
       .halt_i            (halt_ctrl),
       .flush_dcache_i    (dcache_flush_ctrl_cache),
```
