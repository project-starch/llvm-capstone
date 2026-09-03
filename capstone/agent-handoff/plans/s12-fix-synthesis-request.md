# S-12 fix: synthesis request, with the reading fixed in advance

**Status: request. Synthesis and flash are the board owner's step — Vivado is not on the lane
machine. Nothing here should be built until the integration note below is acted on.**

## What to build, and NOT from where it sits

The fix is two files on `capstone-ariane` branch `s12-ldc-rolling-filter`, commits `6fae48465`
(the fix) and `6e8a5aa17` (a comment recording the cone risk):

    core/issue_read_operands.sv    + commit_ack_i port, + the ack term on both WAW-clearing
                                     clauses, + a sim-only probe (translate_off)
    core/issue_stage.sv            + one pass-down of commit_ack_i, already a port there

**Do not synthesise that branch as it stands.** It does NOT contain `80843404c`, the commit the
resident `caplifive_s10fix_80843404c.bit` was built from — 9 commits are in that line and absent
here, including the S-10 fix. Building this branch would silently regress them.

The two files the fix touches are NOT touched by those 9 commits (the only `core/` overlap between
the lines is `wt_dcache_mem.sv`), so the integration is a clean apply rather than a merge: put the
two-file change on top of the bitstream's base, then synthesise that.

## The reading, fixed BEFORE the build

Per the project's own rule, a build does not go unless its predicted reading is informative.

**If the mechanism is right and the fix works**, the SQLite silicon domain that currently traps
completes instead. Concretely, running the same arm used for the a4 measurement:

* `obs` is an ordinary result word (top nibble NOT `0xF` or `0xE`) — on the trap-on build that
  same arm returned `obs=0xE643D221`, and the marker nibble is exactly what distinguishes them.
* No `mcause 25` at `_start+0xF4874` / `+0xF4884`.
* Across >= 4 draws behind a known-good control, with the per-draw image sha verified in-boot.

**If the fix does not work**, the same trap word comes back and that is equally informative: the
mechanism reproduces in simulation and is confirmed on silicon, so a fix that removes it in
simulation and not on hardware would mean the silicon path differs from the RTL we simulated —
which is a finding, not a null.

**This is a go.** Both branches of the prediction change what we do next, which is the test the
rule actually applies.

## Sufficiency — what else must ride along

Nothing found. S-06 is already in the resident bitstream (`80843404c` is contained in
`fpga-testing-dev-s06fix`), so this is not a batch of three despite what `current-state.md` still
implies for S-06/S-08. If any other RTL change is pending when this is built, it must go in the
same bitstream; a second cycle costs another ~90 minutes plus a reflash.

## Evidence travelling with the hash

| gate | result |
|---|---|
| reproducer `stc-ldc-sbpressure` @ `S12_MEM_DELAY=40` | 254 in-loop traps -> **0** (only ARM P remains) |
| `hazard` / `duplicate-live-rd` | 254 -> **0**, 64285 cyc -> **0** |
| matched control `-norel` | unchanged at 0 |
| capstone regression suite (zero-latency build) | base 72/13/3 vs fix 72/13/3 on THIS base, non-passing sets identical — but see the caveat below, this is OUT OF REGIME |
| `rtl-lint-gate.sh` | **FAILS — and fails IDENTICALLY on the unmodified base**, reporting `REGRESSION: UNOPTFLAT 39 -> 40`. The 39 is the gate's stored baseline, from a different lineage; this base genuinely has 40, consistent with `4fee13b2d` ("S-10 FIX ... costs a combinational loop") being in its history. Compared as SIGNAL SETS rather than counts, base and fix are **byte-identical**: zero loops added, zero removed. Tell the synthesis lane this explicitly or they will attribute the 40th loop to the fix and refuse the hash. The baseline file is deliberately NOT rebaselined — weakening a gate to make it pass is not this change's business. |
| synthesis | **NOT RUN — this request** |

**Numbers previously quoted here were from the WRONG TREE and have been removed.** A
`pass=99 fail=15 err=3` figure over 117 tests came from the main tree at `6e8a5aa17`, which differs
from this base by 427 insertions across 9 core files; this base's testlist has 88 entries and does
not even contain `s12min-noburst`. Nothing measured on that tree may be cited as evidence about
this hash.

**The regression above is OUT OF REGIME and must not be read as "no regressions".** The harness
passes no `isscomp_opts`, so it builds with `S12_MEM_DELAY_VAL 0`, and the proof that this matters
is in the artifacts themselves: the same base build shows `PASS stc-ldc-sbpressure` in the
regression and 255 traps in the reproducer. So that suite is a check which provably cannot fire on
this defect, run in the one regime where the fix is close to a no-op. It excludes gross breakage —
72 rows pass, and a wedged issue stage would time out everything — and little else. Sharpening the
concern: all 13 base failures are timeouts at exactly 2,000,013 cycles, and a stall-adding change
under memory pressure is exactly what would push further rows over that cliff.

An IN-REGIME comparison (the same 88 tests, both arms, built with `+define+S12_MEM_DELAY=40`, and
recording each row's cycle count so proximity to the timeout cliff is visible) is the control that
settles this, and it is the gating item before this hash ships.

The out-of-regime baseline was taken in this same worktree. Worth knowing: the first attempt at a
baseline, in a different worktree, was VOID — it lacked the gitignored `core/anvil.Flist`, so
nothing elaborated and every test "failed" identically to the fixed run, a false confirmation
shaped exactly like the wanted answer.

## Is it a FIX, or a timing perturbation?

This bug is documented as perturbation-sensitive, so "the reproducer stopped firing" is not on its
own an argument. Two things answer it, and the STRUCTURAL one should lead:

**Structural.** The config has `NrIssuePorts = 1` and issue is in-order. Blocking the `ldc` blocks
the consumer behind it; and by the cycle the `ldc` does issue, the store's entry has left
forwarding candidacy, so it cannot be the consumer's source. The fix removes the mechanism, not
merely the timing that exposes it.

**Empirical, and honestly weak on its own.** 255 traps -> 1 is ONE configuration measured once per
arm: 256 identical iterations of one shape in one deterministic simulation is not N=255, and a
systematic timing shift would produce exactly the same flip. The cheap control is a DELAY SWEEP —
run the reproducer on both arms at several `S12_MEM_DELAY` values (10 / 20 / 40 / 80). A real fix
gives: base traps across the range, fix traps nowhere. A knife-edge timing coincidence gives a
base that only traps near 40, or a fix that starts trapping again at some other delay.

A sham-signal control was considered and rejected as dishonest: the obvious candidate,
`commit_lsu_ready_i`, is the stall condition itself, so gating on it blocks issue for the same
causal reason and proves nothing. The delay sweep tests the same worry without that confound.

## The risk this build exists to settle

`issue_read_operands.sv` sits INSIDE a standing combinational-loop cone: the UNOPTFLAT at
`scoreboard.sv:129` (`issue_pointer`) has an example path through `scoreboard.sv:167` ->
`issue_instr_o` -> `issue_read_operands.sv:645` -> `rs1_fwd_req` -> `rr_arb_tree`. The stall block
this fix edits reads `issue_instr_i` and drives `issue_ack_o`, which feeds back into that same
`issue_pointer` — and the term added, `commit_ack_i`, crosses a module boundary into it.

That is the highest-risk edit shape on this core, and **every check available here is blind to
it**: the lint gate compares warning COUNTS, and cone membership appears only in "Example path"
continuation lines, not in the file a warning is raised at. A per-file check returns a FALSE CLEAR,
which is what it did here before the raw log was read properly.

It is worse than the generic case: the resident line already carries `4fee13b2d`, "S-10 FIX: works
in simulation, and costs a combinational loop". A new term entering a cone on RTL that already has
one is precisely the interaction nothing in simulation models.

No new loop is expected on the face of it — `commit_ack_o` depends on the commit port and
`commit_lsu_ready_i`, neither of which depends on `issue_ack` — but "not expected" is what
synthesis exists to decide. If `synth_design` runs long or blows up memory, that is the answer,
and the fallback is the codegen mitigation in `s12-codegen-mitigation-proposal.md`.

## What the synthesis lane must REPORT

Not "synth_design completed". Specifically:

* **WNS, against the S-10 build as the reference**, not against an abstract target. The resident
  bitstream exists *because of* a timing fix, so the margin it left is the number that matters.
* **Any "found timing loop" critical warnings.** `commit_ack_o[0]` is a deep signal — store-buffer
  ready, `no_st_pending`, `dom_switch_ack_i` out of an anvil unit whose internal loops are
  invisible to lint (every `.anvil.sv` opens with `lint_off UNOPTFLAT`), `amo_resp_i.ack` — and it
  now feeds `issue_ack_o` for the first time.
* **Runtime of `synth_design`.** The flow finishes in 41-46 minutes when healthy; a build that runs
  far past that is telling you something before it finishes.

## Do not change the flow

Leave `run.tcl` alone, `RETIMING` included. A lane once turned retiming off for a debug build and
synthesis passed two hours without leaving `synth_design` where the same flow finishes in 41-46
minutes.
