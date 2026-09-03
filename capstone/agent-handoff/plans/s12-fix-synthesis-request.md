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
| capstone regression suite | pass=99 fail=15 err=3; **all 18 non-passing baselined against pre-fix RTL and IDENTICAL** — zero regressions |
| `rtl-lint-gate.sh` | PASS. LATCH 52, MULTIDRIVEN 3, ALWCOMBORDER 0, COMBDLY 0, UNOPTFLAT 39, BLKSEQ 2, UNDRIVEN 25, UNUSEDSIGNAL 719, ANVIL_UNOPTFLAT 0 |
| synthesis | **NOT RUN — this request** |

The baseline was taken in a git worktree at the pre-fix commit, and it is worth knowing that the
first attempt at it was VOID: the worktree lacked the gitignored `core/anvil.Flist`, so nothing
elaborated and all 11 tests "failed" identically to the fixed run — a false confirmation that
looked exactly like the answer wanted. The numbers above are from the re-run, where the model is
confirmed built and the failure modes match line for line.

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

## Do not change the flow

Leave `run.tcl` alone, `RETIMING` included. A lane once turned retiming off for a debug build and
synthesis passed two hours without leaving `synth_design` where the same flow finishes in 41-46
minutes.
