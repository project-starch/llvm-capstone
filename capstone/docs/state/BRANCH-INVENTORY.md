# Branch inventory — `capstone-ariane`

> Rewritten 2026-09-08 after the branch clean-up. The 2026-08-20 inventory that used to head this file is kept
> below as history; every branch it lists is now an archive tag.

## Current branches of ours (5)

| branch | tip | role |
|---|---|---|
| `fpga-testing-dev` | `66c4e7517` (2026-09-09: `ef5a8eaf2` + R-26 `9d8797560`, R-25 `42a141c93`, R-27 `66c4e7517`; pushed by the RTL lane under the lead's go-ahead; in synthesis) — previously `ef5a8eaf2` | the canonical line: rebuilt history above `7e4dc440f` (eight commits, no S-07 on-silicon instrument) + the registered switch-in-progress flag. Synthesised, censused, **NOT usable as a bitstream**; the census itself is retracted as a licence (`ref/bitstream-usability-is-the-census-not-the-slack.md`, RETRACTED 2026-09-08). |
| `s12-fix-for-synthesis` | `5097eb166` | provenance of the resident bitstream `caplifive_s12fix_5097eb166.bit`; its board record is the only licence any build has. Frozen. |
| `s12-fix-noinstr` | `6f8345fdb` | provenance of the arm-2 (debug tree tied off) routed checkpoint used in the second-launch queries. Frozen. |
| `s12-ldc-rolling-filter` | `62b09ca92` | live RTL checkout: R-25 directed test and the LDC rolling-filter work. |
| `capstone-bootstrap` | `b860fda3f` | the superproject's recorded submodule line. |

Old tip of `fpga-testing-dev` (`e12a0e3e9`, seven commits that added and then stripped the instrument) is tag
`backup/fpga-testing-dev-2026-08-21`, on origin.

## 2026-09-09 fix-cycle side branches (local, development history; archive as tags once the bitstream is validated)

| branch | tip | local tag (the lead pushes tags) |
|---|---|---|
| `r26-ccsrrw-stale-read` | `67d870cc8` | `backup/r26-ccsrrw-stale-read-2026-09-09` |
| `r25-init-rs1-dup` | `ec50837b5` | `backup/r25-init-rs1-dup-2026-09-09` |
| `r27-revnode-orphan-drain` | `3bfaa544c` | `backup/r27-revnode-orphan-drain-2026-09-09` |
| `p3-final` | `66c4e7517` (= `fpga-testing-dev`) | `backup/p3-final-2026-09-09` |

The first commit of `r26-ccsrrw-stale-read` (`2e4dc369c`) is on origin; everything else in these branches is on origin
only through the three squashed commits on `fpga-testing-dev`.

## Archived 2026-09-08: branch deleted, tip kept as a `backup/*` tag

Every tag below points at the branch's last tip; each was checked to hold zero commits that exist nowhere else
before the branch was deleted. Tags live locally and on origin (pushed by the lead with the hook bypass).

| former branch | tag | what it was |
|---|---|---|
| `fpga-testing-dev-clean` | (= `fpga-testing-dev`, no tag needed; `947327f6d` is `chain-v4`) | the rebuild's working name |
| `r20-fix` | `backup/r20-fix-2026-09-05` | R-20 fix, since carried into the mainline |
| `fpga-testing-dev-s06` | `backup/fpga-testing-dev-s06-2026-09-05` (+ local tip in `backup/local-fpga-testing-dev-merged-backup-2026-09-05`) | S-06 phases |
| `fpga-testing-dev-s06fix` | `backup/fpga-testing-dev-s06fix-2026-09-05` | S-07 + S-10 mainline of August, `80843404c` built |
| `s07-recorder-clear`, `s07-recorder-clear-39b` | `backup/s07-recorder-clear-2026-09-05`, `backup/s07-recorder-clear-39b-2026-09-05` | S-07 LDC recorder instrument (never synthesised into the shipped line) |
| `s10-fix-wip`, `s10-merge-candidate`, `s10-narrow-mcp`, `s10b-fix`, `s10-candidate` | `backup/s10-*-2026-09-05`, `backup/local-s10-candidate-2026-09-05` | the S-10 line (see history below) |
| `timing-control-618f4ce36`, `timing-control-e1140aeea` | `backup/timing-control-*-2026-09-05` | the two timing controls; e1140aeea's was run (−10.629) |
| `timing-multicycle`, `timing-directive-explore` | `backup/timing-multicycle-2026-09-05`, `backup/timing-directive-explore-2026-09-05` | the two NON-RTL flow changes (multicycle constraint; Explore directives) — recover from the tags, see below |
| `s12-ldc-rolling-min` | `backup/s12-ldc-rolling-min-2026-09-05` (remote tip `52fa06b9d`), `backup/local-s12-ldc-rolling-min-2026-09-08` (local tip `f888fd1a7`, a 35-line `run.tcl` note) | retiming-ON arm and its notes |
| `seal-minsize-test` | `backup/seal-minsize-test-2026-09-05` | SEAL min-size repro kept out of the testlist |
| `s12-fix-afpr`, `s12-fix-variant-b` | `backup/local-s12-fix-afpr-2026-09-05` (= `5097eb166`), `backup/local-s12-fix-variant-b-2026-09-05` | S-12 fix variants |
| `s06fix-phases-archive` | `backup/local-s06fix-phases-archive-2026-09-05` | S-06 phase history before the squash |
| `fpga-testing-dev-linear`, `fpga-testing-dev-merged-backup` | `backup/local-fpga-testing-dev-linear-2026-09-05`, `backup/local-fpga-testing-dev-merged-backup-2026-09-05` | August working branches |
| `_tmp_ctl` | (content in `backup/timing-control-618f4ce36-2026-09-05`) | scratch |

## Other people's branches (42): untouched

Live or structural: `fpga-testing` (origin's default branch), `fpga-testing-harness`, `fpga-testing-fix`,
`capt-verilator`, `capt-implementation`, `master` (= `capstone-dev` = `optimise`). Fourteen stale branches carry
unique commits and stay. Twenty-one stale branches have every commit reachable from a live branch
(`virtual-debug`, `pc_cap`, `linear-clearing`, `fpga-testing-old-anvil`, `tag-unit`, `new-version`,
one experimental branch named after its author, `exec-merge`, `user_data`, `new-version-testing`, `branch-128bits`, `original`, `load-fix`,
`vanilla-keystone`, `capstone-dev`, `optimise`, `bugfixes`, `testcases-wip`, `revoke-state-machine-wip`,
`actions`, `int-wip`); deleting them would lose nothing, and they were left alone deliberately.

## How this was checked

For every local and remote branch: tip, date, author; `git rev-list --count <tip> --not <every other ref>`
(unique commits); containment in `origin/fpga-testing-dev`, the backup tag of its old tip, `capstone-bootstrap`,
`master`, `fpga-testing`, `fpga-testing-harness`, `capt-implementation`; which `backup/*` tag sits at the tip;
local-vs-remote divergence. A local branch was deleted only if its tip was still reachable from a tag or another
remote ref at the moment of deletion.

---

# Historical: the 2026-08-20 inventory (S-10 timing branches)


> **2026-09-08 — `fpga-testing-dev` IS the rebuilt branch now.** The lead force-pushed `fpga-testing-dev-clean`
> → `fpga-testing-dev` (both `ef5a8eaf2` on origin) and pushed the backup tag; any checkout of the branch must
> `git reset --hard origin/fpga-testing-dev`. `fpga-testing-dev-clean`
> (tip `ef5a8eaf2` = `947327f6d` + the registered switch-in-progress flag; `947327f6d` is tag `chain-v4`)
> is the shared base `7e4dc440f` plus eight commits: lint gate and
> sweep baseline, S-06, S-08, S-07 fix, S-10, S-12, the mtval-cursor feature, synthesis tooling — every fix
> byte-identical to the flashed `5097eb166`'s version and the S-07 on-silicon instrument never added. All 88
> simulation rows identical to the flashed tip's; each commit carries its measured record; lint baseline
> re-derived at S-06 and S-10. Old tip preserved as `backup/fpga-testing-dev-2026-08-21`. Synthesised
> 2026-09-07: WNS −11.717, 97,438 failing, census 100% `dom_switcher/req_en_q` — **NOT usable as a board
> bitstream**: that register is the busy level that gates commit (census doc, 2026-09-07 entry). The hazard
> is measured in simulation (Experiment B). The fix (`ef5a8eaf2`, a registered switch-in-progress flag for
> the three consumers) is committed, sim-validated and audited. Synthesised 2026-09-07: WNS −12.733, 101,143
> failing; all three pre-registered readings hold (the busy-edge hazard has no failing path), but the census is now
> 100% `issue_read_operands` (`lsu_valid_q`, live on every memory instruction) — **NOT usable either**. No bitstream
> from this branch is usable by the census gate — and, retracted 2026-09-08, neither is the resident `5097eb166`: the
> second-launch query on its own checkpoint shows a live second launch behind 101,604 of its failing endpoints. The
> resident stays in use on its board record; the census is not a licence for any build; a flash is now an empirical
> risk decision for the lead with the board lane (census doc, RETRACTED 2026-09-08 section).
> Full account: `history/07-09-2026_14-00-00_fpga-testing-dev-rebuilt-without-the-instrument.md`.
>
> **RE-CHECKED 2026-09-04 — the alarm below is mostly RESOLVED, and one branch is still exposed.**
> Of the five branches this file called local-only, **four are now on the remote** with nothing
> unpushed: `timing-directive-explore`, `s10-merge-candidate`, `s10b-fix`, `s10-fix-wip`.
>
> **`timing-multicycle` is still local-only, and it has moved** — head `eaa4e7984` here is stale,
> it is now `4c4224afb` with **5 unpushed commits**, among them a RETRACTION ("the broad multicycle
> constraint would have hidden REAL violations") and three measured negative results on the
> spill/reload loss. Those are exactly the results that are expensive to re-derive and impossible
> to reconstruct from an artifact. **It is not on the push allowlist**, so it needs the branch
> owner to push it or to add the line. Until then it is one disk failure from gone.
>
> Also note the superproject branch rename: `capstone-bootstrap` → `dev` (2026-09-04). That does
> not affect the `capstone-ariane` branches named below.

Everything built during the S-07 / S-10 / timing work, what is on it, and what it is waiting for.
**Five of these eight branches exist only on this machine.** Push them or they are one disk away
from gone. *(As of 2026-09-04 that count is one — see the note above.)*

## Ready to run — a synthesis machine can check one out and run `bash synth-guard.sh`

| branch | head | on remote | what it is |
|---|---|---|---|
| `timing-multicycle` | `eaa4e7984` | **local only** | the merge candidate **+** the domain-switcher multicycle constraint. **Run this one first.** |
| `timing-directive-explore` | `8696dfdc9` | **local only** | the above **+** `place`/`route` directives raised `RuntimeOptimized` → `Explore`. **Run only after the multicycle result is in.** |

## The fixes

| branch | head | on remote | what it is |
|---|---|---|---|
| `s10-merge-candidate` | `c2211c9a8` | **local only** | S-07 + S-10 + S-10b combined, both audits run against it |
| `s10b-fix` | `c867dfcbb` | **local only** | S-10b alone — the granule-granular load/store hazard |
| `s10-fix-wip` | `4fee13b2d` | **local only** | S-10 alone, before it was merged. Historical. |
| `fpga-testing-dev-s06fix` | `c3ca1b270` | pushed | the mainline: S-07 + S-10, **no S-10b** |

## Controls — do not "improve" these, their value is being unchanged

| branch | head | on remote | what it is |
|---|---|---|---|
| `timing-control-e1140aeea` | `39b21639d` | pushed | `e1140aeea` RTL byte-identical + analysis scripts. **Already run**: reproduced WNS −10.629, 96727 endpoints, 30 loops, 20.89 GB — proving the flow is deterministic. |
| `timing-control-618f4ce36` | `9ab636896` | pushed | `618f4ce36` RTL + scripts. Never run; superseded, because that build's WNS was never read so there is nothing to reproduce. |

## The two changes that are NOT RTL, so they are easy to lose

**Multicycle constraint** — appended to `corev_apu/fpga/constraints/genesys-2.xdc`:

```tcl
set_multicycle_path 2 -setup -from [get_cells -hier -filter {NAME =~ *dom_switcher/cur_idx_q_reg*}]
set_multicycle_path 1 -hold  -from [get_cells -hier -filter {NAME =~ *dom_switcher/cur_idx_q_reg*}]
```

**Implementation directives** — `corev_apu/fpga/scripts/run.tcl:132-133`:

```tcl
set_property "steps.place_design.args.directive" "Explore" [get_runs impl_1]
set_property "steps.route_design.args.directive" "Explore" [get_runs impl_1]
```
(was `RuntimeOptimized` on both; revert by putting that word back)

## Measured numbers these branches are compared against

| build | WNS | failing endpoints | loops | peak RSS | elapsed |
|---|---|---|---|---|---|
| `618f4ce36` | never read | — | — | 4958 MB* | 2h23m |
| `e1140aeea` | −10.629 | 96,727 | 30 | 20.94 GB | 1h48m |
| `39b21639d` control | **−10.629** | **96,727** | **30** | **20.89 GB** | 1h49m45s |
| `80843404c` (S-10) | −16.400 | 102,774 | 16 | 21.53 GB | 1h45m |

\* single-process RSS, not the tree-summed figure the guard reports — not directly comparable.

The control reproducing `e1140aeea` **exactly** is what establishes that the flow is deterministic
and therefore that the S-10 build's 5.8 ns regression is attributable to the S-10 change.

## What is still open

- **The multicycle scope.** The constraint asserts that *every* path from `cur_idx_q_reg` is
  multicycle. An RTL-oracle review of each consumer was commissioned; if a single-cycle consumer
  exists the constraint must be narrowed with `-to` before it is trusted. **A quiet timing report
  is exactly what a wrongly-scoped multicycle produces**, so quietness is not evidence.
- **AMO over a capability granule** — `wt_axi_adapter.sv:155` omits `ATOMIC_REQ` from `needs_tag`.
  Invariant I4, untouched by any of the three fixes.
- **Composed liveness** of S-07 + S-10 + S-10b — reads as non-cyclic on quoted RTL, never observed.
- **The lint gate is RED by design** on every fix branch: `UNOPTFLAT 39` baseline against 40, the
  loop S-10 adds on `wt_dcache.rd_ctag`. Only a synthesis run settles whether it matters.
