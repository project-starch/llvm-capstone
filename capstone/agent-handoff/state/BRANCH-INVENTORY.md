# Branch inventory — `capstone-ariane`, 2026-08-20

Everything built during the S-07 / S-10 / timing work, what is on it, and what it is waiting for.
**Five of these eight branches exist only on this machine.** Push them or they are one disk away
from gone.

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
