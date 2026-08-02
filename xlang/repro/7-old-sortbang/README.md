# Row 7 — `mrb_bint_reduce` (bigint gem): NOT REPRODUCED

> **This row does not reproduce, and the evidence suggests the row as specified
> does not exist.** This is a documented negative result with a working build, not
> an unattempted row. The full argument is in **`target.md`**.

## Summary of the finding

| Claim in the spec / benchmark table | What was found |
|---|---|
| Issue is mruby **#6701** | #6701's upstream fix (`e50f15c1`, "Fixes #6701") is the **pattern-matching bytecode corruption** — that is **Row 6**, which now reproduces. Searching all history for `6701` returns only that commit. |
| Row is in the **mruby 3.1.0** Tier-1 cluster | `mrb_bint_reduce` does not exist in 3.1.0, 3.2.0 or 3.3.0. It first appears in the 3.4.0 line, only under `MRB_USE_RATIONAL`. The row was never buildable as grouped. |
| **UAF** during GCD in `mrb_bint_reduce` | The plausible hazard (raw `struct RBigint*` locals held across an allocating call) is closed: `mrb_obj_alloc` arena-roots every new object, nothing in the path saves/restores the arena, and `MRB_GC_STRESS` cannot change it. |

## Contents

| File | What it is |
|---|---|
| `target.md` | **Read this first.** The three findings above with source evidence, and what would settle the row |
| `build_config.rb` | host+ASan and riscv64, with `mruby-bigint` + `mruby-rational` enabled |
| `build.sh` | Clean checkout of `cda2567c` → both builds |
| `trigger.rb` | Probe driving `Rational(bignum, bignum)` under maximum GC pressure |
| `run.sh` | Runs the probe; expects clean completion |
| `boundary.md` | The boundary this row *would* have exercised, recorded for a future attempt |

No `asan.txt` — there is no crash to capture. Fabricating one would be worse than
leaving it out.

## How to build and run

```bash
chmod +x build.sh run.sh
./build.sh
./run.sh
```

## Expected outcome

Both native+ASan and `qemu-riscv64` complete with **exit 0** and print
`completed without fault (expected -- see target.md)`.

If a future run *does* fault, that is news: capture the trace and revisit
`target.md`.

## Why the build is kept

It is correct and reusable. Both gems are wired up (`mrb_bint_reduce` compiles only
when `MRB_USE_RATIONAL` is defined, which `mruby-rational` sets), for host+ASan and
riscv64. If someone identifies the real defect, only a trigger is missing.
