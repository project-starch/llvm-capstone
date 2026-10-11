# Results, 2026-10-06 -- three arms over the whole extracted population

The 674 extracted cases run as the domain mruby on the Capstone application VM,
one boot per arm, one verdict line per case. Summaries only; the raw guest
output is not committed (SCHEMA.md rule 6).

| arm | faults | watchdog | completed | of |
|---|---:|---:|---:|---:|
| `sysalloc-none` | 15 | 5 | 654 | 674 |
| **`sysalloc-bounds` (baseline)** | **16** | 5 | 653 | 674 |
| `sublet-gc` | **23** | 5 | 646 | 674 |

Every arm passed its own three controls first -- `scripts/smoke.rb` and the 40-
and 500-frame recursion ladder. An arm whose control fails cannot report a catch.

## What each step adds

`sysalloc-bounds` is the baseline: per-object heap bounds are what an
application gets by default since PR #170, so a catch is a case the bounds arm
does **not** fault on.

| step | adds | of those, spatial or temporal |
|---|---:|---:|
| bounds over none | 1 | 1 |
| **sublet-gc over bounds** | **7** | **6** |
| regressions (bounds faults, sublet silent) | 0 | 0 |

The seventh addition is `23_1737589f0_file-join-recursive-array`: C-stack
exhaustion, which the bounds arm meets as a watchdog kill and the sublet arm as
a fault. It is a real defect and not a memory-safety one, so it is not counted
in the six.

**All six faulted with cause 24**, a dereference of a revoked capability --
not bounds, not tag integrity. Five are classed `temporal` and one `both`, and
those classes come from the upstream fix commits, assigned before this run.

## Two of the six are outside the 165-row ledger

`09_0cf969a2b` and `10_7c5915799` are not ledger rows: their tests **pass** at
the pin, so the ledger's "fails at the pin, passes at master" rule could never
admit them. Host ASan is what found them, and the bounds arm confirms it --
both complete there with the harness reporting PASS. Only the sublet arm reports
at all. A third, `13_4a386f80e`, is the single thing the bounds arm adds over
the unprotected control.

So a third of what Sublet contributes here was invisible to the method that
produced the 165 rows, which is the reason this run used all 674 cases.

## Silence is not one thing

Of the six, two complete with the harness reporting PASS on the baseline
(`09_0cf969a2b`, `03_59552ecb8`): the defect executes and nothing anywhere sees
it. The other four already print a **wrong answer** on the baseline
(`04_606d9a6b2`, `05_628ccec60`, `08_fb4974528`, `10_7c5915799`). For those,
Sublet does not reveal an unknown defect; it converts a silent wrong answer into
a fault at the access. Each case's `arms` field says which it is.

## What the unprotected arm is, and is not

`sysalloc-none` is not a C baseline. It is the first-fit heap with
`CAPSTONE_LEVEL0_OBJECT_BOUNDS=0`, so every pointer carries the whole arena's
bounds and `free` only marks -- but tag integrity is in the hardware and cannot
be switched off. 13 of its 15 faults are cause 24: the program reads a pointer
out of memory that has since been overwritten, and the untagged result cannot be
dereferenced. Those defects need no allocator protection at all. The host ASan
column in each case's `live_proof` is the nearest thing here to a C baseline.

## Reproducing

    bash /tmp/capstone/mruby-arms2/build-one.sh bounds   level0    ''
    bash /tmp/capstone/mruby-arms2/build-one.sh none     level0    '-DCAPSTONE_LEVEL0_OBJECT_BOUNDS=0'
    bash /tmp/capstone/mruby-arms2/build-one.sh subletgc sublet-gc ''
    bash /tmp/capstone/mruby-arms2/run3.sh

`MRBD_SDK_CFLAGS` is the hook the control arm needs; it must arrive as one cmake
argument, because a word-split flag list becomes separate `-D` arguments and
cmake takes an unknown one as a cache variable and compiles without it. That
failure produced a control image byte-identical to the baseline, and comparing
the two images' sha256 is what caught it. The shas are in `inputs.json`.
