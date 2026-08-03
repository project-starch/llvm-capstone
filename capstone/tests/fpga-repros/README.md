# FPGA reproducer packages

Self-contained reproducers for the RTL/FPGA issues in
`capstone/agent-handoff/ref/ISSUES.md`. One directory per issue, named `R<nn>-<slug>`.

These used to live in `/tmp/capstone/*.tar.gz`, which meant every one of them was lost on
reboot and none could be reviewed, diffed or bisected. ISSUES.md says an issue without a
reproducer is not an issue; a reproducer that only exists in `/tmp` is not much better.

| dir | issue | what it shows |
|---|---|---|
| `R01-lsu-hazard/` | **R-1** | a load through one capability register misses a store through another |
| `R02-delin/` | **R-2** | `delin` in domain code wedges the board — `delin.s` vs `nop.s`, one instruction apart |
| `R14-strline-struct/` | **R-14** | straight-line init of a struct array with distinct string constants wedges |
| `R14-frame-pad/` | **R-14** | **the one to hand over.** Two ~10 KB domains whose source differs only in the size of a dead `volatile` pad: one returns 4, the other never returns. Ships bounds/type measurements showing the faulting access is architecturally legal |

## What is committed, and what is not

`R01` and `R02` include their **frozen `.dom` images** (8–38 KB each). That is deliberate:
the point of a reproducer is the exact binary that reproduced, and a rebuild against a
moved compiler may not. They are small enough to carry.

`R14-frame-pad` includes its **frozen `.dom` images and the `lpc` controller** (~41 KB for all
five), pinned by `images/SHA256SUMS`. It is a standalone silicon-ladder rung, not a SQLite
build, which is exactly why it is small enough to carry — and why it should be preferred over
`R14-strline-struct` for any hand-over.

`R14-strline-struct` ships **source and documentation only**. Its four domains are ~1.5 MB each
(6 MB total) because each is a full SQLite build, which is too much to track. Rebuild them with:

    export SQLITE_SUPPORT_OPT_LEVEL=-O1
    for S in 18 20 21 22; do
      OUT_DIR=/tmp/capstone/sqlite-s$S DOMAIN_EXTRA_DEFS="-DCAPSTONE_SQLITE_STAGE=$S" \
        bash capstone/benchmarks/sqlite/build-sqlite-silicon.sh
    done
    #  stage 18 -> variant A (WEDGES)   stage 20 -> variant B (returns 4, expected 16)
    #  stage 21 -> variant C (correct)  stage 22 -> variant D (correct)

The staged-return scaffolding those depend on is in
`capstone/benchmarks/sqlite/sqlite_capstone_domain.c` (`CAPSTONE_SQLITE_STAGE`).

## Running one

Each package has its own `README.md`, and `R02` has a `run.sh`. General board procedure and
the driver contract are in `capstone/agent-handoff/ref/HOW-TO-LAUNCH-ON-FPGA.md`. When
batching several domains into one boot, put a wedging variant **last** — a wedged domain
takes the core with it and everything after it in that session is lost.
