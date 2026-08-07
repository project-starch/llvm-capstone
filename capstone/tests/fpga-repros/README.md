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
| `R16-entry-stall/` | **R-16** | the domain never returns from its FIRST entry (`SHA5` stall, no `SQ: G/enter`). **Resolved** — kept as a one-boot bitstream acceptance test |
| `S01-image-perturbation-hang/` | *(open, unexplained)* | a ~1.6 MB domain returns, but **any** perturbation of its image makes it hang — silently, and correctly under QEMU |
| `RTL-store-user-metadata/` | *(observation)* | every store routes capability metadata into the dcache write-user sideband; the invariant it tests is **correct** in silicon. Its open question — no traced path from `data_wuser` into software-visible data — is **closed by `R18-...` via the STORE side** |
| `R18-scalar-store-metadata-clobber/` | **R-18** | a plain scalar store in the **upper half** of a 16-byte row is silently **zeroed** — its slot receives capability metadata instead of its data. Four frozen images; the sentinel arm starts at 1,000,000 and returns 567 |
| `ARCHIVED/` | — | packages whose defect is **fixed in silicon**; see `ARCHIVED/README.md` |

**Archived 2026-08-04:** both R-14 packages (`R14-frame-pad/`, `R14-strline-struct/`) moved to
`ARCHIVED/`. R-14 and R-16 were the **same** capability operand-forwarding bug
(`capstone-ariane 7aac52f93`), fixed by `caplifive_fixed_forward.bit` and verified on the
board. Neither is an open issue; do not hand either to the board owner as one.

Both remain useful as **bitstream regression tests** — a third bitstream
`caplifive_65536_nodes.bit` exists whose forwarding-fix status is unconfirmed, and either
package answers that in one boot. `ARCHIVED/R14-frame-pad/` is the cheaper check (two ~10 KB
domains with frozen images); `R16-entry-stall/` needs a 1.5 MB SQLite build.

## What is committed, and what is not

`R01` and `R02` include their **frozen `.dom` images** (8–38 KB each). That is deliberate:
the point of a reproducer is the exact binary that reproduced, and a rebuild against a
moved compiler may not. They are small enough to carry.

`ARCHIVED/R14-frame-pad` includes its **frozen `.dom` images and the `lpc` controller** (~41 KB
for all five), pinned by `images/SHA256SUMS`. It is a standalone silicon-ladder rung, not a
SQLite build, which is exactly why it is small enough to carry — and why it should be preferred
over `ARCHIVED/R14-strline-struct` whenever an R-14-shaped check is wanted.

`S01-image-perturbation-hang` is the one **open** package and the only one that is *not*
root-caused. It ships source, recipe and pinned hashes; `run.sh` builds both images, runs a QEMU
differential (both must be correct there) and then the board pair in one boot with a live
control. Its README carries a table of **nine variables already tested and excluded** — read it
before designing any experiment, because every one of those was a plausible-looking hypothesis
that a control destroyed.

`R16-entry-stall` ships **source, recipe and pinned hashes only** — its reproducer is a ~1.5 MB
SQLite build. Its `run.sh` builds, stages and runs it with a control gate, and prints a
present/absent verdict. Note the build is **not bit-reproducible**, so identify an image by
size and carve count as well as by hash.

`ARCHIVED/R14-strline-struct` ships **source and documentation only**. Its four domains are ~1.5 MB each
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

Each package has its own `README.md`; `R02`, `R16-entry-stall`,
`RTL-store-user-metadata` and `S01-image-perturbation-hang` have a `run.sh`. General
board procedure and the driver contract are in
`capstone/agent-handoff/ref/HOW-TO-LAUNCH-ON-FPGA.md`, and the decision procedure is the
`board-run` skill. **Always run a known-entering control FIRST in every boot** — it fails
roughly 1 in 5, and a boot whose control fails is VOID and carries no verdict. When
batching several domains into one boot, put a wedging variant **last** — a wedged domain
takes the core with it and everything after it in that session is lost.
