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
| `RTL-store-user-metadata/` | *(observation)* | every store routes capability metadata into the dcache write-user sideband; the invariant it tests is **correct** in silicon. Its open question — no traced path from `data_wuser` into software-visible data — is answered by **`R19-...`**, not by `R18-...`, and only on the board: R-19 exhibits a slot returning `0x08000000 + n`. R-18 explicitly **retracts** the claim that it closes this |
| `R18-scalar-store-metadata-clobber/` | **R-18** | the **zeroing** signature — a plain scalar store in the **upper half** of a 16-byte row silently loses increments (e.g. 567 where 576 was expected), with raw readbacks showing **no** metadata anywhere. 18 frozen images. A dual-bank splash is demonstrated in RTL simulation, though not the same slot the board damages |
| `R19-movc-zero-metadata-in-slot/` | **R-19** | the **metadata-in-slot** signature — the store's own slot comes back holding `compress_cap(NULL) + n`, e.g. `0x08000A31` = `0x08000000` + 2609, where the program only ever wrote an integer. Does **not** reproduce in RTL simulation. Shares a trigger class and a workaround with R-18; whether they are one defect or two is **unknown** |
| `R20-stc-rs1-cursor-forward-x10/` | **R-20** | after `stc rX,0(a0)`, a `ld a0,0(a0)` is read by the **next instruction** as the store's BASE ADDRESS, not the loaded value. Only on **x10/a0** (the same shape on `t1` is clean), only with a **capability** store (`sd` is clean), and only while both adjacencies hold — one `nop` either side cures it. Corrupts no memory. Reproduced standalone in a **13 KB** rung; the poisoned value is measured, not inferred |
| `S06-untagged-ldc-stc-high-half/` | **S-06** | a 16-byte `ldc`/`stc` round trip over **plain, untagged** data keeps only its **low 8 bytes** — each chunk loses its high half. That pair is the only copy that preserves capability **tags**, so it is what every pointer-bearing struct is copied with, and half of every such buffer is silently destroyed. Corrupts **memory**, unlike R-20. Root cause read out of the D-cache (`wt_dcache_mem.sv:310` force-zeroes bank 1 when the shadow tag is clear; `:140` then gates the store on metadata *content* so it never writes the high half). Reproduces in **RTL simulation in 499 cycles with its own control**, and standalone on the board in a **10 KB** rung that returns 16 where 32 is correct. QEMU has an explicit `scalar_hi` field for this case and cannot show it |
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
