# FPGA reproducer packages

Self-contained reproducers for the RTL/FPGA issues in
`capstone/docs/ref/ISSUES.md`. One directory per issue, named `R<nn>-<slug>` (RTL-level) or `S<nn>-<slug>` (observed on silicon, cause not yet localised to a specific RTL site).

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
| `S07-capability-untagged-on-reload/` | *(open, not root-caused)* | a **genuine capability** read back from memory comes back **UNTAGGED**, so the next `cincoffset`/dereference raises **mcause 25**. Three instances in three unrelated functions — a `memcpy` stack slot, the shared-region payload capability in `output_text`, and SQLite's lookaside pointer — so it is not specific to any caller or primitive. **SPORADIC**: the same binary passes 5 of 6 genuine executions and wedges once. **NOT S-06** (that is plain data losing its high half, and raises nothing); do not merge them. Four rungs with firing controls exclude the simple round-trip properties, and disassembly excludes a partial overwrite and the write-buffer `.user` clobber |
| `S08-s06fullfix-bitstream-cannot-run-domains/` | **S-08** | on `caplifive_s06fullfix.bit` the monitor took an UNHANDLED TRAP just after a domain's first share returned, so that bitstream could run no domains at all. **RESOLVED — fixed in RTL and verified on silicon 2026-08-15** on `caplifive_s06fixs08fix.bit`: `EXCX:0000E002` count 0 against 4 of 4 boots on the broken build, and the S-06 acceptance gate became readable in the same boot and passed |
| `S09-write-buffer-tag-forgery/` | **S-09** | a capability **survives the plain store meant to destroy it** — a scalar store over a capability leaves the tag set, which is a forgery primitive. Carries a **narrowed timing caveat** (2026-08-20): the fix is exonerated as the cause of the timing failure, but the measurement was taken on a bitstream missing setup (WNS −10.629 ns on one clock) |
| `S10-write-buffer-forward-residual/` | **S-10** | a **scrubbed capability still reads back LIVE** while the store that scrubbed it is in the write buffer — the forwarding path returns the pre-scrub value. Same timing caveat as S-09, and the two share a bitstream |
| `S11-seal-minsize-alignment-inert/` | **S-11** | `SEAL` enforces **neither its minimum size nor its base alignment** — an instruction-semantics defect, not a cache or tag-path one. Its README opens by redirecting readers who arrived chasing a tag-loss symptom, because it shares nothing with S-06/S-07/S-09/S-10 |
| `S12-wherecode-notcap-operand-vs-memory/` | **S-12** | a capability operand arrives at its consumer as `cnull` (`mcause 25`, `tval 0`) while **memory is intact** — the fault is in what the execution unit *ingested*, not in what was stored. **ROOT-CAUSED 2026-09-03**: a scoreboard WAW clear that ignores `commit_ack`, letting a written-back-but-unretired `stc` entry forward its nulled source to a younger `ldc`. Fixed in RTL, synthesised, flashed. Post-flash verification is **consistent with fixed, not proven** (p = 0.071) — read `01-DECISION-TABLE.md` and `02-REFUTED.md`, which record what the evidence does *not* support |
| `S13-o1-dyn-rev-node-hang/` | **S-13** | at `-O1` the domain **hangs in the DYN/rev-node path with no exception**. `OPEN`. The single-entry store syncer is **CLOSED as a cause** — three structural arguments, not absent counters, which is why the missing `req_set` guard that any later reader will reach for is not the answer |
| `RTL-cap-mcause-off-by-one/` | *(spec violation)* | every capability `mcause` from the **data path is one code too high**, and 25 aliases. Verified against the reference model; **not** reproduced as a functional failure — its cost is **misclassification**, which matters because several reports above classify faults by `mcause` |
| `RTL-domain-trap-vector-unset/` | *(root-caused, half-fixed)* | a domain enters with **no trap vector** — `create_domain` never writes the trap-vector context slot, so a fault in a domain storms at address 0. Firmware half **confirmed on silicon 2026-09-02**; writing `dom_seal[1]` moves the failure to "the handler's own prologue faults in the wrong capability context". **Not firmware-only** — the remaining half is that a trap does not switch domains |
| `ARCHIVED/` | — | packages whose defect is **fixed in silicon**; see `ARCHIVED/README.md` |

**Archived 2026-08-04:** both R-14 packages (`R14-frame-pad/`, `R14-strline-struct/`) moved to
`ARCHIVED/`. R-14 and R-16 were the **same** capability operand-forwarding bug
(`capstone-ariane 7aac52f93`), fixed by `caplifive_fixed_forward.bit` and verified on the
board. Neither is an open issue; do not hand either to the board owner as one.

Both remain useful as **bitstream regression tests** — a third bitstream
`caplifive_65536_nodes.bit` exists whose forwarding-fix status is unconfirmed, and either
package answers that in one boot. `ARCHIVED/R14-frame-pad/` is the cheaper check (two ~10 KB
domains with frozen images); `R16-entry-stall/` needs a 1.5 MB SQLite build.

## Package layout — the convention, and where it is not yet followed

**A folder IS the report.** It is handed to the hardware side as a single link with no message
body, so anything not in the folder does not reach anyone. Everything below follows from that.

A new package should look like:

```
<ID>-<slug>/
├── 00-README.md      the report. Self-contained, and its FIRST paragraph names the sibling
│                     issues, so a reader who arrived with the wrong symptom is redirected
│                     immediately rather than after reading the whole thing.
├── SHA256SUMS        hashes of every committed binary
├── src/              the source of the reproducer
├── board/            frozen .dom images — the exact binaries that reproduced
└── evidence/         result lines, decode tables, disassembly extracts
```

**One issue per folder, always.** Never two signatures, even when they share a trigger and a
workaround: two have had to be split *after* the link was already out, which left the live page
showing the other issue's evidence.

**No status lines that go stale silently** — "not yet shared", "to be confirmed next week".
Nothing prompts a re-read, so they simply become false. A dated status (`RESOLVED 2026-08-15`) is
fine; an undated forward-looking one is not.

**Existing packages do not all match this**, and that is deliberate — several are live links held
by the hardware side, and renaming a file inside one changes a page someone may be reading. The
spread as it stands: `README.md` vs `00-README.md`; `SHA256SUMS` vs `IMAGE-HASHES.txt` vs neither;
payload under `src/`, `board/`, `sim/`, `evidence/` or `minimal-repro/`. **Apply the convention to
new packages; do not retrofit it to sent ones.**

### A measurement package MUST carry its artifacts; a source-level one need not

Audited 2026-09-04: three packages contain no file other than markdown —
`RTL-cap-mcause-off-by-one/`, `RTL-domain-trap-vector-unset/` and `S13-o1-dyn-rev-node-hang/`.
**Two of those are fine and one is not**, and the difference is what the package claims.

- A **source-level** finding ("the RTL does X, here is the line, here is the reference model")
  is fully reproducible from the quoted `file:line` against a named revision. The two `RTL-*`
  packages are this kind. No binary is needed and adding one would be theatre.
- A **measurement** ("this image wedged in 8 of 8 boots") is reproducible only against the image
  that produced it. `S13-o1-dyn-rev-node-hang/` is this kind, and its images are **gone** — no
  `.dom`, no `SHA256SUMS`, no build command, no compiler revision. 63 boots of unusually strong
  evidence, and nothing anyone can re-run. A fresh build of "the same" configuration is a
  **different image** and can only speak about the class, never about the draws on record.

So the rule is not "always commit binaries". It is: **if the claim is about a specific artifact,
the artifact ships with it — and if it cannot, the package must say so in its first paragraph**,
because a reader has no other way to discover that the evidence is unrepeatable. S-13 now says so.

This is cheap at the time and unrecoverable afterwards: the images cost one board session to
produce and are now unobtainable at any price, because the compiler that built them has moved.

### Three states, not two

A package is **open**, **archived**, or **resolved-but-retained**. The third is real and is why
`R16-entry-stall/` sits alongside the open packages while being marked resolved: it is the
cheapest one-boot acceptance test for a bitstream's forwarding-fix status, so moving it into
`ARCHIVED/` would hide the instrument. `ARCHIVED/` is for packages nothing further will be run
against.

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
`capstone/docs/ref/HOW-TO-LAUNCH-ON-FPGA.md`, and the decision procedure is the
`board-run` skill. **Always run a known-entering control FIRST in every boot** — it fails
roughly 1 in 5, and a boot whose control fails is VOID and carries no verdict. When
batching several domains into one boot, put a wedging variant **last** — a wedged domain
takes the core with it and everything after it in that session is lost.
