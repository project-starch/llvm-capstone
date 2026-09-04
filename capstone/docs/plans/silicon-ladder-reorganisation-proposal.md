# Proposal: reorganising `capstone/tests/runtime-qemu/silicon-ladder/`

**Status: PROPOSAL (2026-09-04). Nothing here is implemented. It needs a decision before any
file moves, because several of the assumptions people have been making about this directory
turn out to be false.**

## What is actually there

891 files in one flat directory: 667 `.c`, 209 `.h`, 7 `.py`, 6 `.sh`, 2 `.S`.

**142 of them are byte-identical duplicates**, in 29 groups. The largest:

| copies | representative |
|---|---|
| 32 | `bnd2_host.c` |
| 14 | `beebs_prime1m_host.c` |
| 9 | `beebs_prime.c`, `blobpeek0_app.c`, `blobpeek0_fpga_app.c`, `blobpeek0_host.c`, `k1200_host.c` |
| 8 | `al_host.c`, `beebs_prime1m_fpga_app.c`, `janne_diag_host.c` |

These are the per-rung `_host.c` / `_app.c` / `_fpga_app.c` scaffolds. A rung varies its *kernel*;
the scaffolding around it is copied unchanged, so the same file exists 32 times under 32 names.

## The correction that blocks the obvious plan

The working assumption — recorded in the cleanup survey and worth stating plainly so nobody acts
on it — was that the large `cp*_kernel.h` files (**5.0 MB across 208 headers**, up to 1.18 MB for
`cp512_kernel.h`) are **generated**, and therefore cheap to drop and regenerate.

**They are not. There is no generator for them anywhere in the tree.** The 13 scripts in the
directory were checked; none emits them.

Their *content* is machine-shaped — `cp512_kernel.h` is 512 never-called functions, and size
tracks the number almost exactly (cp64 146 KB → cp512 1183 KB) — but the shaping happened once,
outside the repo, and the header comment on each is handwritten. Each file also encodes a
specific `.text` displacement, which is the entire variable under test:

> *1024 never-called functions shift `.text` by roughly 64 KiB. The probe is the same bounded
> strlen over a pointer read out of a global struct.*

So a regenerated file that shifted `.text` differently would be a **different experiment wearing
the same name**, and the draws taken against it would silently stop being comparable to the
recorded ones. **Do not delete, regenerate, or reformat any `cp*_kernel.h`.**

## What is safe, and what is not

**Safe — mechanical, reversible, no artifact changes:**

1. **Subdirectories per rung family.** `blobpeek/`, `beebs_prime/`, `cp/`, `bnd/`, `k*/`, `janne/`.
   Pure `git mv`; the build path is `build-ladder-domain.sh`, which takes explicit file names.
2. **A per-rung manifest** — one small table naming, for each rung, its kernel, its scaffold, and
   what question it was built to answer. That is the thing this directory most lacks: with 891
   flat files, "has this already been tried?" currently costs a full read.

**Not safe without more work:**

3. **Collapsing the 142 duplicate scaffolds into a shared template.** Correct in principle and it
   is where the file count comes from, but each `_host.c` is one half of a frozen board artifact.
   The board results on record were produced by *those* files. Replacing them with a shared
   template plus per-rung defines changes what the compiler sees, and this project has already
   learned once that a code-layout change with no semantic content flips a result
   (`cp*_kernel.h` exists *because* of that). Any collapse must be gated on rebuilding a sample
   of rungs and showing the artifacts are **byte-identical** — the same check that settled the
   `janne-complex` twins, where two scripts that looked different produced the same SHA-256.

**Never:**

4. Deleting `cp*_kernel.h`, per above.

## Recommendation

Do 1 and 2. They are reversible and cost nothing.

Treat 3 as a separate piece of work with the byte-identity gate written down *first*, and do not
start it while a board campaign is in flight — a rung whose artifact silently changed mid-campaign
is the most expensive kind of instrument fault this project has.
