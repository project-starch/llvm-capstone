# A1 in a Capstone domain: the free-to-reuse gap of SQLite's allocators

The question of A1 in the nested-allocators paper: does level 0 see the objects in the custom
allocator? Here level 0 is memsys5, SQLite's own buddy allocator over a region the host lends,
and the custom allocator is the lookaside pool above it. `memhook.c` records every
allocation and free at both levels from inside the domain and prints the paper's text format on
the payload; `speedtest1`, SQLite's own benchmark, is the workload. Two arms: `memsys5`, the
unprotected build, and `sublet`, the same domain with both allocators on the Sublet port
(`capstone/ports/sqlite/sublet/`). Six probes read one byte each where
the two arms must differ.

The port, the domain and the build are `capstone/ports/sqlite/`; this directory holds what
is the experiment's: the instrument, the patches that call it, the probes, the runner that sets
the sizes, and the passes as they were recorded.

```
# the diagnostic QEMU built to CAPSTONE_QEMU_BINARY (see the last section); run.sh exports
# CAPSTONE_GP_NONLIN=1
bash run.sh all --size 1 --reps 3               # both arms, files in results/<stamp>/
bash run.sh sublet --size 100 --reps 3          # the paper's pass, about seven minutes a run
bash run.sh memsys5 --ref results/20260910T014839Z   # verdict: bit-identical to that pass?
bash run.sh probe 2                             # the control first, then the port
```

## What is where

| File | What it is |
|---|---|
| `run.sh` | the arms, the probes, the sizes; cuts the instrument's lines and SQLite's statistics from the log, records the images' sha256 and a provenance line for every input |
| `memhook.c` | the instrument, one source for the three platforms of A1: hosted on x86 and CheriBSD, where the paper repository fetches it from here at a pinned commit, and freestanding in the domain under `-DMEMHOOK_FREESTANDING`, with what a domain lacks replaced under that one macro: no libc level, no mmap (a hashed table in the image for the pool's slots, a direct-mapped one for memsys5's atoms, carved by the domain beside the arena), 32-bit clocks, the report on request |
| `hook-3530300.patch` | the thirteen hunks that call it, applied with `-F0` to copies of the sources in the build directory; `hook-3530300-sublet.patch` the same hunks with the port's contexts |
| `probes.c` | the six probes, one read each (below); `speedtest1_probe` runs instead of the benchmark when the port runner links it in |
| `results/<stamp>/` | a pass: `<arm>.rep<i>.hook.txt`, `<arm>.rep<i>.stats.txt` (SQLite's `--stats` and the port's primitive counts), `image.sha256`, `provenance.txt`, `probe<n>.txt` |

The seams in the port that this uses: `SPEEDTEST1_HOOK=1` with `SPEEDTEST1_HOOK_SRC` and
`SQLITE_HOOK_PATCH` links the instrument in, `SPEEDTEST1_PROBE=n` with `SPEEDTEST1_PROBE_SRC`
hands the run to a probe; the instrument defines `speedtest1_hook_table_bytes`, `speedtest1_hook_install`
and `speedtest1_hook_report`, and the domain exports its two payload writers for both.

## The pass

memsys5 is the configured heap (`SQLITE_CONFIG_HEAP`) over a region the host creates and lends
(`sqlite_host.user --pool`), so there is no level below it and no libc line, as in the paper's
memsys5 arm on x86. Under the port the pool is a linear region (`--arena`) and memsys5's tables
sit in a second region (`--tables`) beside the instrument's table. Above 4 MiB a region comes
from the kernel's CMA area; the port runner passes `cma=1G`. At `--size 100` the pool is 2^21
atoms, 130 MiB, and both arms hold the same atoms so every address matches. Floating point is
omitted (`SQLITE_OMIT_FLOATING_POINT`), the domain has no libm. Three repetitions per arm,
bit-identical: the domain is deterministic and the runner's line endings are normalised across
read boundaries, which is what made them so.

The instrument's check is the paper's x86 pass at the same size and configuration: the pool's
count agrees within 0.01 percent and its curve to three digits at every mark, memsys5's count
within 0.2 percent. The pool's bytes are larger in the domain, 755 against 435 MB for the same
requests, because a purecap pointer is 16 bytes and SQLite's objects grow with it.

## The Sublet arm

The port itself, the recipe as it lands in the two allocators, the bookkeeping of A7 by
class, and what it gives up are written where the port lives,
`capstone/ports/sqlite/sublet/README.md`; the port is applied on top of the compatibility
port and counted apart from it. What this pass measured on it:

- `--verify` passes in both arms, 32 of 32 tests.
- Profile: per test, the lookaside's counts are the unprotected arm's to the allocation and its
  histograms are identical on both clocks. memsys5 has one allocation more per connection, the
  pool's side table, visible as `@alloc[0]` plus one and `@peak` plus one in every test, and
  5408 bytes. Its reuse curve moves within 0.2 points at every bucket edge.
- Counters, sublet arm: 2529498 delin, one per allocation at both levels but the pool's block,
  which stays linear; 3027344 mrev and 3027344 revoke, the allocations plus 497845 merges plus
  the pool's block; 498013 splits, 497845 at memsys5, 168 for the pool's slots, none at the
  initial carve, the pool being one block of 2^21 atoms; 497846 init, one per merge and one for
  the pool at close. The arithmetic closes.

## The probes

`run.sh probe N` builds the unprotected domain first, as the control, and the port second; the
verdict is against what the probe is for. The marker strings are the ones the recorded passes
carry.

| N | The read | Unprotected | The port |
|---|---|---|---|
| 1 | a column name after `sqlite3_finalize` freed it, a lookaside slot | reads through, `NOTRAP` | halts at the read, cause 24: the reloaded alias lost its tag with its node |
| 2 | a memsys5 block after `sqlite3_free` | reads through, `NOTRAP` | halts, cause 24 |
| 3 | a memsys5 block after its buddy was freed | reads on | reads on: a revocation reaches one object, not the block around it |
| 4 | a lookaside slot after the slot beside it was finalized | reads on | reads on |
| 5 | the last byte of a 128-byte block, a 100-byte request | reads | reads: the alias carries the block's bounds |
| 6 | the byte after that block | reads through, `NOTRAP`: the alias carries the whole arena | halts, cause 5, out of bounds |

## What the numbers are scoped to

An independent review of the port against the emulator and the specification (2026-09-10)
confirmed the coverage, the tree invariants, the index arithmetic and the instrument's
fairness, and named the limits below.

- The emulator. It reuses no revocation node, and its pool is sized at start
  (`CAPSTONE_REV_NODES`, `run.sh` sets eight million); a run at `--size 100` takes 3525357
  (498013 splits and 3027344 mrevs), at `--size 1` 43417. Its tag map is indexed since this
  pass; before, a run at `--size 100` took over an hour in a linear scan, now about seven
  minutes. The silicon's pool holds 1021: the port needs the RTL with node reclamation to run
  on the board.
- The emulator's revoke and init. When a revoke finds a linear node below the handle (a free
  buddy, a free slot, a slot's handle) the region comes back uninitialised, and the port turns
  it linear again with one `init`. The specification lets that init succeed only after the
  region has been written through: revoke leaves the cursor at the base, a capability-grained
  store at the cursor advances it, and init wants it at the end. So every merge and the pool's
  destruction cost a write of the block, and `sublet_give_to` writes it. The emulator the
  recorded passes ran on leaves the cursor at the end instead, where that loop runs no
  iteration and the init is the one instruction it always was; the merge line has since moved
  to the specification's rule (Q-07, capstone-qemu 72fb56be86, which also makes init trap
  rather than abort). The counts are the same either way, the cycles are not, and the same
  hook text comes back on both: the sublet arm run against the recorded passes on a build of
  `diag/domain-runs` is bit-identical to them at `--size 1` and at `--size 100`.
- The emulator's stores of linear capabilities do not null the source register; hardware that
  enforces linearity does, which is why `sublet_take_linear` reads the base before the store and
  the readers of a slot store the capability back after loading it.
- The specification's split puts the upper half's base one past the split address where the
  emulator puts it at the address. Every carve in the platform's own code passes exclusive ends
  the same way; if the RTL follows the text, that is a platform question, not the port's.
- One more allocation. The pool's side table is a memsys5 block per connection. It shifts
  memsys5's buddy layout, so memsys5's histograms differ bin by bin (145815 distinct addresses
  against 146902) while its curve stays within a point, 0.17 at the widest. The lookaside's
  are bit-identical.
- Not covered. SQLite's page cache takes one memsys5 block of twenty pages at open and recycles
  pages inside it through its own list. That reuse is neither measured nor revoked in either
  arm; the claim is about the objects of memsys5 and the pool.
- Robustness. A double free of a memsys5 block revokes a linear handle and aborts the emulator
  instead of faulting the domain.

## The fault the tooling found

The instrument first halted the domain at `sqlite3RunVacuum` with "cjalr requires capability
in rs1". The cause is not the instrument. The compiler emits `movc s4, a5` and then calls
through a5, and on Capstone a `movc` of a linear capability moves it. The callee capability was
linear because QEMU re-fabricates gp from pc_cap at every call, and pc_cap is linear again once
the first call returns to the entry frame. The diagnostic QEMU (capstone-qemu branch
`diag/domain-runs`, the same five commits rebased onto the merge line) has
`CAPSTONE_GP_NONLIN=1`, which keeps the fabricated gp non-linear, the type the entry glue gives
it on purpose. Every domain pass here runs with it, `run.sh` exports it. Which build of it a
pass can use is the previous section's last point: every arm runs on either, the sublet arm
since the port writes a block through before `init`.

## Passes recorded

| Stamp | Size | Arms | What it established |
|---|---|---|---|
| `20260910T002220Z` | 1 | memsys5, 3 reps | the instrument in the domain: both levels recorded from inside, the counts within three percent of x86 |
| `20260910T014839Z` | 1 | memsys5 and sublet, 3 reps | the port passes all 32 tests, three runs bit-identical; the review's limits; probes 1 to 6 located here |
| `20260910T050512Z` | 100 | memsys5 and sublet, 3 reps | the paper's pass: the pool and the tables as host regions from CMA, released after the run; the counters above |

The provenance records name the working branch and the submodule branches the passes were run
from, and the port's files at their path of the time, `adapted/`, before the port moved to
`sublet/`; the sha256 beside each path is what identifies the file. The patch's header comment
was reworded on the move, so its recorded sha256 names the pre-move file; the hunks are the same.
The content those commits carry is, file for file, what the reviewed branches
`sqlite/1-speedtest1-domain`, `sqlite/2-sublet-port` and this one hold, with the instrument
and the probes moved here from the port and the domain's seam for them made explicit; the
images' sha256 in each stamp are the check.
