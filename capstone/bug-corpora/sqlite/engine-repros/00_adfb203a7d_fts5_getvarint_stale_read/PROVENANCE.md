# adfb203a7d_0 -- adfb203a7d

Upstream fix `adfb203a7d`. Collected in round R3 (backport filter).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
adfb203a7d_0 -- upstream corrupt-database regression image, backported to 3.22.0.
 *
 * Collected by the backport filter: upstream fix adfb203a7d removes code that is still
 * present in 3.22.0 (or guards a function 3.22.0 has unguarded), and its own regression
 * test ships the triggering database as a hex dump. Host ASan on stock 3.22.0:
 *
 *     heap-buffer-overflow in sqlite3Fts5GetVarint (an over-read; see PROVENANCE.md)
 *
 * The image is embedded and served by repro322_memfs.c, an in-domain in-memory FILESYSTEM.
 * ext/misc/memvfs.c cannot be used here: it serves only the main database, so the pager
 * cannot create a rollback journal, and it takes its buffer as an INTEGER in the URI --
 * a pointer forged from an integer, which on a capability machine is untagged and faults
 * on first use.
 *
 * STATEMENT BY STATEMENT, deliberately. Handing the whole script to one sqlite3_exec()
 * stops at the first error, and these scripts routinely take SQLITE_CORRUPT on an early
 * statement while the defect is reached by a later one. Screening the whole script in one
 * call scored 3 of 17; per statement it scores 10 of 17, matching a real file exactly.
 *
 * REACHABILITY: every statement's rc is printed. SQLITE_CORRUPT (11) on the statement that
 * should reach the defect means SQLite rejected the image first -- a real negative, not a
 * silent pass.
```

## An over-read, not a use-after-free (reviewed 2026-10-15)

This case was collected and recorded as a use-after-free (`free / stale read`, axis temporal).
It is an over-read of a live object, and is now classed `read past the end of the record`.
The directory name keeps `stale_read`: the run bundles and `tags.tsv` refer to the case by that
name, and renaming it would orphan the measurements that cite it. The verdicts on all three
arms are unchanged; only the axis moves.

**The defect.** `fts5SegIterReverse` (`ORDER BY rowid DESC`) searches forward for the page that
holds the doclist's last rowid and keeps it as `pLast`, then does

    iOff = fts5LeafFirstRowidOff(pLast);          /* a u16 from the page header */
    iOff += fts5GetVarint(&pLast->p[iOff], ...);

with no check of `iOff` against the page. The trigger's corrupt leaf has a 24-byte body and a
header offset of 148, so the decoder reads 124 bytes past the end of `pLast`, which is live at
that point (only the pages it replaced are released). Upstream's own description of
`adfb203a7d` is "Fix a buffer overread in fts5 that could occur when processing an "ORDER BY
rowid DESC" query against corrupt database records"; the fix reads both pages with
`fts5LeafRead`, which rejects a leaf whose `szLeaf` exceeds its size.

**Why it was recorded as a use-after-free.** ASan names a report by the shadow state of the byte
touched, not by where the pointer came from. The over-read lands in a page that
`fts5SegIterReverse` has just released. In the configuration of
`results/2026-10-11-host-asan` it lands in that block's redzone and ASan reports
heap-buffer-overflow, with the neighbour's free and allocation stacks; an earlier heap layout
put it inside the freed block, which ASan reports as heap-use-after-free, and that report is
what the collection recorded.

**Evidence** (result lines; the probe is `shared/probe/`, sites 20 and 33):

- Host ASan, stock 3.22.0 on the system allocator, `run-host-asan.py` flags:
  `heap-buffer-overflow ... 4 bytes to the right of 64-byte region`, access
  `sqlite3Fts5GetVarint` <- `fts5SegIterReverse`, freed by `fts5DataRelease` <- `fts5SegIterReverse`.
  The same build with upstream's two `fts5DataRead` -> `fts5LeafRead` changes: no report,
  `s1_rc=11` (SQLITE_CORRUPT), `RETURNED`.
- Probe site 33, at the decode in `fts5SegIterReverse`, checks the object itself. On native
  memsys5, on virtual Capstone (stock memsys5) and on CheriBSD purecap it prints the same line
  four times: `LB_OBJECT site=33 iOff=148 nn=24 szLeaf=20502 oob=1`, with `witnesses=0` for
  `pLast` itself. With upstream's fix the site is never reached (`hits=0`).
- Probe site 20, in `sqlite3Fts5GetVarint`, judges by address and reports `kind=UAF` (2 or 3
  times out of 4 reads, depending on the platform's heap layout): the address it is handed is
  inside the released neighbour. The fourth read lands in a live block and is invisible to an
  address check.
- On CheriBSD the pointer behind the read carries `tag=1 cb=<arena>+262144`: its bounds are the
  whole 256 KiB memsys5 arena, and the read is inside them, so CHERI is right to stay silent.
- Verdicts reproduced on the review platform (sources at the commit that recorded them):
  virtual-malloc MISSED (controls complete, complete), virtual-nested-pools CAUGHT with
  cause 28 (`CAP_OOB`) in `sqlite3Fts5GetVarint` (controls fault, fault), CheriBSD
  `RETURNED rc=0` with its positive control faulting under revocation and completing without it.
  The Sublet catch is a bounds catch, which is what an over-read of a request-bounded block gives.

## Re-measured 2026-10-17: one arena on every arm, every arm from the corpus source

Two things were wrong with the earlier spatial readings and both are fixed here. The arms did not
share an arena -- the virtual images were built at the 1 MiB header default, CheriBSD at 262144 --
and on an arena-granular arm the arena IS the bound, so the two were comparing arenas rather than
mechanisms. And this arm compiled its own copy of the case from `ports/sqlite/cheribsd/cases/`
rather than the corpus `case.c`. Both are closed: the case declares `arena_bytes` (1048576 here),
all three builders read it, and `build-corpus-cheri.sh` now takes the case program from the corpus.

| arm | verdict | evidence |
| --- | --- | --- |
| virtual-malloc | **MISSED** | both controls as declared; reached and silent, by the Sublet catch below on the same build pair |
| virtual-nested-pools | **CAUGHT** | both controls as declared; `cause=28 pc=0x3fa022224c` in `sqlite3Fts5GetVarint`, a declared fault site |
| cheribsd-revocation | **MISSED** | `RETURNED rc=0`; `poscontrol` exit 162 with `revocation=1` in the same session |

Reachability on CheriBSD is not inferred from the return code, which is what made the first reading
of case 30 void: probe site 33 was built into this arm's binary -- verified by `strings` to carry the
site's own payload, not just the runtime's `LB_ARMED` line -- and reports `hits=4 witnesses=0` (site 33 tests the object itself and makes no address judgement, so witnesses=0 is expected).
The defect site is entered and the access happens; the mechanism is silent because its bound is the
whole arena and the over-read stays inside it.

Both binaries were verified before the run: built after the change, carrying the embedded image
(`repro_memfs`) rather than `/root/corpus/*.db`, at the declared 1048576-byte arena, and matching
by sha256 in the guest.
