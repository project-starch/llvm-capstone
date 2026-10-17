# fz10 -- fz10

Upstream fix `FZ10`. Collected in round R2 (fuzz-corpus diff).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
fz10 -- R2 fuzz-corpus diff image, trigger re-derived 2026-10-03.
 *
 * Host ASan on stock 3.22.0 (x86):
 *     statDecodePage:177956, SEGV / out-of-bounds read
 *
 * PROVENANCE OF THE TRIGGER. This entry carried only the database and an ASan
 * log; the SQL was never recorded, because fuzzcheck pairs every database with
 * every script in the corpus and the provenance kept only the database id. The
 * statement below was re-derived by replaying candidate statements one process
 * at a time against a FRESH copy of the image -- a cumulative script does not
 * work here, since the earlier statements rewrite the very corruption the later
 * ones need. It reproduces the recorded crash at the SAME source line.
 *
 * The image is embedded and served by repro322_memfs.c, an in-domain in-memory
 * FILESYSTEM: the pager needs a rollback journal for any write, and memvfs.c
 * serves the main database only and takes its buffer as an integer in the URI,
 * which on a capability machine is a pointer forged from a scalar.
 *
 * REACHABILITY: every statement's rc is printed. SQLITE_CORRUPT (11) on the
 * statement that should reach the defect means SQLite rejected the image first
 * -- a real negative, not a silent pass.
```

## Review 2026-10-16 -- the CheriBSD cell becomes a detection

### The four readings

| reading | result |
|---|---|
| host ASan | re-run: `heap-buffer-overflow` in `statDecodePage` at the cell-pointer walk, via `statNext <- statFilter` -- identical to the recorded bundle |
| virtual-malloc | **MISSED**, `RETURNED` -- against the physical `spatial` row's `cause 5` |
| virtual-nested-pools | **CAUGHT**, `cause=28` in `statDecodePage` |
| cheribsd-revocation | **CAUGHT**: `SIGPROT(34) si_code=1 PROT_CHERI_BOUNDS`, pc resolving to `statDecodePage+828` |

The CheriBSD reading changes for the reason set out in case 24's PROVENANCE and retracted in
case 18's: the 2026-10-06 reading was taken without the corrupt image present, so it was a
NO-READING rather than a miss. `si_code=1` is a bounds violation on a **valid** capability, not
one of the `si_code=2` tag faults the false positives in this review turned out to be.

### Why this one is visible at malloc granularity

Probe site 64 hooks all three unchecked reads in `statDecodePage`. On this image:

```
ENTER     aData=0x210220 aHdr=0x210220 iPgno=2
CELLWALK  iOff=52992 reads=0x21d120
LB_WITNESS kind=CAP-OOB n=1 tag=1 cb=0x1d8c00+262144 ca=0x21d120
<<FAULT>> signal=34 si_code=1 (PROT_CHERI_BOUNDS/valid-cap-out-of-bounds)
```

`cb=0x1d8c00+262144` is the **whole 256 KiB arena**, and `ca=0x21d120` is past its end at
`0x218c00`. The cell-pointer walk, with `iOff` reaching 64768 in eight iterations, leaves the
arena -- and an over-read that leaves the arena is visible even where bounds are at `malloc`
granularity. Case 32's over-read stays inside the arena and that arm is silent on it; Sublet
catches both, because it bounds each sub-allocation. **So the two arms are separated here by the
distance of the over-read, which is a property of the corrupt input, and not by the class of the
defect.**

The free-block chain never runs on this image, which is what distinguishes this case from 32.

### The two base builds disagree

As in case 24: the physical `spatial` row reports `cause 5` and the virtual-malloc build
completes. Base Capstone holds the arena as one capability, so the fault depends on where the
over-read lands relative to the arena end, which differs between builds.

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
| virtual-nested-pools | **CAUGHT** | both controls as declared; `cause=28 pc=0x3f95a12aa4` in `statDecodePage`, a declared fault site |
| cheribsd-revocation | **MISSED** | `RETURNED rc=0`; `poscontrol` exit 162 with `revocation=1` in the same session |

Reachability on CheriBSD is not inferred from the return code, which is what made the first reading
of case 25 void: probe site 64 was built into this arm's binary -- verified by `strings` to carry the
site's own payload, not just the runtime's `LB_ARMED` line -- and reports `hits=9 witnesses=7`, every witnessed read `kind=OOB`.
The defect site is entered and the access happens; the mechanism is silent because its bound is the
whole arena and the over-read stays inside it.

**This case is where the arena is visible.** At 262144 it is CAUGHT on BOTH other arms -- `cause=28`
in `statDecodePage` on `virtual-malloc`, `SIGPROT si_code=1 PROT_CHERI_BOUNDS` on CheriBSD -- and at
1048576 it is MISSED on both, from the same source. The over-read leaves a 256 KiB arena and lands
inside a 1 MiB one; nothing about the mechanisms differs. The declared size is 1048576 because five
of the eight spatial cases cannot run below it, and the arms must share one value. Both readings are
real and the smaller one is recorded here rather than left out.

Both binaries were verified before the run: built after the change, carrying the embedded image
(`repro_memfs`) rather than `/root/corpus/*.db`, at the declared 1048576-byte arena, and matching
by sha256 in the guest.
