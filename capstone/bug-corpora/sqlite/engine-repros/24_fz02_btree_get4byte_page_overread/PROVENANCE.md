# fz02 -- fz02

Upstream fix `FZ02`. Collected in round R2 (fuzz-corpus diff).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
fz02 -- R2 fuzz-corpus diff image, trigger re-derived 2026-10-03.
 *
 * Host ASan on stock 3.22.0 (x86):
 *     sqlite3Get4byte:29655, heap-buffer-overflow
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

## Review 2026-10-16 -- the CheriBSD cell becomes a detection, and the ASan citation is wrong

### The four readings

| reading | result |
|---|---|
| host ASan | **re-run**: `heap-buffer-overflow`, 253 bytes after a 776-byte region, `statDecodePage <- statNext <- statNext` -- identical to the recorded bundle, and **not** in `sqlite3Get4byte` |
| virtual-malloc | **MISSED**, `RETURNED` -- against the physical `spatial` row's `cause 5 load out of bounds` |
| virtual-nested-pools | **CAUGHT**, `cause=28` in `sqlite3Get4byte` |
| cheribsd-revocation | **CAUGHT**: `SIGPROT(34) si_code=1 PROT_CHERI_BOUNDS`, pc resolving to `sqlite3Get4byte+16` |

### Why the CheriBSD cell changes, and why it is not about the mechanism

The nine spatial cases were all recorded `not-detected` on 2026-10-06, and those readings were
taken **without their corrupt database images present** -- the silent failure found while
reviewing case 21 and fixed in `e9199203`: `sqlite3_open` on a missing path creates an empty
database, the first statement returns *no such table*, and the case prints `RETURNED rc=0`,
which the runner scores as a clean completion. With the images restored, on the same binaries
and the same arm, this case faults. The 2026-10-06 TSV records it as `PASS`.

`si_code=1` is `PROT_CHERI_BOUNDS`: a bounds violation on a **valid** capability. That is
categorically different from the `si_code=2` tag faults that every false positive in this review
turned out to be -- those were dereferences of forged pointers, this is a real read outside real
bounds.

Probe site 64 shows why a mechanism whose bounds are at `malloc` granularity can see this case
at all: the over-read runs far enough to leave the **whole 256 KiB arena**. On CheriBSD the
witness reports `kind=CAP-OOB` with `cb=<arena>+262144` and `ca` past its end. An over-read that
stays inside the arena is invisible there, which is why the other spatial cases are silent on
that arm -- and is exactly the distinction case 27 makes.

### The ASan citation does not hold

`live_proof` and the CheriBSD copy's header both state that the re-derived trigger reproduces as
`heap-buffer-overflow in sqlite3Get4byte at sqlite3.c:29655`, and the CheriBSD header adds "It
reproduces at the SAME source line as the log" as its acceptance test. **That test is not
satisfied.** Host ASan, re-run on a restored runtime and agreeing with the 2026-10-11 bundle,
reports `statDecodePage` at the free-block chain line. The reason is simple: this image trips an
earlier over-read in the same statement, and ASan aborts at the first one.

The `sqlite3Get4byte` over-read is nonetheless real -- **two independent capability arms fault
there**, Sublet with `cause 28` and CheriBSD with `PROT_CHERI_BOUNDS`. What is wrong is the
claim about the oracle, not the site.

### Not a duplicate of 32, measured rather than inferred

Cases 29 and 32 report the **same ASan line**, which is what the 00-against-25 check taught to
distrust. Probe site 64 hooks all three unchecked reads in `statDecodePage` and the three dbstat
images split cleanly:

| case | free-block chain | cell-pointer walk | its own site |
|---|---:|---:|---|
| **29** | 1 call, `iOff=1027` | **613 calls, `iOff` to 52996** | plus the `sqlite3Get4byte` read both arms fault on |
| 31 | never runs | 8 calls, `iOff` to 64768 | `statDecodePage` |
| 32 | **1 call, `iOff=17234`** | 10 calls, `iOff` to 506 | `statDecodePage` |

So 29's image trips the chain, the cell walk **and** its own `sqlite3Get4byte` read; the first of
those is what ASan reports and the last is what the arms fault on. Three distinct unguarded
reads, and the stacks alone could not have separated them.

### The two base builds disagree

The physical `spatial` row reports `cause 5`, load out of bounds; the virtual-malloc build of the
same case **completes**. That is the build-dependence case 20 showed, but here in the direction
of a genuine mechanism rather than a forged pointer: base Capstone holds the arena as one
capability, so an over-read is visible exactly when it leaves the arena, and where it lands
relative to the arena end differs between builds. The published cell is `D`, from the physical
row, and it is a real bounds fault; what cannot be claimed is that it would fire on any build.

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
| virtual-nested-pools | **CAUGHT** | both controls as declared; `cause=28 pc=0x3f9506ad20` in `sqlite3Get4byte`, a declared fault site |
| cheribsd-revocation | **MISSED** | `RETURNED rc=0`; `poscontrol` exit 162 with `revocation=1` in the same session |

Reachability on CheriBSD is not inferred from the return code, which is what made the first reading
of case 25 void: probe site 64 was built into this arm's binary -- verified by `strings` to carry the
site's own payload, not just the runtime's `LB_ARMED` line -- and reports `hits=608 witnesses=557`, every witnessed read `kind=OOB`.
The defect site is entered and the access happens; the mechanism is silent because its bound is the
whole arena and the over-read stays inside it.

Both binaries were verified before the run: built after the change, carrying the embedded image
(`repro_memfs`) rather than `/root/corpus/*.db`, at the declared 1048576-byte arena, and matching
by sha256 in the guest.
