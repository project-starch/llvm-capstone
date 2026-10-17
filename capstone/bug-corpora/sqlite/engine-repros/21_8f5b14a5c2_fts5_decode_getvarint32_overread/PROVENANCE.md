# 8f5b14a5c2_0 -- 8f5b14a5c2

Upstream fix `8f5b14a5c2`. Collected in round R3 (backport filter).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
8f5b14a5c2_0 -- upstream corrupt-database regression image, backported to 3.22.0.
 *
 * Collected by the backport filter: upstream fix 8f5b14a5c2 removes code that is still
 * present in 3.22.0 (or guards a function 3.22.0 has unguarded), and its own regression
 * test ships the triggering database as a hex dump. Host ASan on stock 3.22.0:
 *
 *     heap-buffer-overflow in sqlite3Fts5GetVarint (fts5_decode)
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

## Review 2026-10-16 -- the verdicts hold; the CheriBSD arm had to be repaired first

### The defect, as an inequality

`fts5DecodePoslist` guards the **start** of each varint and not its end:

```c
static int fts5DecodePoslist(int *pRc, Fts5Buffer *pBuf, const u8 *a, int n){
  int iOff = 0;
  while( iOff<n ){
    int iVal;
    iOff += fts5GetVarint32(&a[iOff], iVal);
    ...
```

`fts5GetVarint32` may read up to five bytes from `a[iOff]`, so any `iOff` within four bytes of
`n` over-reads. Probe site 61 is that inequality: it fires when `n - iOff < 5` and prints `iOff`,
`n`, the bytes remaining and whether the first byte carries a continuation bit.

One fidelity caveat, checked in the pin: the consumer is `fts5_decode`, a diagnostic SQL
function. It is registered **unconditionally**, at `sqlite3.c:200679` inside
`sqlite3Fts5IndexInit`, not behind `SQLITE_TEST`, so it is callable by anyone with SQL access
and the reachability is real -- but the threat model is narrower than for a case on an ordinary
query path.

### The four readings

| reading | result |
|---|---|
| host ASan | `heap-buffer-overflow`, `sqlite3Fts5GetVarint <- sqlite3Fts5GetVarint32 <- fts5DecodePoslist <- fts5DecodeDoclist <- fts5DecodeFunction`; 2026-10-11 bundle, not re-run |
| virtual-malloc | **MISSED**, `RETURNED`, controls as declared |
| virtual-nested-pools | **CAUGHT**, `cause=28` in `sqlite3Fts5GetVarint32` -- CAP_OOB, the **bounds** check, which is the right code for a spatial over-read and not the `cause 25` the temporal cases turn on |
| cheribsd-revocation | **MISSED**, `RETURNED rc=0`, and reached: site 61 reports `hits=60 witnesses=26` |

No verdict changes.

### The CheriBSD arm was reading nothing, and the log did not say so

The first run of this case on that arm today:

```
8f5b14a5c2_0 open_rc=0
8f5b14a5c2_0 s1_rc=1
8f5b14a5c2_0 s1_err: no such table: t1_data
8f5b14a5c2_0 RETURNED rc=0
LB_SUMMARY site=61 hits=0 witnesses=0
```

`hits=0`: the defect site had never been reached. The cause is a gap in this arm's tooling, not
in the case. CheriBSD has a real VFS, so its copy of the case opens the corrupt image as an
ordinary file -- `#define DBPATH "/root/corpus/<tag>.db"` -- where the Capstone arm embeds the
bytes and serves them through an in-domain memfs. **Those files existed only inside the guest,
nothing in the repository created them, and `deploy-corpus.sh` begins with
`rm -rf /root/corpus`**, which this review ran today.

The failure is silent, which is what makes it expensive: `sqlite3_open` on a missing path
**creates an empty database** and returns `SQLITE_OK`, the first statement returns *no such
table*, the case prints `RETURNED rc=0`, and the runner scores a clean completion. **Nine of the
33 cases open such a file** -- 00, 25, 26, 27, 28, 29, 30, 31, 32, the whole spatial group -- so
one redeploy can turn that entire column into clean completions with nothing in the log
disagreeing.

Fixed at the root. `ports/sqlite/cheribsd/make-db-images.py` regenerates the images from the
byte arrays in the corpus case sources -- which are in git, and are the same bytes the Capstone
arm serves -- and writes them into `$WORK/out`, which `deploy-corpus.sh` already ships;
`deploy-corpus.sh` now calls it before building the tarball, so a wiped `/root/corpus` is
restored by a redeploy. Four images this arm opens have no corpus source, because those four
tags were moved out of the corpus on 2026-10-06 as not-in-scope; the tool names them rather than
failing silently.

**The campaign's own readings are not in doubt.** The arm's probe recorded
`hits=13720 witnesses=4765` for this case on 2026-10-04/05, so the images were present then.
What was missing was any way to restore them, and this review is what destroyed the guest's
copies.

### With the image back

```
8f5b14a5c2_0 s0_rc=11  (database disk image is malformed -- integrity_check, as a corrupt image should)
8f5b14a5c2_0 s1_rc=7   (out of memory)
LB_SUMMARY site=61 hits=60 witnesses=26
```

The defect site is reached and the over-read is witnessed 26 times, and the arm says nothing:
that is a genuine MISSED. The `SQLITE_NOMEM` is recorded because it would matter if the site
count were zero -- the 256 KiB arena cannot hold `fts5_decode`'s output for a 40 KiB image -- but
it happens **after** the site is reached, so it does not affect the reading.

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
| virtual-nested-pools | **CAUGHT** | both controls as declared; `cause=28 pc=0x3f96221e8c` in `sqlite3Fts5GetVarint32`, a declared fault site |
| cheribsd-revocation | **MISSED** | `RETURNED rc=0`; `poscontrol` exit 162 with `revocation=1` in the same session |

Reachability on CheriBSD is not inferred from the return code, which is what made the first reading
of case 25 void: probe site 61 was built into this arm's binary -- verified by `strings` to carry the
site's own payload, not just the runtime's `LB_ARMED` line -- and reports `hits=46 witnesses=12`, every witnessed read `kind=OOB`.
The defect site is entered and the access happens; the mechanism is silent because its bound is the
whole arena and the over-read stays inside it.

Both binaries were verified before the run: built after the change, carrying the embedded image
(`repro_memfs`) rather than `/root/corpus/*.db`, at the declared 1048576-byte arena, and matching
by sha256 in the guest.
