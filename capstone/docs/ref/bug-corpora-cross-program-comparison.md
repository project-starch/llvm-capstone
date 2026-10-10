# Bug corpora across programs: FFmpeg, tshark and memcached against the other six (2026-10-10/11)

**What this is.** The cross-program half of the whole-corpus audit of 2026-10-10. Its run-by-run record
(pre-registrations, outcomes, retractions) is
`docs/history/10-10-2026_23-30-00_whole-corpus-audit-cross-program.md`.
- **Scope:** all 28 corpora and their 327 declared cases. Of those, 245 defects are on the
  all-program board, across nine programs; the rest are corpora listed off the board with a reason,
  duplicates, and the cross-language set.
- **Method:** the three target programs (FFmpeg, tshark, memcached) are compared with CPython, httpd,
  mruby, Perl, PostgreSQL and SQLite.
  - *How:* what each program's corpus measures, with which arms, under which judge.
  - *What:* which defect shapes each corpus holds.

The board tables themselves stay in their generated homes:
- `docs/ref/spatial-vs-temporal-three-programs.md` sections 0, 0b and 0c (the three programs, physical
  and virtual);
- `docs/ref/three-columns-all-programs.md` (all nine programs, physical and virtual).

## 1. The corpora, side by side

Read from each `corpus.json` and its `case.json` files.
- **On the board:** the case's rows on the all-program board. A duplicate, and the corpora listed in
  `NOT_ON_BOARD` with their reasons, read 0.
- **Arm counts:** the corpus's `required_arms`.
- **Derived:** the arms whose verdicts `tools/derive-verdicts.py` writes from `tools/verdicts.py`
  bundles. Every other arm's verdict is a `case.json` field that cites its result record.

| program | corpus | cases | on the board | physical arms | virtual arms | arms derived from `verdicts.py` bundles |
|---|---|---:|---:|---|---|---|
| CPython | allocator-boundary-repros | 32 | 0 | 3 | -- | -- |
| CPython | pymalloc-repros | 20 | 20 | 5 | virtual-malloc, virtual-nested-pools | virtual-malloc, virtual-nested-pools |
| FFmpeg | carved-repros | 13 | 13 | 11 | virtual-malloc, virtual-nested-pools | virtual-malloc, virtual-nested-pools |
| FFmpeg | plain-heap-repros | 25 | 24 | 9 | virtual-malloc | virtual-malloc |
| FFmpeg | plain-temporal-repros | 13 | 13 | 7 | virtual-malloc | virtual-malloc |
| FFmpeg | plane-repros | 1 | 1 | 11 | -- | -- |
| FFmpeg | pool-repros | 4 | 4 | 8 | -- | -- |
| FFmpeg | subobject-repros | 9 | 9 | 10 | virtual-malloc | virtual-malloc |
| httpd | apr-pool-repros | 1 | 1 | 7 | -- | -- |
| httpd | bucket-repros | 8 | 8 | 7 | -- | -- |
| memcached | allocator-repros | 9 | 9 | 7 | virtual-malloc, virtual-nested-pools | virtual-malloc, virtual-nested-pools |
| memcached | plain-heap-repros | 9 | 9 | 9 | virtual-malloc | virtual-malloc |
| memcached | plain-temporal-repros | 3 | 3 | 7 | virtual-malloc | virtual-malloc |
| mruby | gc-slot-repros | 0 | 0 | 0 | -- | -- |
| mruby | release-differential | 23 | 23 | 4 | -- | -- |
| Perl | release-differential | 11 | 11 | 4 | -- | -- |
| PostgreSQL | c-repros | 5 | 5 | 2 | virtual-malloc | virtual-malloc |
| PostgreSQL | mmgr-repros | 5 | 5 | 6 | virtual-malloc, virtual-pg-pools | spatial, sublet, virtual-malloc, virtual-pg-pools |
| PostgreSQL | sql-repros | 9 | 9 | 3 | virtual-malloc, virtual-pg-pools | virtual-malloc, virtual-pg-pools |
| SQLite | capi-repros | 0 | 0 | 1 | -- | -- |
| SQLite | engine-repros | 33 | 33 | 3 | virtual-malloc, virtual-nested-pools | virtual-malloc, virtual-nested-pools |
| tshark | plain-heap-repros | 12 | 12 | 10 | virtual-malloc | virtual-malloc |
| tshark | plain-temporal-repros | 10 | 10 | 7 | virtual-malloc | virtual-malloc |
| tshark | wmem-repros | 23 | 23 | 6 | virtual-malloc, virtual-nested-pools | virtual-malloc, virtual-nested-pools |

**Two ways of recording a verdict.**
- **Bundles.** Since 2026-10-10, CPython pymalloc, PostgreSQL and SQLite record their virtual arms
  as `verdicts.py` bundles, which one judge re-reads. Since this audit, ten of the three programs'
  twelve corpora do too:
  - FFmpeg pool and plane have no virtual recipe;
  - wmem's bundles are the 23-case R9b run.
- **`case.json` fields.** Every physical arm, and httpd's, mruby's and Perl's corpora entirely, keep
  hand-recorded `case.json` verdicts that cite a result record.
- **Agreement.** Both forms pass the same `check-corpus.py`, and the strict board treats them the same.
  But only the bundles are re-judged automatically when the judge or a configuration's controls change
  (`derive-verdicts.py --check`).

## 2. Population

The all-program board's case counts, by kind and allocator axis. "Unsplit" is mruby's and Perl's
whole-program arms, which do not separate the allocator layer.

| program | temporal, nested | temporal, plain | spatial, nested | spatial, plain | unsplit (whole program) | total |
|---|---:|---:|---:|---:|---:|---:|
| FFmpeg | 4 | 13 | 14 | 33 | 0 | 64 |
| tshark | 14 | 10 | 9 | 12 | 0 | 45 |
| memcached | 5 | 3 | 4 | 9 | 0 | 21 |
| CPython | 20 | 0 | 0 | 0 | 0 | 20 |
| httpd | 9 | 0 | 0 | 0 | 0 | 9 |
| mruby | 0 | 0 | 0 | 0 | 23 | 23 |
| Perl | 0 | 0 | 0 | 0 | 11 | 11 |
| PostgreSQL | 5 | 0 | 9 | 5 | 0 | 19 |
| SQLite | 25 | 0 | 8 | 0 | 0 | 33 |
| **Total** | **82** | **26** | **44** | **59** | **34** | **245** |

The three target programs hold 130 of the board's 245 defects, and 50 of its nested ones:
- FFmpeg 18;
- tshark 23;
- memcached 9.

Of the other six, CPython, httpd, PostgreSQL and SQLite are nested-only corpora (pymalloc, APR,
memory contexts, memsys5), plus PostgreSQL's plain c-repros.

## 3. Defect shapes in nested allocators

The audit compared shapes, not just counts. Six shapes recur in nested-allocator defects:

| shape | what it is |
|---|---|
| S1 | a re-entrant callback mutates storage inside a nested allocator |
| S2 | a double free into a nested allocator |
| S3 | use of a destroyed or foreign allocator instance |
| S4 | corruption of the allocator's in-band free list |
| S5 | a realloc or grow inside a nested allocator leaves a stale pointer |
| S6 | a vacated slot is reused with no allocator event |

How the table was built:
- a keyword pass over every case's own `title`, `shape` and `lifetime_ender`;
- then each hit kept only if its title confirms the shape.

So the table lists examples, and it is not exhaustive.

| shape | instances on the board (other programs) | FFmpeg / tshark / memcached |
|---|---|---|
| S1 | CPython pymalloc 00 (re-entrant `itertools.groupby`); mruby 03, 06, 10, 12 (a Ruby callback frees or rehashes storage the caller still holds) | none before the audit; the hunt found none |
| S2 | PostgreSQL mmgr 00 (a tuple freed twice through a stale array entry) | **tshark wmem 22, added by this audit** (USB HID, `c702b44a01`) |
| S3 | httpd apr-pool 00 (`mod_watchdog` reuses a pool it destroyed) | none; tshark's allocator-mismatch rows are vetted, not built (ASan-visible) |
| S4 | (PostgreSQL mmgr 04 reuses a slab free-list slot: reuse, not corruption) | **tshark wmem 22**: the double free corrupts BLOCK's recycler, which loses a chunk |
| S5 | CPython pymalloc 10 (`bytearray` search after a realloc moved the block); Perl 10 (`SvGROW` frees the buffer under a write) | FFmpeg plain-temporal has realloc cases on the PLAIN heap; none nested |
| S6 | mruby 05 (a string swept and its GC slot reused) | none that separates the arms (FFmpeg's DPB-slot candidates are NULL dereferences) |

The hunt behind the right-hand column, with its rejected lists and blocked candidates, is
`docs/ref/nested-shape-hunt-2026-10.md`. In short:
- FFmpeg's pool use-after-free fixes are overwhelmingly on plain `av_malloc` state;
- memcached's S1-S5 material sits in the page mover, the proxy and extstore, all off in the port;
- Wireshark had two S2 candidates, and the simpler one is now case 22.

## 4. Are the programs scored the same way?

Mostly. Four places differ, and each is stated where the cell is read.

**1. Column 1: CHERI, with quarantine counted as caught.** The "held" rule applies only where a freed
chunk reaches `free()`: the plain temporal cases.
- Every nested temporal case reads 0 on stock CheriBSD, in every program, because the nested allocator
  never frees to libc on the defect's path.
- mruby's and Perl's CHERI catches are not separated from bounds, as
  `three-columns-all-programs.md` says. Perl's five fault with revocation off. mruby's are "a strict
  subset of our 16" bounds-only catches.
- So a CHERI temporal catch means a lifetime catch only for the three target programs, CPython,
  PostgreSQL and SQLite.

**2. Column 2: the system allocator under a stock nested allocator.**
- **Physical:** for the three programs, httpd and Perl, it is Sublet as the system allocator:
  `sublet-malloc`, `sublet` on a plain case, and Perl's `sysalloc-sublet`. mruby's column 2 has no run
  on the board.
- **Virtual:** for CPython, PostgreSQL and SQLite, it is virtual mallocng (`virtual-malloc`).
- **Since this audit:** the three programs have both. Their virtual and physical column 2 agree cell
  for cell where both exist:
  - wmem 0/23;
  - memcached allocator 1/9;
  - FFmpeg carved 0/13.

  In every program the mechanism is per-object bounds and retirement in malloc, which a nested
  allocator's chunks hide.

**3. Column 3 for a plain case.**
- **Physical:** the three programs run their port linked and live (`sublet-full`, a non-interference
  check).
- **Virtual:** the virtual profile reuses the column-2 run, because there is no nested allocator on a
  plain path.
- **mruby and Perl:** whole-program arms, unsplit.

**4. NO-READING.**
- Bundle corpora use `verdicts.py`'s closed reasons.
- `case.json` corpora use the board's word mapping (`catch-tables.py` `VERDICT_WORDS`). Since the
  audit, an unknown word is an error, not a miss.
- Both print `(+N ?)` instead of averaging a hole away.

**Attribution: where the programs differ in rigour.**
- **The three programs' catches** must be at a labelled probe or at a site declared before the run
  (`fault_sites`), and since this audit that holds on every arm:
  - the subobject waiver is removed;
  - memcached 08 is attributed at the scan;
  - the supervised CheriBSD runs resolve sites, including in libc;
  - the PoisonCap runner resolves each case's label.
- **SQLite and PostgreSQL** attribute most virtual catches by function. See item 1 under section 6.

## 5. What this audit changed

On dev, landed as separate commits: attribution, virtual arms, the new case, and this document.

**Attribution, no verdict change** (R1-R5, R2c, R3c):
- memcached 08's CHERI catch is at `mc_case_body+0x1e8`, the defective scan (`case.c:66`);
- the subobject arm's seven Capstone catches stand at declared sites, on images byte-identical to the
  earlier run's;
- the supervised CheriBSD runs, subobject and plane, name their sites;
- wmem PoisonCap 13-21 resolve their own labels;
- pool column 2 has an in-boot Sublet-heap control.

**Virtual columns for the three programs** (R6-R8, R9b): 10 of 12 corpora on both virtual columns,
each as predicted.
- **wmem and memcached** need a process build of their Sublet ports:
  - `WM_SUBLET` and `MCP_SUBLET`, from the unmerged branch `virtual-capstone-bug-corpora`;
  - the physical images are byte-identical before and after (185 and 134);
  - the corpora's `driver.c` files are untouched.
- **FFmpeg pool and plane** are not measured. Their cases are fixtures of the FFmpeg app port, whose
  pool variants `build-virtual.sh` refuses.

**A new case, wmem-repros 22** (R9, 11 of 11 as predicted). It is the three programs' first double
free into a nested allocator.
- It is caught by the chunk port at the allocator's handback, and by PoisonCap mode 1.
- Every allocator-granular mechanism misses it:
  - CHERI with libc;
  - ASan;
  - the region-granular port;
  - Sublet as the system allocator;
  - virtual mallocng.
- Mode 0 of the chunk port also refuses it, through the port's own bookkeeping (`WM return=260`).
  That is a property of the port, not of a capability.

**Corrections to other programs** (only where a record contradicted itself):
- Perl 03 moved from MISSED to NO-READING. Its reason was then retracted and corrected on dev
  (76bc5a72f2a5).
  - The trigger's only check is an unconditional `pass()`, so it cannot show the stale slot is absent.
  - The cell stands, because nothing observes reach.
- PostgreSQL sql-repros 03's record and its results README were made to agree.
- `catch-tables.py` names unknown verdict words and hidden NO-READING cells.
- `SCHEMA.md` and `three-columns-all-programs.md` were brought up to date (e9e492b42a89).

## 6. Open questions for other corpora's owners (recorded, not changed)

1. **`fault_sites` committed with the bundles they attribute.**
   - The cases: SQLite engine-repros declares sites in 21 cases and PostgreSQL in 6 (c-repros 2,
     sql-repros 3, mmgr-repros 1).
   - Each was committed in the same commit as the virtual bundle that resolved the fault there
     (65d46bbc49d1, 0f257e5af3f0), so history cannot show that it was declared before the run.
     `SCHEMA.md` requires a site to be declared from source before the run.
   - In those four corpora, 32 virtual CAUGHT cells are attributed by function. How many rest on those
     declarations, rather than on the runner's own consumer lists, is for the owners to establish.
   - Running the cases again under a pre-registration would settle it.
2. **mruby's and Perl's temporal catches are not separated from bounds**, and Perl's five CHERI
   catches fault with revocation off.
   - Until a bounds-only arm carries a verdict, their temporal columns cannot be compared with the other
     seven programs' lifetime columns.
3. **CPython allocator-boundary-repros.** Its README says 11 of its 32 cases "re-measure"
   pymalloc-repros' issues through the interpreter. If the corpus is placed on the board, those 11
   would count defects twice.
4. **PostgreSQL sql-repros 03 on `virtual-malloc`.**
   - Its NO-READING was judged against the 64-variant control, which PR #215 showed crosses the uint16
     threshold itself.
   - The corrected 48-variant control has not been run on virtual. That needs the virtual PostgreSQL
     image and cluster fixture, which this host does not have.
