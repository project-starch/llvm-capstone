# SQLite memory-safety bugs against the nested-allocator blind-spot thesis

*Table: `sqlite-bugs.csv` — 396 entries, 2000-2026. Rejected non-bugs are kept
separately in `sqlite-bugs-excluded.csv` so nobody researches them twice. The
mechanism this table tests is documented, with file:line evidence, in
`sqlite-arenas.md` — read that first. Sibling case study: `mruby.md`.*

## What the table is

Every SQLite memory-safety defect that could be found from public primary
sources, classified by two properties: **which allocator the affected object
came from** (`alloc_arena`) and **what kind of code it lives in**
(`code_category`).

**Read `code_category` first — 396 rows is not 396 SQLite library bugs.**

| `code_category` | Rows | What it means |
|---|---|---|
| `core` | 227 | `src/` — the library every consumer links. The real target. |
| `extension` | 120 | `ext/` — needs a `-DSQLITE_ENABLE_*` flag; shipped, but optional |
| `cli` | 15 | the `sqlite3` shell, a separate program, not the library |
| `not-sqlite` | 10 | filed against SQLite, actually a third party's defect |
| `c-api` | 8 | only reachable by a C caller passing bad arguments |
| `duplicate` | 7 | a second CVE id for a bug already in the table |
| `survey` | 5 | **not bugs** — recorded negative results |
| `demo` | 2 | `ext/misc` demo code, built for the test suite only |
| `sqlite2` | 2 | the SQLite 2 codebase, unrelated to 3.x |

**374 distinct SQLite defects** once survey, duplicate and not-sqlite rows are
taken out; **227 of them in the core library**.

| Source | Entries | What it contributes |
|---|---|---|
| sqlite.org CVE page + changelog | 54 + 18 | the project's own assessment, verbatim |
| Fossil check-ins with no CVE | 240 | the bulk of the real defect history |
| NVD / MITRE / cvedetails | 92 CVEs total | pre-2019 coverage, incl. 13 Apple-assigned 2017 CVEs that keyword search does not return |
| OSS-Fuzz + dbsqlfuzz | 16 | ASAN crash states and downloadable testcase ids |
| Vendor advisories (Talos, Magellan) | 12 | root cause and PoCs; Magellan 2.0 split per bug |
| Component survey | 58 | per-subsystem coverage, including negative results |
| Distro trackers (Debian/RH/Ubuntu/SUSE) | 64 | fix commits, introducing commits, and reachability verdicts |

348 of 396 entries carry a fix commit, and 29 also name the commit that
INTRODUCED the bug -- a version window that makes building an affected tree
straightforward rather than a bisect.

## The headline result

Crossing the two columns is the result, and the separation is almost total:

| | lookaside (blind) | direct-malloc (covered) | other |
|---|---|---|---|
| **core library** (227) | **134** (37 verified + 97 probable) | 2 | 91 |
| **extensions** (120) | 2 | **109** | 9 |

**The bugs CHERI catches in SQLite are the extension bugs; the ones it misses
are the core-engine bugs.** Not because the core is worse code — because the
core allocates from a pool and the extensions call malloc. The 91 "other" core
rows are mostly `UNKNOWN`: fossil check-ins whose one-line message names no
file. They are unclassified on purpose, not silently assumed.

## Bug classes, and why two of them matter more than their count

| Class | Entries |
|---|---|
| heap out-of-bounds read | 118 |
| uninitialised read | 85 |
| use-after-free | 49 |
| null deref | 22 |
| integer overflow | 22 |
| stack overflow | 20 |
| heap overflow write | 15 |

Two observations that change how the table should be read:

1. **The 85 uninitialised reads are a second, independent blind spot.** CHERI
   checks in-bounds and tag-valid; it does not check *written*. Worse, a freed
   lookaside slot is **not scrubbed in release builds** (the trashing memset is
   `#ifdef SQLITE_DEBUG`), so a recycled slot can hand back a still-valid
   capability rather than untagged junk. Details in `sqlite-arenas.md`.
2. **The 118 OOB reads are mostly corrupt-database-file overreads** of one to
   nine bytes, in FTS5/sessions/zipfile — i.e. `direct-malloc` territory. This
   is the class CHERI *should* catch cleanly, and it is the natural positive
   control for any measurement: if a harness does not trap these, the harness is
   broken, not SQLite.

## Reproduction leads, ranked

1. **CVE-2019-5018** — window-function UAF, RCE-rated, `Window` object in
   lookaside. Plain SQL, Talos advisory has the PoC. The flagship candidate.
2. **CVE-2021-20227** and **CVE-2020-13630** — read-after-free via a HAVING-0
   subquery, and FTS3 `snippet()` UAF. Single SQL statements, fix commits known
   (`30a4c32365`, `0d69f76f08`). These two are a matched pair: one lookaside,
   one direct-malloc, same bug class — exactly the differential the study wants.
3. **CVE-2024-0232** — JSON parser UAF. `sqlite3DbRealloc` growth means the
   verdict flips with payload size; see the directed experiment in
   `sqlite-arenas.md`.
4. **CVE-2025-3277 / CVE-2020-13434** — `concat_ws()` and `printf()` integer
   overflows, pure SQL, real heap. Should be caught; good controls.
5. **CVE-2025-29088** — three-line C repro against `sqlite3_db_config` with
   out-of-range lookaside arguments.
6. The four OSS-Fuzz issues with downloadable ClusterFuzz testcases
   (`instrFunc`, `jsonTranslateBlobToText`, `sqlite3BtreeIndexMoveto`,
   `jsonbPayloadSize`).

## Honest limits

- **Reachability is recorded where a distro established it**, in `notes`, for 9
  rows: Debian marks CVE-2022-35737 unimportant because its builds omit
  `-DSQLITE_ENABLE_STAT4`, CVE-2025-70873 because the zipfile extension is not
  built, and CVE-2019-8457 "ignored" because the code is present but unused.
  That signal decides whether a row is reachable in OUR build at all, and it
  should be checked against the Capstone build flags before any repro work.
- **`verification` says how each row was reached.** `arena-verified-by-component`
  and `arena-inferred-from-file` are inferences from the allocator survey, not
  from tracing that specific object. No row here has had its object traced
  individually yet; that is the next work.
- **PoC availability is worse than the entry count suggests.** SQLite's public
  OSS-Fuzz reports have all but stopped since the in-house `dbsqlfuzz` took
  over, and reproducers largely live in the proprietary TH3 suite or are folded
  anonymously into `test/fuzzdata*.db`. Most fossil rows will need a repro
  reconstructed from the fix diff.
- **Two incomplete-fix chains are flagged**: CVE-2019-19926 exists only because
  the CVE-2019-19880 fix was partial, and CVE-2020-13871's first fix was
  superseded by `44a58d6cb135a104`. A repro built against the first fix would
  measure the wrong thing.
- **OSS-Fuzz is structurally blind to everything under `ext/`.** `test/ossfuzz.c`
  includes only `sqlite3.h` and registers no extension, and the target is built
  with a plain `configure --shared=0`, so nothing in `ext/misc`, `ext/expert`,
  `ext/session` or `ext/rbu` is linked in. Corroborated by check-in counts:
  `ossfuzz` matches 94, `ossfuzz rtree` and `ossfuzz geopoly` match 0. So "no
  OSS-Fuzz issue for component X" is worth NOTHING as evidence for any
  extension, and must not be read as reassurance.
- **Enumeration was partly blocked.** The OSS-Fuzz list/search API silently
  ignores queries, so only individually-resolved issues are present. A
  brute-force scan of the buganizer id range (~75k ids, ~25 min) is feasible and
  would enumerate the rest; not run, because it is a lot of traffic at a third
  party and that is a decision to take deliberately.
- **`blade.tencent.com` refused connections**, so the two Magellan advisories are
  cited but unfetched; the cluster was reconstructed from Red Hat and upstream
  commits instead.
- Two claims that arrived without evidence were **rejected rather than recorded**:
  an alleged carray overflow whose offered fix hash returns "No Such Object" at
  sqlite.org, and an RBU claim with no hash, URL or CVE. Both rejections are
  noted in-table so nobody re-litigates them from memory.
- CVE-2019-19880 is a **type confusion**, not the null-deref it is usually filed
  as, which makes it the single most relevant entry here for a capability study.
- 16 entries were **excluded as non-bugs** — six AI-hallucinated 2026 CVE ids that
  sqlite.org calls unreproducible, plus third-party/application bugs filed
  against SQLite's name. They are in `sqlite-bugs-excluded.csv` with the reason.

## Reproducing the table itself

    python3 merge-cves.py        # merge per-source JSON fragments, dedupe by id
    python3 classify-arena.py sqlite-bugs.csv   # assign arenas from the file map
    bash make-xlsx.sh --upload   # regenerate the spreadsheet view, push to Drive

The CSV is the source of truth and is what gets reviewed; the `.xlsx` is a
derived view that lives in Google Drive and is gitignored here.
