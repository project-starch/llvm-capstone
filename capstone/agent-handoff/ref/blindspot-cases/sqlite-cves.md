# SQLite CVEs against the nested-allocator blind-spot thesis

*Companion table: `sqlite-cves.csv` (opens in Excel/LibreOffice). Sibling case
study: `mruby.md`. This is the INTERNAL axis — bugs inside SQLite itself. The
API-misuse axis (bindings calling the C API wrongly) lives in
`benchmarks/sqlite/cve-repros/` and is a different study; do not merge them.*

## The thesis this table tests

CHERI purecap sets bounds and revocation at the **malloc boundary**. SQLite
sub-allocates behind that boundary, so a bug whose object lives inside one of
SQLite's own arenas is a candidate the hardware naturally misses:

| Arena | Mechanism | Why CHERI is blind there |
|---|---|---|
| **Lookaside** | per-connection pool; small allocs (Expr, parse-tree, VdbeOp lists) carved as fixed slots from ONE malloc'd buffer; freed slots go to a connection-local freelist | slot reuse never calls free(), so revocation never fires; a capability derived from the pool spans neighbouring slots unless the port narrows it |
| **pcache1** | page cache; pages recycled via LRU, often carved from one configured buffer | page recycling is invisible to revocation; page-to-page overflow stays inside one allocation |
| **memsys5** | optional buddy-allocator pool over one static heap | everything is one allocation to the hardware |
| **db-heap** | sqlite3DbMalloc above the lookaside size threshold falls through to real malloc | NOT blind — ordinary CHERI coverage |
| **direct-malloc** | sqlite3_malloc64 users (FTS3/FTS5 buffers, rtree nodes, JSON) | NOT blind — ordinary CHERI coverage |

So the load-bearing column in the CSV is **alloc_arena**: the same bug class
flips between CAUGHT and BLINDSPOT depending on where the object lives. The
interesting deliverable is the set of rows that end up
**use-after-free/overflow × lookaside (or pcache)**.

## Legend (cheri_expectation)

- `BLINDSPOT-candidate` — expected to survive on CHERI purecap; measurement target.
- `CAUGHT-spatial` / `CAUGHT-tag` — bounds or tag validity should trap it.
- `CAUGHT-temporal` — trapped once revocation quarantines the region; a
  `TEMPORAL-WINDOW` row passes until then (realloc-move staleness).
- `ARENA-DECIDES` / `UNKNOWN` — verdict blocked on pinning the allocation site.
- `N/A-*` — no memory-safety differential (NULL derefs trap everywhere; logic DoS).

## Method and honesty rules

1. CVE list seeded from **sqlite.org/cves.html** (fetched 2026-09-02, the
   project's own assessment column) plus NVD/search for the pre-2019 set
   (Magellan, rtree, window functions).
2. Arena assignments marked `-CANDIDATE` are **source-reading hypotheses, not
   findings**. Every row stays `verification: unverified` until the allocation
   site is pinned in the pinned SQLite tree (3.53.3, the version the
   api-classification used) — grep the function in the CSV's `affected_site`,
   follow the object to its allocator, record file:line in this doc.
3. A `BLINDSPOT-candidate` graduates only with a repro: build the PoC against
   purecap CHERI (cheri-baseline harness), show the harness catches a plain
   malloc-boundary control in the same boot, then show the CVE case surviving.
   That is the mruby ary-delete discipline; `tests/cheri-baseline/` has the
   pattern.
4. sqlite.org dismisses several CVEs as misinformation/not-SQLite; those are
   excluded from the CSV entirely rather than carried as noise.

## Priority queue (as of seeding)

1. **CVE-2019-5018** — window-function UAF, RCE-rated, parse-tree object,
   strongest lookaside candidate. Flagship if the arena confirms.
2. **CVE-2020-13871** — same arena argument, read-only UAF.
3. **CVE-2020-15358** — OOB read where the arena genuinely decides the verdict.
4. **CVE-2024-0232** — realloc-staleness; measures the revocation window rather
   than a pure blind spot.
5. Break out the Magellan-2.0 cluster (CVE-2019-13734/13750..53) per bug.

## Build-configuration facts (checked in-tree at seeding time)

- The **Capstone domain build** disables lookaside outright
  (`build-sqlite-capstone.sh:115: -DSQLITE_DEFAULT_LOOKASIDE=0,0`) and uses
  `SQLITE_ZERO_MALLOC`. Silicon work has already met lookaside once:
  `build-sqlite-silicon.sh` carries a documented probe for
  `db->lookaside.pSmallFree already untagged`.
- The blind-spot MEASUREMENT however runs on the **CHERI baseline**, and stock
  CHERI builds SQLite with lookaside ON by default. Per the project's
  stay-close-to-defaults rule that stock configuration is the one to measure;
  do not disable lookaside there to make anything easier.

## Open questions
- Which CVEs have public PoCs that run as plain SQL? (fuzzer inputs port easily;
  the window-function ones are plain SQL from the Talos advisory.)
- pcache rows are absent so far: no public CVE clearly lives in page-cache
  memory. Worth one dedicated pass over the fossil timeline for non-CVE
  page-cache corruption tickets.
