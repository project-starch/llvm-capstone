# SQLite testing on capability silicon: what exists, what fits, what to run next

*Proposal for the project lead, 2026-09-07 (board lane). Audited the same day by `claim-auditor`;
its twelve corrections are folded in and marked **[audit]**. Sources: https://www.sqlite.org/testing.html
(numbers quoted from the page), the SQLite 3.53.3 source tree (`test/` inventory read locally, corpus
counts recomputed), and this project's own runners. Status of the corpus that motivated this: the
seven-file SQLLogicTest set is identical to native on silicon as of today
(`ref/fpga-silicon-measurements-for-paper.md` §7c; rows B8, sw23–sw29).*

## 1. What the SQLite project itself runs (from testing.html)

| suite | what it is | size | public? | harness | fits a capability domain? |
|---|---|---|---|---|---|
| **TCL tests** | the primary suite: scripts driving the C API through TCL bindings | 51,445 test cases in 1,390 scripts per the page (1,189 `test/*.test` files in the 3.53.3 tree, 20.8 MB); "veryquick" subset 304.7k instances | yes | TCL interpreter + 27 KSLOC of C glue | **no**: only 40 of the 1,189 files use `:memory:` and never `test.db` **[audit]**; the rest need a file VFS, and the interpreter is a ~200 KSLOC port |
| **TH3** | 100 % branch + MC/DC coverage of the deployed configuration; built for embedded targets without TCL | 50,362 cases, ~2.4 M instances | **proprietary** | pure C, published interfaces only | would fit perfectly — and is not available |
| **SQL Logic Test** | cross-engine SQL correctness | 7.2 M queries, 1.12 GB | yes | a few hundred lines of C (our `slt_runner.h`, 650 lines) | **yes** — 10,807 records done; the rest is time (§2) |
| **dbsqlfuzz** | structure-aware fuzzer mutating SQL and database file together | ~10⁹ cases/day | proprietary | libFuzzer | no |
| **fuzzcheck** | replay of the "interesting" historical fuzz cases (AFL, OSS-Fuzz, dbsqlfuzz) | 3.53.3: 8 corpus files, 65.4 MB: **35,932 SQL scripts** (20.1 MB of text, median 190 B, max 164 KB) + **10,461 database images** (median 2.8 KB, p99 6 KB, one 254 KB outlier) | yes, ships in `test/` | plain C (`fuzzcheck.c`, 2,763 lines). **[audit]** It runs cases two ways: 749 "combined" cases (all in `fuzzdata8`) through `sqlite3_deserialize` (`fuzzcheck.c:1369`); the other 35,183 through its **own in-memory file VFS** (`inmemVfsRegister`, `:1699-1714`; `sqlite3_open_v2("main.db", …)` at `:2649-2657`), images opened writable with a rollback journal | **yes, as a re-implementation** (§3A), not as a port: the domain has no file VFS (its default VFS returns `SQLITE_CANTOPEN`, `sqlite-vfs-skeleton/capstone_sqlite_vfs.c:112`) |
| OSS-Fuzz / AFL | continuous fuzzing | — | external | libFuzzer / AFL | no (needs the engine); its harness `ossfuzz.c` is 206 lines and its *corpus* is what fuzzcheck replays |
| speedtest1, mptester, threadtest3 | performance / multi-process / multi-thread stress | — | yes | C, but processes/threads | speedtest1 already runs here; the other two need an OS |
| anomaly tests (OOM, I/O error, crash) | fault injection via `sqlite3_config(MALLOC)` and an instrumented VFS | — | in TCL/TH3 | VFS + malloc hooks | OOM injection **would** fit (§3C); I/O-error and crash tests need a VFS with files |

**The page's framing matters for us:** TH3's 100 % branch coverage is measured "in an as-deployed
configuration". **[audit — this paragraph was wrong in the first draft, and the project had already
made and corrected the same mistake once (`build-slt-native.sh:18-21`).]** The deployed configuration
is the always-active `SQLITE_DEFINES` list in `benchmarks/sqlite/build-sqlite-capstone.sh:94-118`:
`SQLITE_OS_OTHER`, `THREADSAFE=0`, `TEMP_STORE`, `ZERO_MALLOC` + `ENABLE_MEMSYS5`, `UNTESTABLE`,
`DQS`, `DEFAULT_LOOKASIDE`, `DEFAULT_MEMSTATUS`, and **sixteen `SQLITE_OMIT_*`**: LOAD_EXTENSION,
LOCALTIME, MMAP, WAL, SHARED_CACHE, TEMPDB, AUTOINIT, COMPILEOPTION_DIAGS, **FLOATING_POINT**, UTF16,
INCRBLOB, GET_TABLE, DEPRECATED, **EXPLAIN**, FOREIGN_KEY, **JSON**. The fourteen-define
`SILICON_TRIM` list in `build-sqlite-silicon.sh:987-1002` (progress callback, datetime, the
LIKE/OR/BETWEEN optimisations, …) is **gated off** unless `SQLITE_TRIM=1` (`:1020`) and was not
active in any board result. So: no floating-point literals, no JSON, no foreign keys, no EXPLAIN,
no UTF-16 — any suite we run measures *that*, and any coverage claim must say so.

## 2. What we have, precisely

- **SLT, seven files, identical to native on silicon** (this week). One draw each. Runner
  `benchmarks/sqlite/slt/slt_runner.h`, `sqlite3_open(":memory:")` (`:306`), file delivered whole in
  the shared payload region (input in the top half; 1/2/4 MiB region classes; memsys5 heap 256 KiB by
  default, 1 MiB or 2 MiB per class — a 2 MiB heap only fits with the stack cut to 1 MiB).
- Silicon rate **[audit: a range, not a number]**: `select3` 3,351 records in ~300 s = 11.2 records/s;
  `select4` 3,857 records with 1,025 inserts in ~4,680 s = 0.82 records/s. The remaining upstream SLT
  tree (~7.2 M queries) is therefore between ~8 days and ~3 months of pure execution, plus one boot
  (2–3 min, one domain) per file — not "a week". A sampled slice is a campaign; the tree is not.
- The domain has **no VFS files**; the only alternatives are `:memory:` and `sqlite3_deserialize`.
  `sqlite3_deserialize` is compiled in (symbol present in the campaign image; `OMIT_DESERIALIZE` set
  nowhere) but **has never been called in this domain** **[audit]**; it internally runs
  `ATTACH x AS %Q` (`sqlite3-capstone.c:56181`), which works only because `OMIT_ATTACH` is absent.
  First QEMU step of any plan below is to call it once.

## 3. The options, ranked by evidence-per-boot

**A. fuzzcheck corpus replay in the domain — recommended next, as a re-implementation.** The
corpus's cases fit our delivery model: SQL scripts (median 190 B) and small database images (median
2.8 KB). What we would build is **not** fuzzcheck: a runner of `slt_runner.h`'s size that, per case,
opens `:memory:` (scripts) or `sqlite3_deserialize`s the image with the image left in the region
(`mFlags = 0`, no heap copy; `SQLITE_FULL` on growth — a documented departure from upstream, whose
`inmem` VFS gives the image a writable file with a rollback journal) and runs the SQL under
`SQLITE_LIMIT_VDBE_OP = 25000` (`ossfuzz.c:156`) plus a **deterministic per-statement step budget**
— fuzzcheck's own cutoff is wall-clock (`fuzzcheck.c:859-863`) **[audit]**, which on silicon two
orders of magnitude slower than the oracle would fire on a different set of cases on each side and
make result codes machine-speed-dependent. Oracle: the native x86 run of the identical runner,
comparing **result code and row count** per case (row count, not values: order-insensitive). Two
oracle hazards to close first: `random()`/`randomblob()` appear in 6–17 % of the SQL sets and the
domain's `SQLITE_UNTESTABLE` blocks fuzzcheck's PRNG-seed control, so those cases are excluded and
counted; and the `xRandomness` VFS hook differs. Cases are bundled into region-sized containers
(length-prefixed; ~900 scripts or ~200 images per 512 KiB half-region by the measured sizes).
**Bundle size must come from a TIME budget, not bytes** — the corpus is timeout-bounded upstream
precisely because some cases are pathological — so the first step is native `fuzzcheck --timer` over
`fuzzdata5`/`fuzzdata7` to get the per-case time distribution and scale it by the SLT-measured
slowdown; until then the boot count is **unresolved** (the first draft's "~20 boots" had no basis).
Large images need the 1 MiB-heap class: memsys5 rounds allocations to powers of two and the page
cache for a 254 KB image exhausts 256 KiB.

**The vacuous-match share, measured [audit].** The deployed build removes features the corpora
lean on. Share of each SQL set touching at least one removed feature (regex over the text; JSON,
float literals — rejected by the tokenizer under `OMIT_FLOATING_POINT` — datetime, EXPLAIN, UTF-16,
foreign keys):

| set | cases | removed-feature share | note |
|---|---|---|---|
| fuzzdata1 (SQL fuzz) | 9,917 | 27 % | |
| fuzzdata2 (AFL 2015) | 9,959 | 26 % | |
| fuzzdata4 (JSON1) | 2,575 | **71 %** (55 % pure JSON) | **drop** |
| fuzzdata5 (OSS-Fuzz) | 8,834 | 19 % | first campaign |
| fuzzdata6 (UPSERT) | 3,896 | 22 % | |
| fuzzdata8 (dbsqlfuzz combined) | 749 | **97 %** (96 % float) | drop |
| fuzzdata3 / fuzzdata7 (db images) | 2,316 / 8,145 | — | **strongest target**: raw corrupt b-tree images exercising pager/btree corruption handling the OMIT list barely touches; median 1.5 KB; the one set where `deserialize` is a faithful stand-in for a file open |

fuzzcheck also registers 13 extension modules per case (`vt02`, `series`, `regexp`, …,
`fuzzcheck.c:1392-1409`), none in our build. A case that fails to parse on both sides "agrees" and
proves nothing. **Any claim must therefore report cases that reached the VDBE, not cases that
agreed** — the runner records the result code of `sqlite3_prepare` separately from execution.

**B. More SLT (sampled `index/` and `random/` families).** Zero new code, lower value per boot
than A (shapes `select1–5` already covered); keep as the periodic regression bar, one file per board
session that touches the compiler, the monitor or the bitstream.

**C. OOM injection (the anomaly test we can afford).** TCL/TH3's OOM method needs no VFS.
**[audit]** `SQLITE_CONFIG_MALLOC` *replaces* the allocator, so the working pattern is fuzzcheck's
(`:709-712`): `SQLITE_CONFIG_HEAP` first (memsys5, as today), `GETMALLOC` to capture it, then install
a delegating wrapper that fails the N-th allocation — ordering is ours because `OMIT_AUTOINIT` is
set. The oracle does **not** carry over unchanged: the native baseline is built without memsys5
(`build-slt-native.sh:44` excludes `ZERO_MALLOC`/`ENABLE_MEMSYS5`), so "the N-th allocation" is a
different allocation on each side. Either rebuild the native baseline with memsys5 and the same heap,
or run C as a no-crash/no-wedge ladder rather than a comparison. Small runner change; medium value;
after A.

**D. TCL suite — not now.** The interpreter port plus a file-backed VFS in a 1.4 MB code budget is a
project; 97 % of the scripts assume `test.db`. A translation route (extract SQL + expected results
natively, replay through A's mechanism) exists, costs about a week for its first useful family, and
overlaps A and SLT heavily.

**E. Not applicable in-domain:** dbsqlfuzz, TH3 (proprietary); mptester/threadtest (need
processes/threads); I/O-error and crash tests (need a VFS with files and durability semantics).

## 4. Decision proposed

1. **Measure before building:** native `fuzzcheck --timer` over `fuzzdata5` and `fuzzdata7` for the
   per-case time distribution; call `sqlite3_deserialize` once in the domain under QEMU. Half a day.
2. **Build runner A**, validate under QEMU against native on `fuzzdata7` (8,145 corrupt images, the
   strongest set) and `fuzzdata5` (8,834 OSS-Fuzz scripts), with a **negative control** (a bundle
   with a deliberately wrong expected code must FAIL) and the `random()` exclusion counted. Then
   silicon, one bundle per boot, `k800` first. Report: cases reaching the VDBE, cases identical to
   native, every divergence as its own row with the case id. About a day of work plus the boots the
   timing step sizes.
3. **Keep SLT as the regression bar** (one corpus file per relevant board session).
4. **Then C** (OOM ladder) on one SLT file, with the native baseline rebuilt on memsys5.
5. **Defer D; decline E.**

What this would let the paper say, if it holds: *SQLite executes N thousand of its own distilled
fuzz-corpus cases — including 8,145 corrupt database images from dbfuzz2 — inside a capability
domain on silicon, with result codes and row counts identical to native for every case that reaches
the VDBE.* What it cannot say: anything about branch coverage (TH3 or gcov on silicon, neither
available) or about I/O and crash behaviour (no VFS).

## 5. Risks, stated

- The containers and the deserialize path are new code; the negative control is not optional.
- Wall-clock is not an oracle on silicon: the step budget must be deterministic and per statement,
  and the board driver's silence budget must exceed the worst bundle — unknown until step 1.
- One draw per case; a divergence needs a second draw before it becomes a finding.
- Board time cannot start until the Q-03 firmware change is validated on the board: bundles create
  exactly the region shapes that used to spin.
- The 16 `OMIT_*` defines are the deployed build; a result on them is not a result on stock SQLite,
  and the write-up must carry the list.
