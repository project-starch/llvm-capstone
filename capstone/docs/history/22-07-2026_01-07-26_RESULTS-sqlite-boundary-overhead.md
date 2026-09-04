# Host↔engine boundary overhead on SQLite (revoke-at-free, selective boundary)

**Finding.** Applying revoke-at-free temporal safety to the pointers SQLite hands
across the host↔engine boundary costs **~1% on a real mixed workload** and **≤~6% in
a pure column-scan phase** — the boundary-densest work a program can do. A protected
pointer hand-out occurs only once per **~2,850** in-domain instructions even in the
tightest scan loop, and once per **~21,000** across a full benchmark. Revoke is rare.

**Vehicles.** (1) `speedtest1` — the canonical SQLite benchmark — built native and
instruction-counted with `perf`, boundary events counted by a shim that renames only
the embedder→engine calls. (2) A bulk-scan workload run **inside a pure-capability
domain**, instruction-counted with `csrdicount` under QEMU `-icount` (the functional
proxy the paper's QEMU comparison already uses). Boundary unit cost is the silicon
borrow super-op (`21-07-2026_*_RESULTS-fpga-borrow-cost-*`).

## Operations

- **borrow** — the revoke-at-free protection for one boundary pointer hand-out
  (`sqlite3_column_text`): `mrev` (mint revocation node) + `delin` (delegate a working
  cap) + load + `revoke` (reclaim). The **added** cost over an unprotected load is
  `mrev`+`delin`+`revoke` = **171 cyc** on CVA6 silicon (the load itself is already the
  baseline work).
- **revoke** — the reclaim half, fired at row advance / statement completion
  (`sqlite3_step` / `reset` / `finalize`). Bundled into `borrow` above; frequency is
  the same order (1 per ~29 k instructions on the mixed workload).
- **B** — boundary events = column pointer hand-outs. **T** — retired instructions of
  the measured region. **Overhead** = B·171 / (T·CPI); CPI 1 is the conservative
  headline (a larger CPI only enlarges the baseline and lowers the percentage).

## Super-table

| setting | B (borrows) | T (instrs) | 1 borrow / N instr | overhead @CPI1 | @CPI1.5 |
|---|--:|--:|--:|--:|--:|
| **speedtest1 `main` size 100** (native, whole benchmark) | 794,079 | 17.01 B | 21,423 | 0.80% | 0.53% |
| **speedtest1 `main` size 50** (native) | 404,556 | 7.66 B | 18,937 | 0.90% | 0.60% |
| **speedtest1 full suite** (native) | 1,537,255 | 22.71 B | 14,771 | 1.16% | 0.77% |
| **in-domain scan, 200 rows** (QEMU `-icount`, real PureCap SQLite) | 400 | 1,145,047 | 2,863 | 5.97% | 3.98% |
| **in-domain scan, 400 rows** | 800 | 2,276,562 | 2,846 | 6.01% | 4.01% |

Boundary unit cost (silicon, CVA6): borrow super-op **171 cyc** = `mrev` 50 + `delin`
+ `revoke` 121; the unprotected load it guards is 8 cyc.

## Interpretation

The two vehicles bracket the answer. The **in-domain scan** is the ceiling: a phase
that does nothing but read columns across the boundary still spends ~2,850 instructions
of real SQLite work (VDBE step, btree traversal, column extraction — all under
pervasive capabilities) per hand-out, so one 171-cycle protection lands per ~2,850
cycles → **≤~6%**. The **whole benchmark** is the realistic case: inserts, index
maintenance, updates and computation dominate, spacing hand-outs ~7× further apart →
**~1%**. Both hold the frequency ratio steady across sizes, so the numbers are the
workload's shape, not an artifact of scale.

The boundary is defined **by the interface**: the subset of protected pointers is
exactly the ones the engine lends the host (`sqlite3_column_*`), reclaimed at the
contract point (`step`/`finalize`). Spatial safety, by contrast, is pervasive — every
pointer is a bounded capability — and the same SQLite runs **correctly** under it:
`sqlite3_open` → `CREATE`/`INSERT`/`SELECT`, transactions, a secondary index, prepared
bound statements, `UPDATE`/`DELETE`, aggregates and the sorter, `JOIN`, `GROUP BY`, and
string functions all execute in a pure-capability domain and return correct results.
Selective temporal protection at the interface adds ~1%; pervasive spatial protection
is already correct end to end.

## Reproduce

- Tier-1: `scratchpad/tier1/` (`count_wrap.c` + `speedtest1.c` from the SQLite source
  tree, `-D`-renamed boundary calls; `perf stat -e instructions`).
- In-domain: `benchmarks/sqlite/sqlite_boundary_cost_domain.c`, built via
  `build-sqlite-capstone.sh DOMAIN_SRC=…` (knob `-DBOUNDARY_ROWS`), run through
  `run-domain-smoke.py` with `-icount`.
