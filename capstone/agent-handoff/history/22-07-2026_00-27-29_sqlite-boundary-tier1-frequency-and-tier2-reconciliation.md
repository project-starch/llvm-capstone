# SQLite host↔engine boundary — Tier-1 frequency estimate + Tier-2 status reconciliation

*Date: 2026-07-22. Serves the Compatibility eval (Experiment B, selective-boundary
temporal protection) and the PI's "protect a small number of boundary pointers,
revoke is rare" question. Two outcomes: (1) a Tier-1 boundary-frequency measurement
on `speedtest1`; (2) a correction to the "Tier-2 is blocked" premise — SQLite already
runs a rich workload in a pure-capability domain today.*

## TL;DR

- **Revoke is rare — measured.** On `speedtest1` (canonical SQLite benchmark), the
  host↔engine boundary (each `sqlite3_column_text` hand-out = one borrow; `step`/
  `reset` = revoke) fires **~1 borrow per 15,000–21,000 retired instructions**
  (0.005–0.007 % of instructions). Estimated aggregate overhead **≈ 0.8–1.2 %**
  (borrow super-op = 171 cyc measured on silicon; baseline ≥ instr count, CPI ≥ 1),
  **≈ 0.5–0.8 %** at CPI 1.5. Directly supports the PI's hypothesis.
- **Tier-2 is NOT blocked (premise corrected).** The prior working assumption —
  "full SQLite faults during init at `sqlite3RegisterBuiltinFunctions`, so in-domain
  overhead can't be measured directly" — is **stale**. All 8 cap-tag-preservation
  gaps were resolved by 2026-07-03; SQLite runs **end to end in a domain** with an
  extended workload. Re-verified fresh this session (see below). The stale claim
  lived in `benchmarks/sqlite/README.md` (dated 2026-06-30); corrected.

## What was actually run

### Tier-1 (native, boundary frequency) — the number

Built `speedtest1.c` (from the full SQLite 3.53.3 source tree; the amalgamation zip
does not ship it) against the standard amalgamation, natively (x86, gcc -O2). A
boundary-only counting shim renames **only speedtest1's** embedder→engine calls via
`-D` macros (`sqlite3_column_text→w_column_text`, `…_blob`, `…_step`, `…_reset`,
`…_finalize`, `…_close`), so SQLite-internal calls are not counted. Total retired
instructions from `perf stat -e instructions`.

| workload (`:memory:`) | borrows | instrs | 1 borrow / N instr | borrow % | added Mcyc (×171) | ovhd @CPI1 | @CPI1.5 |
|---|--:|--:|--:|--:|--:|--:|--:|
| `--testset main --size 50`  |   404,556 |  7.66 B | 18,937 | 0.0053 % |  69.2 | 0.90 % | 0.60 % |
| `--testset main --size 100` |   794,079 | 17.01 B | 21,423 | 0.0047 % | 135.8 | 0.80 % | 0.53 % |
| full default suite `--size 100` | 1,537,255 | 22.71 B | 14,771 | 0.0068 % | 262.9 | 1.16 % | 0.77 % |

- The ratio is **size-stable** (~1 borrow per ~19–21 k instr across sizes), as
  expected: both borrows and total work scale with the workload.
- Revoke frequency (`main/size100`): **1 reset-revoke per 28,829 instr**; if the
  revoke is modeled per `step` instead, 1 per 17,247 instr. Same order as borrows.
- Cost model: each protected `column_text` hand-out is one **borrow super-op**
  (`mrev`+`delin`+`load`+`revoke` ≈ 171 cyc, measured on captype-fixed CVA6 silicon
  — see `21-07-2026_*_RESULTS-fpga-borrow-cost-*`). Modeling borrow-side (`mrev`+
  `delin` ≈ 50 cyc) and revoke-side (`delin`+`revoke` ≈ 121 cyc) separately lands in
  the same ~1 % ballpark.

Interpretation: even under the **conservative** assumption that *every* column
pointer handed to the host is borrowed+revoked, boundary protection touches ~0.005 %
of instructions and adds **~1 %**. "Revoke is rare" holds.

Repro: `scratchpad/tier1/` — `count_wrap.c`, `speedtest1_counted`, build one-liners
in this note. Amalgamation at `/tmp/capstone/sqlite-src/sqlite-amalgamation-3530300`;
`speedtest1.c` from `sqlite-src-3530300.zip` (`test/speedtest1.c`).

### Tier-2 status re-verification (in-domain, QEMU)

Ran `capstone/benchmarks/sqlite/run-sqlite-memory.sh` fresh (build + QEMU). Result:
**both** `__CAPSTONE_SQLITE_EXTENDED_PASSED__` and `__CAPSTONE_SQLITE_MEMORY_PASSED__`,
with correct rows (`alpha=11/beta=22/gamma=33`). The extended in-domain workload
already exercises: transaction, secondary `INDEX`, `INTEGER PRIMARY KEY`+`REAL`,
bound prepared inserts, `UPDATE`/`DELETE`, aggregates + sorter (`COUNT`/`SUM`/`MAX`,
`ORDER BY DESC`), index-driven `WHERE`, `JOIN`, `GROUP BY`, string funcs
(`upper`/`length`). This is the pure-capability (PureCap) compatibility evidence:
a real, substantial application runs **correctly** under pervasive capabilities.

## Gap-scoping conclusion (Tier-2 remaining work)

The remaining work to a **direct in-domain speedtest1 overhead** measurement is NOT a
long tail of compiler cap-tag gaps (those are closed). It is:

1. **Wire the borrow/revoke boundary into the in-domain SQLite** (engine = lender,
   host harness = borrower). The single-row pattern is already validated
   (`tests/runtime-qemu/sqlite-borrow-revoke-probe/`, 2026-07-06); scale it to the
   workload.
2. **Port speedtest1's `main` testset into the domain build.** The domain build omits
   many features (WAL, JSON, UTF-16, mmap, shared cache, load-ext) but `main` uses
   only core SQL (CREATE/INSERT/SELECT/UPDATE/DELETE, indexes, ORDER BY, txns) — all
   supported. Use `SQLITE_STATIC` bindings, never `SQLITE_TRANSIENT` (gap 9: the
   `.h`/`.c` sentinel mismatch in the TRANSIENT patch; see
   `03-07-2026_00-00-02_sqlite-workload-hardening-and-gap9-transient.md`).
3. **Residual risk = the 8-byte-alignment class** (gaps 6/8) "may surface more
   instances under wider workloads." Bounded, known-how-to-fix class (16-align the
   offending embedded struct), not a new class.

So Tier-2 is a wiring + workload-port task with a bounded alignment risk, not a
deep-compiler-gap task. That materially raises the priority of finishing Experiment B
directly (it now serves Compatibility **and** the direct overhead number at low risk).

## Tier-2 direct in-domain measurement (DONE 2026-07-22)

Rather than combine Tier-1's native-x86 denominator with a silicon unit cost, this
measures the denominator **in the real environment**: a bulk SQLite scan running in a
pure-capability domain, instruction-counted with `csrdicount` under QEMU `-icount`
(the same functional-model proxy the paper's QEMU-to-QEMU comparison uses).

Probe: `benchmarks/sqlite/sqlite_boundary_cost_domain.c` (built via
`build-sqlite-capstone.sh DOMAIN_SRC=…`, knob `-DBOUNDARY_ROWS`). It inserts N rows
into `t(id,a,b)` (two TEXT cols), then brackets a `SELECT a,b FROM t ORDER BY id`
scan that reads each column via `sqlite3_column_text` (the borrow site), counting
borrows B and retired instructions T of the scan.

| rows | borrows | scan instrs (in-domain, icount) | instrs / borrow | scan overhead @CPI1 | @CPI1.5 |
|--:|--:|--:|--:|--:|--:|
| 200 | 400 | 1,145,047 | 2,863 | 5.97 % | 3.98 % |
| 400 | 800 | 2,276,562 | 2,846 | 6.01 % | 4.01 % |

- Size-stable: **~2,850 target-ISA in-domain instructions of real PureCap SQLite work
  per column hand-out**; scan instrs scale linearly.
- Overhead = B·171 cyc / (T·CPI); 171 = the silicon **added** protection cost
  (`mrev`+`delin`+`revoke`; the load is already in the scan). `-icount` cannot see
  the multi-cycle `mrev`/`revoke` HW cost, so the *unit* stays the silicon number;
  QEMU supplies only the faithful denominator.

**Result bracket (both confirm "revoke is rare"):**
- **Worst case — a pure column-scan phase (boundary-densest work):** ~**6 %** @CPI1,
  ~4 % @CPI1.5. This is the ceiling: even when the program does nothing but read
  columns across the boundary, a borrow occurs only once per ~2,850 instructions.
- **Typical — a mixed workload** (Tier-1 speedtest1, inserts/index/updates dilute the
  scan): ~**1 %**.

The SQLite bulk workload itself ran **correctly** in-domain throughout (further
compatibility evidence); the only snag was infra (below), not a capability fault.

## Infra finding: stale shared QEMU binary (fixed by rebuild)

`csrdicount` (the `-icount` readout op) was illegal in-domain at first: the built
`capstone/capstone-qemu/build/qemu-system-riscv64` was dated **Jul 10**, but the op
was added to the pinned submodule source on **Jul 13** (`fb3217d179`) — the shared
test binary was 12 days behind its own pinned source (`strings` showed no
`csrdicount`). An **incremental** `ninja qemu-system-riscv64` (13 s, 12 steps,
submodule source unchanged/clean) brought it in sync. Worth noting for the nightly
orchestrator: the QEMU binary is not auto-rebuilt when the submodule source moves.

## Doc corrections made

- `benchmarks/sqlite/README.md`: stale "not yet green / faults at
  `sqlite3RegisterBuiltinFunctions`" status replaced with the end-to-end-pass status
  (all 8 gaps resolved; extended workload green).
- `plans/compatibility-eval-silicon-app.md` §4: prioritization + two-tier framing
  (added earlier this session).
