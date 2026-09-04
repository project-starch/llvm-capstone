# Plan: benchmark the FULL SQLite boundary interface (all 3 contracts)

**Status:** PROPOSAL (2026-07-22), from the PI's boundary questions. Draft answers:
`/tmp/capstone/pi-boundary-answers.md`.

## Why

The overhead study so far measures boundary-event frequency mainly on the **E→H**
column hand-outs (`speedtest1` + an in-domain scan). The PI asks: are we enforcing
all 3 contracts (E→H, H→E, cb), how do we benchmark the *full* interface, and is
TPC the right workload. Enforcement spans all three (E→H column, H→E bind, CB
sealed callback; corpus 17/17 on RTL); the gap is the *overhead* coverage.

## Approach

1. **Per-contract boundary counting.** Extend the counting shim (currently
   `sqlite3_column_*` = borrow; `step`/`reset`/`finalize` = revoke) to break out
   events by direction using the existing API classification
   (`plans/agent2-api-classification-prompt.md`): E→H (`column_*`/`value_*`), H→E
   (`bind_*`, `exec`/`prepare` SQL-in), CB (`create_function`/hooks). Emit
   per-direction counts *B_dir* + total retired instrs *T*.
2. **Full-interface workload = SQLite's own test suite** (sqlite.org/testing.html:
   TCL harness / SLT / TH3-style). It exercises every API path, so it drives H→E
   and CB crossings, not just E→H — a true full-interface frequency — and doubles
   as exhaustive compatibility evidence (a real, comprehensive workload runs
   correctly under pervasive capabilities). Run native with the per-contract shim
   (Tier-1 frequency) and, where feasible, a representative slice in-domain
   (Tier-2 worst-case), same two-tier method as the current study.
3. **Overhead rollup.** overhead = Σ_dir B_dir · (silicon per-op cost) / (T · CPI),
   reported per contract and aggregate. Headline the aggregate ~1% (mixed) /
   ≤~6% (dense), never the 22× per-op ratio.
4. **TPC (optional).** `speedtest1` (SQLite canonical) is the primary; if a
   "standard DB benchmark" number is wanted, add a TPC-C/H-style workload — but the
   test suite above is the stronger *interface-coverage* vehicle.

## Deliverables

- Per-contract boundary-frequency table (E→H / H→E / CB) over the SQLite test
  suite; aggregate + per-direction overhead at the silicon per-op cost.
- A compatibility statement: the full SQLite test suite runs correctly under
  pervasive capabilities (extends the existing end-to-end result).
- Paper: fold the per-contract frequency + full-interface overhead into the
  boundary-overhead subsection.

## Constraints
Propose-before-implement (this is the proposal). Two-tier method + report style per
`ref/report-style.md`. No new firmware needed for CB (sealed-callback probe already
composes from existing ops).
