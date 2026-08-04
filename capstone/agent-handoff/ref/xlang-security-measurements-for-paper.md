# xlang corpus — security measurements for the paper

**Paper-facing source of truth for the xlang cross-language FFI comparison.**
Numbers land here first; the paper is edited separately and only with the
project lead's go-ahead.

**Scope: the SECURITY axis only, QEMU-to-QEMU, no silicon.** This corpus answers
"do the mechanisms stop the bugs?" and nothing else. Compatibility is not an axis
here — the paper's compatibility claim is carried by SQLite. Performance on these
programs is unmeasured.

 Both columns are functional emulation —
CHERI-QEMU for the CHERI column, our Capstone QEMU fork for ours. This is the
security question ("does the mechanism catch the defect?"), which needs no
cycle-accurate vehicle. The cost question is a separate measurement and lives
in `tests/cheri-perf/RESULTS.md` under the PI's rule: QEMU-to-QEMU for the
comparison, RTL only for the Capstone absolute. **Do not mix the two tables.**

Last updated 2026-08-02.

---

## 1. The table

15 rows. Identical shims on both columns, identical mock allocator, `-O0` both
sides. ✓ = the offending access was blocked.

| # | Boundary | Reference | Defect class | CHERI spatial | CHERI async *(default)* | CHERI eager | **Capstone** | Blocked by |
|---|---|---|---|:---:|:---:|:---:|:---:|---|
| 1 | Lua↔Rust | rlua #19 | heap UAF | ✗ | ✗ | ✓ | **✓** | revocation |
| 2 | Lua↔Rust | rlua #97 | **stack** UAR | ✗ | ✗ | ✗ | **✗** | — neither |
| 3 | Rust→C | GHSA-f56g-chqp-22m9 | heap UAF | ✗ | ✗ | ✓ | **✓** | revocation |
| 4 | Ruby↔C | CVE-2022-1071 | heap UAF | ✗ | ✗ | ✓ | **✓** | revocation |
| 5 | Ruby↔C | CVE-2022-1934 | heap UAF | ✗ | ✗ | ✓ | **✓** | revocation |
| 6 | Ruby↔C | CVE-2026-1979 | **spatial** (W) | ✓ | ✓ | ✓ | **✓** | **bounds** |
| 7 | Rust→C | RUSTSEC-2022-0070 | heap UAF | ✗ | ✗ | ✓ | **✓** | revocation |
| 8 | Ruby↔C gem | CVE-2020-6838 | heap UAF | ✗ | ✗ | ✓ | **✓** | revocation |
| 9 | Ruby↔C | mruby #3829 | heap UAF | ✗ | ✗ | ✓ | **✓** | revocation |
| 10 | Ruby↔C | CVE-2022-1106 | heap UAF | ✗ | ✗ | ✓ | **✓** | revocation |
| 11 | Ruby↔C | CVE-2018-10191 | **spatial** (R) | ✓ | ✓ | ✓ | **✓** | **bounds** |
| 12 | Ruby↔C gem | CVE-2018-10199 | heap UAF | ✗ | ✗ | ✓ | **✓** | revocation |
| 13 | Ruby↔C gem | CVE-2020-6840 | heap UAF | ✗ | ✗ | ✓ | **✓** | revocation |
| 14 | Ruby↔C | CVE-2017-9527 | heap UAF | ✗ | ✗ | ✓ | **✓** | revocation |
| 15 | Ruby↔C | mruby #3722 | heap UAF | ✗ | ✗ | ✓ | **✓** | revocation |
| | | | **blocked** | **2/15** | **2/15** | **14/15** | **14/15** | |
| | | | **temporal at the contract point** | 0/13 | **0/13** | 12/13 | **12/13** | |

Sources: `xlang/cheri/RESULTS.md`, `xlang/capstone/RESULTS.md`.

## 2. The three claims this table supports

**(1) Base CHERI purecap is blind to the entire temporal class.** Spatial-only
blocks 2/15, and both are bounds catches on the two spatial rows. Zero temporal.

**(2) Realistic CHERI temporal safety still blocks nothing at the contract
point.** The async default — the deployed configuration — blocks the same 2/15.
It blocks **0 of 13** temporal defects when the offending access happens. The
dangling capability is reclaimed only by a later sweep.

**(3) Capstone's design-point configuration matches CHERI's most aggressive
one.** Revoke-on-free blocks 14/15, the same rows `eager` blocks, missing the
same one. The difference the table does not show is deployability: CHERI's
`eager` is explicitly an expensive non-default upper bound, while revoke-on-free
is what Capstone is designed to do.

## 3. Caveats that must travel with the numbers

- **Shims, not real software — on BOTH columns.** Real engines recycle objects
  on internal free lists no revocation scheme observes, so both columns are
  upper bounds. The bias is symmetric because both compile the identical shim
  against the identical mock allocator, and that symmetry is what makes the
  comparison fair even though neither absolute is realistic.
- **Capstone's column uses PERVASIVE revocation**, not the boundary-only scheme
  the design describes (`plans/compatibility-eval-silicon-app.md`). Every
  allocation is independently revocable, so 14/15 is an upper bound for
  boundary-only too. Under boundary-only, rows 1/3/7/12 are genuine cross-domain
  lends and would still qualify; the six VM-register-stack rows (4, 5, 8, 10,
  13, 15) would not, because their stale pointer is engine-internal and never
  crosses a domain line. **That number is unmeasured.**
- **9 of Capstone's 12 temporal catches manifest as a QEMU assert** rather than a
  delivered monitor fault, because the rows compute `regs + offset` on an
  untagged capability and `op_helper.c` has no exception path for that (13 bare
  tag asserts against 46 real raises). The access is prevented either way — the
  only escape would be arithmetic that restored a tag — but which fault real
  hardware delivers is unresolved and needs RTL or silicon. (Count corrected
  from 8 on 2026-08-03 — row 7 is in the assert group. Same day, the column's
  compiler provenance was audited against C-16, which trips the same assert:
  the column was measured pre-fix but is demonstrated unaffected — all 30
  binaries byte-identical under the post-fix compiler, controls run to
  completion, rows 1 and 5 re-run green. Trail:
  `agent-handoff/history/03-08-2026_09-42-37_c16-xlang-column-provenance-audit.md`.)
- **Row 2 is the floor for both systems.** A stack-use-after-return involves no
  allocator, so no allocator-mediated mechanism can observe it. It is in the
  corpus to mark that boundary, not as a defect either system should have caught.

## 4. Corpus composition — state it before quoting "15 rows"

- **13 temporal, 2 spatial.** Rows 6 and 11 are overflows, and both are
  *inside* mruby rather than across a lend, so **the corpus has no row testing
  bounds across a domain boundary**.
- **Six of the 13 temporal rows are the same mechanism** — a raw interior
  pointer cached across a re-entrant Ruby callback while the VM register stack
  is reallocated (rows 4, 5, 8, 10, 13, 15). Rows 8 and 13 are the *same defect*
  at the same commit, reached through two gem methods.
- **12 of 15 are Ruby↔C (mruby).** Rows 1–3 and 7 are the only non-mruby rows.
- **Row 7 replaced a specification error.** The original ("mruby #6701 /
  `mrb_bint_reduce`") does not exist: its issue number belongs to row 6, the
  function is absent from the assigned versions, and the GC hazard it describes
  is closed by the allocation arena. See `xlang/repro/7-old-sortbang/`.

## 5. Reproducibility

| Column | Status |
|---|---|
| Capstone | **`REPRODUCED 15/15`** from a wiped build directory, green in one pass, 2026-08-02. `xlang/capstone/reproduce.sh`; full output in `reproduction-log.txt`. |
| CHERI | 14 rows reproduced from an empty `CHERI_ROOT`, verdicts byte-identical. **Row 7 measured twice with identical verdicts, but on an existing vehicle** — it has not been through a from-scratch provision, so it does not meet the same bar as the other 14. `ONLY_ROWS=7 ./run-cheri-baseline-xlang.sh` takes ~6 min; a from-empty rebuild takes hours. |

Predictions for every row were committed **before** its measurement
(`xlang/cheri/rows.tsv`, `xlang/capstone/rows.tsv`), which is what makes these
tests rather than descriptions. Row 7's Capstone prediction was written after
its first measurement and then re-measured against by the 15/15 clean run.
