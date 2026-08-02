# xlang corpus — what we have measured

**Start here.** One page: what is done, what is not, where it lives, and how to
re-run it. Per-column detail is in `cheri/RESULTS.md` and `capstone/RESULTS.md`;
the paper-facing version is
`capstone/agent-handoff/ref/xlang-security-measurements-for-paper.md`.

Last updated 2026-08-02.

---

## Scope: this is the SECURITY result, and only that

**Do the mechanisms stop the bugs?** That is the whole question this corpus
answers, and it is answered: 15 rows, both columns, reproduced.

Compatibility ("does the real application run unmodified?") is **not an axis of
this corpus** — the paper's compatibility claim is carried by SQLite, which
compiles, links and runs end to end as a purecap Capstone domain. What we learnt
porting mruby appears below as the justification for using shims, which is what
it is, rather than as a half-finished second axis.

Performance on these programs is **not measured**. See the end.

---

## The result

15 real cross-language defects, each reproduced on stock toolchains, then
measured under CHERI-RISC-V purecap (three revocation configs) and under
Capstone. Identical shims on both columns, identical mock allocator, `-O0` both
sides.

| | CHERI spatial | CHERI async *(the deployed default)* | CHERI eager | **Capstone** |
|---|:---:|:---:|:---:|:---:|
| rows blocked | 2/15 | **2/15** | 14/15 | **14/15** |
| temporal blocked **at the contract point** | 0/13 | **0/13** | 12/13 | **12/13** |

Three things that table says:

1. **Base CHERI purecap is blind to the whole temporal class.** Its 2/15 are
   both bounds catches on the two spatial rows.
2. **CHERI's deployed default catches nothing temporal at the contract point** —
   0 of 13. The dangling capability is reclaimed only by a later sweep.
3. **Capstone matches CHERI's most aggressive config** — same 14 rows, same
   single miss — but `eager` is an expensive non-default upper bound, while
   revoke-on-free is Capstone's design point. The security comparison is
   therefore a near-tie at a configuration nobody deploys; the separating axis
   is cost, which we have not measured.

**Row 2 is the floor for both systems.** A stack-use-after-return involves no
allocator, so no allocator-mediated mechanism can observe it. It is in the
corpus to mark that boundary.

---

## Where things live

```
xlang/
├── repro/     15 defects on stock toolchains — GROUND TRUTH, no dependency on our stack
├── cheri/     the purecap column: shims, fidelity gate, 3 configs
└── capstone/  our column: domain, revoking allocator, host, verifier
```

**Two directories are EVIDENCE, not part of the measurement.** They are kept
because each one makes a claim in this file falsifiable; nothing runs them
during a measurement, and neither is dead code:

| Directory | Backs the claim | Check it by |
|---|---|---|
| `cheri/mruby-port/` | "purecap mruby runs, but it took four changes, only 1 of 9 pinned trees is proven, and CHERI clang was **silent** on the fatal bug" | `./build-purecap-mruby.sh`; the silent-compiler claim is `why_warnings_miss_it.c`, compiled with every CHERI diagnostic |
| `repro/7-old-sortbang/` | "row 7's first replacement candidate was rejected because **both ASan and valgrind mask it**, and its free is a shrink-in-place with nothing to revoke" | its `build.sh` + `run.sh` still reproduce all three outcomes |

The original row 7's build was kept for the same reason — a negative result that
cannot be re-run is an assertion. If you are looking for something to delete,
these are not it; the dead code was `xlang/shim/`, removed 2026-08-02.

| Want | Look at |
|---|---|
| what each defect IS | `repro/<n>/target.md`, `boundary.md` |
| the CHERI numbers | `cheri/RESULTS.md` |
| the Capstone numbers | `capstone/RESULTS.md` |
| why a shim is allowed to stand in for the real defect | `cheri/check_shim_fidelity.py` |
| what the allocator must do, and what `-O0` does not prove | `capstone/ALLOCATOR-CONTRACT.md` |
| predictions, committed before each run | `cheri/rows.tsv`, `capstone/rows.tsv` |

---

## Re-running it

```bash
xlang/repro/reproduce.sh              # 15/15, minutes, no CHERI or Capstone needed
python3 xlang/cheri/check_shim_fidelity.py     # 19/19, ~8s, only needs clang
xlang/capstone/reproduce.sh           # 15/15, ~1h, clean rebuild + 30 QEMU boots
cd xlang/cheri && ./run-cheri-baseline-xlang.sh          # ~70 min, all 15 x 3 configs
cd xlang/cheri && ONLY_ROWS=7 ./run-cheri-baseline-xlang.sh   # one row, ~6 min
```

The CHERI vehicle is built by
`capstone/tests/cheri-baseline/provision-cheri-vehicle.sh`, verified from an
empty `CHERI_ROOT`. It does **not** build CheriBSD world — that is blocked on a
modern Linux host and the script header explains why, because the obvious fix is
to try world again.

**Reproduction status.** Capstone: `REPRODUCED 15/15` from a wiped build
directory, green in one pass. CHERI: rows 1–6 and 8–15 reproduced from an empty
`CHERI_ROOT` with byte-identical verdicts; row 7 measured twice identically but
on an existing vehicle, so it does not meet that bar.

---

## Caveats that must travel with the numbers

- **Shims, not real software — on BOTH columns.** Real engines recycle objects
  on internal free lists no revocation scheme observes, so both columns are
  upper bounds. The bias is symmetric — identical shim, identical mock
  allocator — and that symmetry is what makes the comparison fair even though
  neither absolute is realistic.
- **Capstone's column is PERVASIVE revocation**, not the boundary-only scheme
  the design describes. Every allocation is independently revocable, so 14/15
  bounds the boundary-only number too. Rows 1/3/7/12 are genuine cross-domain
  lends and would survive it; the six VM-register-stack rows would not, because
  their stale pointer never crosses a domain line. **Unmeasured.**
- **8 of Capstone's 12 temporal catches manifest as a QEMU assert**, not a
  delivered fault: `op_helper.c` has no exception path for arithmetic on an
  untagged capability. The access is prevented either way, but which fault real
  hardware delivers needs RTL or silicon.
- **Corpus composition, before anyone quotes "15 rows".** 13 temporal, 2
  spatial. Six of the 13 are the *same mechanism* (a raw interior pointer cached
  across a re-entrant Ruby callback while the VM stack is reallocated). Rows 8
  and 13 are the *same defect* at the same commit via two gem methods. 12 of 15
  are Ruby↔C. And **no row tests bounds across a domain boundary** — the two
  spatial rows are overflows inside mruby.

---

## Why shims, and what that cost CHERI

The real engines are not the vehicle, and the reasons are worth stating because
they are themselves a result about CHERI:

- purecap mruby **does** run, but it took four changes to get there — one source
  edit, two ABI flags, one upstream config switch (`cheri/mruby-port/`).
- The corpus pins **nine mruby versions spanning 2017–2026**, and only one is
  proven to boot purecap.
- **No purecap Rust toolchain exists**, so rows 1, 3 and 7 cannot run under
  CHERI in any form.
- CHERI clang was **completely silent** on the fatal provenance bug — tested,
  not assumed (`cheri/mruby-port/why_warnings_miss_it.c`). "Compiles purecap"
  and "runs purecap" are separate milestones.

So each defect is re-expressed as a shim reproducing the same allocate / free /
offending-access with the same geometry, and `cheri/check_shim_fidelity.py` is
what keeps that substitution honest — 19/19, each shim rebuilt natively under
ASan and required to produce the same defect class, access width and geometry
as the real row.

---

## What is missing

**Performance on these programs.** Nothing measures the cost of running the
corpus under either system. `cheri-perf/` and
`runtime-qemu/revoke-cost-probe/` measure a `malloc → touch → free`
microbenchmark, not these workloads. Both columns' shims already build and run
under QEMU, so instruction counts on the same 15 workloads is largely harness
work — and this is the axis the paper says actually separates the two systems,
given that the security comparison is a near-tie at `eager`.

**Row 7 on CHERI** has not had a from-empty-`CHERI_ROOT` provision.

**A boundary-only Capstone config** would turn the pervasive-revocation caveat
above into a number instead of an upper bound.
