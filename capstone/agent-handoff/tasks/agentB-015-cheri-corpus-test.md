# Agent-B task 015 — CHERI corpus test (paper "Task 1", security baseline)

*Hand this whole file to Agent-B (`claude-b`), clone `/home/alexey/dev/llvm-capstone-b`,
branch `capstone-bootstrap-b`. Obey `./CLAUDE.md` and
`capstone/agent-handoff/{MULTI-AGENT-WORKFLOW,COORDINATION}.md`.*

## Why this task (read first)

The paper pivoted (see `plans/ndss-pivot-master-plan.md`). It is now a systems-security
paper whose **#1 baseline is CHERI**. The security storyline: for each corpus CVE, show it
is **not blocked under the baseline** and **is blocked on our system**. The meeting made
this **Task 1**: *"compile the corpus on CHERI and test whether this class of
vulnerabilities is blocked or not."* Reviewers will demand the CHERI comparison; without it
the paper is rejected.

This task produces the **empirical CHERI column** of the security table. It is *measurement +
classification*, not a system build.

## The thesis you are testing (state it, then let the data confirm or refute)

CHERI's temporal safety is **`free()`-triggered and asynchronous** — the Cornucopia /
CHERIvoke model quarantines freed allocations and later runs a **stop-the-world sweep** that
invalidates dangling capabilities. Our corpus defects occur at **lifecycle contract points**
(`step` / `reset` / `finalize` / `close`), and several of them leave the memory **still
allocated** (the handle is *logically* invalid, not freed). So we predict CHERI either
**misses** them (stale-but-allocated: the capability is still in-bounds and un-revoked) or
catches them **only after a sweep, not synchronously at the boundary**. Base CHERI purecap
without a revocation sweep is spatial-only and should miss the whole temporal class.

**Confirm or refute this empirically. A refutation is a valid, important result — report it.**

## Predicted per-row oracle (15-row trimmed corpus — verify each)

Corpus is now 15 rows (rows 15/17/18/19 were removed; old row 16 → row 15). Expected CHERI
outcome under **(A) base purecap (spatial only)** and **(B) purecap + revocation sweep**:

| Row | Defect | (A) spatial-only | (B) +revocation sweep | Our system |
|----|--------|------------------|-----------------------|-----------|
| 1 | CPython gh-142830 callback ctx freed mid-call | miss | catches only post-sweep, **not in-window** | S+R fault synchronously |
| 2 | rusqlite closure UAF | miss | post-sweep only | L/R/S synchronous |
| **3** | **diesel column ptr cached across `step` (buffer reused, NOT freed)** | **miss** | **miss** (no free → no sweep hit; cap still in-bounds) | **R revokes on `step`** |
| 4 | PHP stmt after `close` (UAC) | miss | post-sweep only | H synchronous |
| 5 | PHP stmt/db destruction order | miss | post-sweep only | H |
| 6 | PHP UAF via UDF | miss | post-sweep only | L/R/S |
| 7 | CPython gh-99886 subclass dealloc crash | miss | likely post-sweep | H |
| 8 | CPython backup on closed conn (UAC) | miss | post-sweep only | R/H |
| 9 | sqlite3-ruby finalize-time segfault | miss | post-sweep only | H |
| 10 | sqlite3-ruby stmt reuse after close (UAC) | miss | post-sweep only | H |
| 11 | go-sqlite3 double-free | miss | catches (revoked cap) | **L: 2nd free unrepresentable** |
| 12 | expo unfinalized-stmt NPE on close | traps (null) | traps (null) — **weak differentiator** | H, deterministic |
| 13 | CPython null-deref deleted row_factory | traps (null) | traps (null) — weak | L/R deterministic |
| 14 | CPython uninitialised Connection | traps if null, else **miss** | same | **U: no read authority** |
| 15 | datasette authorizer ctx lifetime | miss | post-sweep only | L/R/S |

**The headline cases:** row 3 is the clean "CHERI can't" (stale-but-allocated); rows
1,2,4,5,6,8,9,10,15 are the volume (freed, but CHERI is async so it does not fault
synchronously at the contract point); rows 12/13 are weak (both fault) — say so honestly.

## Vehicle

- **CHERI-RISC-V purecap on QEMU** (matches our RISC-V Capstone work). Use `cheribuild`
  (github.com/CTSRD-CHERI/cheribuild) — `cheribuild.py run-riscv64-purecap` and the
  CheriBSD/QEMU images; their site has full install instructions. **QEMU is fine here** — we
  are testing *catch / no-catch*, not performance.
- **Two configs matter.** (A) stock purecap = spatial only. (B) temporal safety needs a
  revocation sweep (Cornucopia / CHERIvoke / the CheriBSD revoker). **Determine whether the
  toolchain you get has temporal revocation available and ON**, and record it — it is the
  crux of the comparison. If (B) is not readily available, report (A) results + the
  documented design of (B) and argue the async-sweep point from the literature.

## Method (per row)

1. Take each repro's **minimal vulnerable shim** (the `cve-repros` sources; reuse the
   host-shim form). Compile purecap with CHERI-Clang.
2. Run under CHERI-QEMU. Classify the outcome at the offending access:
   - **BLOCKED-SYNC** — CHERI traps *at* the offending access.
   - **BLOCKED-SWEEP** — no trap at the access; a later revocation sweep invalidates it
     (report the latency / that it is not at the contract point).
   - **MISS** — no trap, wrong/stale data returned or corruption (the exploit survives).
3. Record config (A/B), whether temporal safety was on, and a one-line *why*.

## Deliverables

- `capstone/tests/cheri-baseline/RESULTS.md` — the filled 15-row table (CHERI outcome +
  config + why), plus a short methodology note (toolchain version, revocation on/off). This
  is what the paper's security section (Lane A) cites.
- History trail → `history/DD-MM-YYYY_HH-MM-SS_cheri-corpus-test.md`.
- Report: the table, the (A)/(B) config reality, and **any row that refutes the oracle**.

## Scope / lane rules

- **Additive, measurement only.** CHERI toolchain lives **outside** our llvm tree — do not
  vendor it into `llvm/`; install/build it separately (a scratch dir or note the path). No
  `capstone-qemu` change needed. No `llvm/` change.
- Do not touch A's paper, repros' semantics, `start.S`, the monitor, or `capstone-c`.
- Claim the QEMU rootfs lock only if you boot *our* QEMU (you likely won't for this — CHERI
  has its own QEMU). Commit on `capstone-bootstrap-b`, exact paths, **no `Co-Authored-By:`**,
  no debug/report files.

## Closing note

The single most valuable output is a **defensible, reproducible catch/no-catch per row with
the config stated**. If CHERI (with revocation) catches more than predicted, that is the
important finding — surface it immediately, because it reshapes the paper's contribution
claim over CHERI.
