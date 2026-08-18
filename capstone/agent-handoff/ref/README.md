# `ref/` — what is live, and what to trust

`ref/` is the reference shelf: procedures and registries meant to be **read and acted on**, unlike
`history/` (a dated record, correctly frozen) and `design/` (decisions, superseded by later ones
rather than edited). If a file here is wrong, it misleads someone at a keyboard, so this index
records which are load-bearing and which have gone quiet.

Added 2026-08-12 during a documentation audit. Keep it updated when a doc changes status.

## The four that are read constantly

| doc | what it is | status |
|---|---|---|
| `ISSUES.md` | the issue registry (`R-nn`, `S-nn`, `C-nn`) — every defect, retraction and elimination | **LIVE, authoritative.** 4587 lines; read the entry, not the file |
| `SILICON-BLOCKER.md` | the SQLite-on-silicon investigation, 2026-08-01..08-06 | **SUPERSEDED — do not build on it.** The defect it chased is S-06, fixed in silicon 2026-08-14. Its own line 5170 retracts the method behind most of the reasoning above it; that lesson now lives in `RATE-RULE.md`. **DO NOT RENUMBER OR TRIM:** its line numbers are cited from `tests/fpga-repros/` (a live sent link), `state/current-state.md` and several `history/` notes. |
| `REPO-MAP.md` | repos, branches, submodules, gitlinks, and which remote each lives on | **LIVE** |
| `HOW-TO-LAUNCH-ON-FPGA.md` | the board procedure in full | **LIVE.** The distilled version is the `board-run` skill, which auto-loads |

**A warning that applies to both blocker docs.** Capability `mcause` values in older sections were
named using the `ex_code` inline comments, and **those comments were off by one** until 2026-08-12.
The measured mapping is `mcause = 24 + exception_code`: `UNEXPECTED_OPERAND 25`,
`INVALID_CAPABILITY 26`, `UNEXPECTED_CAP_TYPE 27`, `INSUFFICIENT_PERMISSION 28`, `OUT_OF_BOUNDS 29`,
`ILLEGAL_OPERAND_VALUE 30`. Two published conclusions were wrong because of it — the SQLite wedge
(`25` is `UNEXPECTED_OPERAND`, not `INVALID_CAPABILITY`) and a domain fault in `SILICON-BLOCKER.md`
(`28` is `INSUFFICIENT_PERMISSION`, not `OUT_OF_BOUNDS`). Both are annotated in place. **Check any
capability `mcause` in either file against that table before acting on it** — assume more remain.

## Procedures

| doc | use it for |
|---|---|
| `HOW-TO-RUN-ON-QEMU.md` | the QEMU gate that must pass before a board run |
| `HOW-TO-MEASURE-OVERHEAD.md` | overhead methodology |
| `fpga-debugging-recipes.md` | board debugging patterns |
| `fpga-debug-cycle-optimization.md` | making board sessions cheaper |
| `fpga-borrow-cost-reproduction.md` | reproducing the borrow-cost numbers |
| `gp-free-silicon-smoke-runbook.md` | the gp-free smoke sequence |
| `known-good-controls.md` | which control to run first, and what it must return |
| `testing-matrix.md` | the corpus matrix |
| `SUBAGENTS.md` | the subagent roster and prompt patterns |

## Results and paper inputs

| doc | note |
|---|---|
| `fpga-silicon-measurements-for-paper.md` | **where new silicon numbers go.** Reporting here needs no permission; editing `paper/` does |
| `xlang-security-measurements-for-paper.md` | the xlang case-study numbers |
| `table6-cheri-vs-capstone-explained.md` | how that comparison table is derived |
| `report-style.md` | how results reports should read |

## Background, stable

`capstone-purecap-pointer-model.md`, `capstone-coding-conventions.md`, `runtime-terms-glossary.md`,
`project-structure-overview.md`, `fpga-user-manual.md`, `capstone-agent-test-instructions.md`,
`beebs-benchmark-bringup-manual.md`. Slow-moving; last touched June–July 2026.

## Quiet — check before relying on

* `delegation-guidance.md` (2026-06-19) — superseded in practice by `SUBAGENTS.md` and the
  delegation section of `CLAUDE.md`. The archived peer-lane guide is
  `history/29-07-2026_ARCHIVED_DELEGATION-lane-a-b.md`.

## Conventions

* **`history/` is append-only.** It records what was believed at a date, including claims later
  retracted. Do not edit it to make it correct — annotate the live doc instead.
* **A retraction stays visible.** The pattern here is to strike the old text and state the
  correction next to it, never to delete quietly; several retractions cost days and the trail is
  the point.
* **`design/` holds decisions.** A bug fix or root-cause investigation belongs in `history/`, dated.
