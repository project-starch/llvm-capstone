# Reuse distance at memsys5, PoisonCap and Sublet: instrument and `--size 1` pilot (2026-10-25)

A pilot, not the result: it shows the instrument works and the comparison is set up. The result runs are
`--size 100`, no arguments, as their driver runs.

## What is measured

Reuse distance at SQLite's memsys5, observed from outside the allocator by `reuse/reuse-wrap.c`, linked
into the unmodified speedtest1 with `-Wl,--wrap=sqlite3_config` (and `sqlite3_step`, `sqlite3_reset`, to
count SQL errors). The program's own `SQLITE_CONFIG_HEAP` call installs a copy of memsys5's methods whose
malloc, free and realloc call memsys5's and record addresses as integers. Reuse distance: allocations
since the preceding free of the same start address, the reuse included. Release is the application's
free, so a quarantine inside memsys5 is part of the distance.

Each system against its own control, spatial protection on in both:

| plot | temporal off (control) | temporal on |
|---|---|---|
| PoisonCap | their mem5.c with their own switch off: `#define NESTED_TEMPORAL 1` deleted (their commit 42adb5ca, "compile variable to control temporal safety"; their paper's "CHERI spatial-safe MEMSYS5") | their artifact as published |
| Sublet | Sublet port with block-sized bounds, per-object revocation removed (`sublet/sublet-spatial-only.py`) | Sublet port with block-sized bounds (`sublet-block-bounds.py`) |

Block-sized bounds on our side match PoisonCap's allocation policy and bounds granularity
(`cheri_setbounds(p, memsys5Size(p))`): 1,180 allocations at size 1 in all four arms. The port as ported
(request-sized bounds) makes SQLite allocate differently (1,306) and is a side note.

PoisonCap side: their speedtest1 and SQLite built by `build-speedtest1.sh` (their pinned cheribuild,
`--enable-memsys5`), wrapped by `reuse/build-reuse-poisoncap.sh`, run on the kernel with their FPGA
kernel's options, libc revocation on from the run (the protocol of the completed 20-test run). Sublet
side: their speedtest1.c (one dead function removed, `reuse/make-speedtest1-capstone.py`, CheriBSD headers
from `reuse/shim/`) with SQLite 3.22.0, built by `reuse/build-reuse-capstone.sh`, run on the dev-v6
virtual platform by `reuse/run-reuse-capstone.py`.

## Qualification (`qualification.txt`)

- **Native** (GNU ld, upstream memsys5): `reuse/test-reuse-wrap.c` prints every call and address;
  `reuse/check-reuse-wrap.py` recomputes the counters and bins independently: ALL PASS, including output
  identical with and without the wrapper and the SQL error control. A copy with the bin edges off by one
  fails it (negative control).
- **Sublet arms** (stock, as ported, block-bounds, spatial-only): ALL PASS on the platform.
- **PoisonCap**: with a static test heap, the wrapped and traced programs pass every oracle check; the
  unwrapped one ended in a kernel panic in their revoker. With the heap from malloc (as their main()),
  all four builds (wrapped, traced, unwrapped twice) end identically at the same output line (5,281, exit
  162, SIGPROT), on a free of a pointer whose address became heap+1, never handed out: their system's
  behaviour, the same with and without the wrapper. Not investigated further.
- **Pilots**: wrapped and unwrapped speedtest1 complete the same 20 tests in every arm; installed=1,
  error=0, no SQL error; attempts == issues + failed, issues == distinct + reuses, sum(bins) == reuses,
  live == issues - releases. The wrapper's cost at size 1: 6.09 s against 6.04 s (PoisonCap), 12.35 s
  against 12.21 s (Capstone stock).
- With lld, `--wrap` also redirects SQLite's own calls inside sqlite3.c (PCACHE2 from
  sqlite3PCacheSetDefault, sqlite3_exec's steps), which GNU ld does not; the wrapper forwards every
  configuration operation typed and aborts on one it does not know.

## Pilot, `--size 1` (`pilots-size1.json`, `plots/`)

| arm | allocations | reuses |
|---|---|---|
| PoisonCap, temporal off | 1,180 | 982 |
| PoisonCap, temporal on | 1,180 | 0 |
| Sublet, temporal off | 1,180 | 988 |
| Sublet, temporal on | 1,180 | 988, every bin identical |

Sublet's revocation moves no address. PoisonCap's quarantine gives no block back at this size: 1,180
frees never fill its 4,096 entries, and it never holds 16 MiB. The two controls differ slightly (982 and
988) because the Sublet port keeps a size table inside the heap; each protected curve is read against its
own control.

Plots: one per system, no title (`reuse/plot-reuse.py`; the form the project lead approved). Each comes
with its caption (`*.caption.txt`, counts filled in from the reports) and its values (`*.csv`).
