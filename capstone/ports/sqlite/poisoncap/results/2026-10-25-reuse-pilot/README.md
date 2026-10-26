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

PoisonCap side: their speedtest1 and SQLite built by `speedtest1/build-speedtest1.sh` (their pinned cheribuild,
`--enable-memsys5`), wrapped by `reuse/build-reuse-poisoncap.sh`, run on the kernel with their FPGA
kernel's options, libc revocation on from the run (the protocol of the completed 20-test run). Sublet
side: their speedtest1.c (one dead function removed, `reuse/make-speedtest1-capstone.py`, CheriBSD headers
from `reuse/shim/`) with SQLite 3.22.0, built by `reuse/build-reuse-capstone.sh`, run on the dev-v6
virtual platform by `reuse/run-reuse-capstone.py`.

The pilot's Sublet rows (`pilots-size1.json`: images 55cc0ed6 / 086e9fd8, built with the wrapper before
the review fixes) ran on the flat silicon-twin kit (`-DCAPSTONE_VM_FLAT_TABLE`, at most 65,536 records,
no growth), which holds a `--size 1` run. From `--size 20` on, Sublet runs died there with cause 30 in
`capstone_cap_derive`, the table full; that is the kit's limit, not the platform's. The result runs
(`../2026-10-25-reuse-size100/`) use the paged platform: the default `runtime/virtual/build-adapter.sh`
module (paged lifetime table, grows on demand), v7 QEMU with `x-capstone-test-csrs=true`, module argument
`supervisor_slots=32`. There the qualified images (9836684d / 6bfe4d1b) give the pilot's bins exactly
(`qualification.txt`, second and third parts). On the paged platform
both Sublet arms complete `--size 20` (exit 0, 8,470 allocations, 7,835 reuses, identical bins).
`reuse/launch-capstone-arm.sh` and `reuse/launch-poisoncap-arm.sh` start one result run each.

## Qualification (`qualification.txt`)

The record has two parts: the wrapper of the first commit, and the wrapper, test program and checks
after a review (exact `va_arg` types, a failure on `SQLITE_CONFIG_HEAP` with a null heap after install,
the realloc oracle reading the old offset before the call; the checks now require every test program to
exit 0, the traced output to equal the wrapped one, the exit report to dominate the alloc-done snapshot
bin by bin, and at least 1,000 allocations and 1,000 SQL steps in a pilot). The second part passes on
every platform, and its three edited-report negative controls fail. The bullets below describe the first part
unless they say otherwise. A second review found the exit-bin check passing a report that put every
reuse in one bin (shown: the previous checker passes it), and the pilot gate checking attempts but not
reuses; both are fixed and the record's third part re-runs every set. Two statements of the review
commit are corrected here: `reuse-wrap.c` reads the PCACHE2 and MALLOC arguments as non-const method
pointers where SQLite's own fallback and default-cache calls pass const ones, the same as SQLite's own
`sqlite3_config` reads them (a qualifier difference, not a different object type); and the module
arguments are recorded in each result row (`module_args`), not in the `@@PARAMS` line.

- **Native** (GNU ld, upstream memsys5): `reuse/test-reuse-wrap.c` prints every call and address;
  `reuse/check-reuse-wrap.py` recomputes the counters and bins independently: ALL PASS, including output
  identical with and without the wrapper and the SQL error control. A copy with the bin edges off by one
  fails it (negative control).
- **Sublet arms** (stock, as ported, block-bounds, spatial-only): ALL PASS on the platform.
- **PoisonCap**: with a static test heap, the wrapped and traced programs pass every oracle check; the
  unwrapped one ended in a kernel panic in their revoker. With the heap from malloc (as their main()),
  all four builds (wrapped, traced, unwrapped twice) end identically at the same output line (5,281, exit
  162, SIGPROT), on a free of a pointer whose address became heap+1, never handed out: their system's
  behaviour, the same with and without the wrapper. Not investigated further. After the review the test
  program takes its step count as an argument: at 4,000 steps, before that line, all three PoisonCap
  test programs exit 0 and pass every oracle check, in quarantine mode (`REUSE_EXPECT_QUARANTINE=1`: a
  quarantining allocator need not reuse at all).
- **Pilots**: wrapped and unwrapped speedtest1 complete the same 20 tests in every arm; installed=1,
  error=0, no SQL error; attempts == issues + failed, issues == distinct + reuses, sum(bins) == reuses,
  live == issues - releases. The wrapper's cost at size 1: 6.09 s against 6.04 s (PoisonCap), 12.35 s
  against 12.21 s (Capstone stock).
- With lld, `--wrap` also redirects SQLite's own calls inside sqlite3.c (PCACHE2 from
  sqlite3PCacheSetDefault, sqlite3_exec's steps), which GNU ld does not; the wrapper forwards every
  configuration operation typed and aborts on one it does not know.

## Pilot, `--size 1` (`pilots-size1.json`)

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

The size-1 plots are not kept: the result plots are `../2026-10-25-reuse-size100/plots/`, in the form
the project lead approved on the pilot (`reuse/plot-reuse.py`).
