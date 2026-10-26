# Reuse distance at memsys5, PoisonCap and Sublet: `--size 100`, their default (2026-10-25/26)

The result runs the pilot (`../2026-10-25-reuse-pilot/`) prepared: the same instrument, arms, builds and platforms,
speedtest1 with no arguments, which is `--size 100`, as their driver runs it. What is measured and how is in the
pilot README; the qualification of every binary named here is in its `qualification.txt`.

| plot | temporal off (control) | temporal on |
|---|---|---|
| PoisonCap | their mem5.c with their own switch off (`NESTED_TEMPORAL` deleted) | their artifact as published |
| Sublet | Sublet port, block-sized bounds, per-object revocation removed | Sublet port, block-sized bounds |

## Files

- `runs.json`: every collected run's result lines (binary and kernel hashes, exit status, the 20 test times,
  the wrapper's `REUSE-GAP` and `SQL-CHECK` lines at exit); `reports.jsonl`: the report lines the plots read.
- `checks.txt`: `reuse/collect-reuse.py`'s verdict per run (admissible: exit 0, the 20 tests, installed=1,
  error=0, 32 bins, the four identities, >= 1,000 issues, no SQL error), whether repetitions agree, and
  which set each plot shows (the lowest set whose two runs used the qualified binaries).
- `plots/`: one plot per system (`reuse/plot-reuse.py`, the approved form: no title, shared x axis), each with
  its caption and values (`.csv`).
- `kit-comparison.txt`: the Sublet arms on the result runs' SDK kit (llvm-capstone f4f58b02, 2026-10-20) and on
  one built from dev 9c779890, `--size 20`, side by side on one host: the same counters and bins, the same time
  (1,737 / 1,736 s and 1,626 / 1,630 s). Both sides are built at -O0, PoisonCap's own level.

## Runs

| set | PoisonCap off / on | Sublet off / on |
|---|---|---|
| 1 | p13; the wrapper before the review fixes (93c95687 / 09d52ff2) | p13; 9836684d / 6bfe4d1b |
| 2 | p12; aad6a979 / 97b02659 | p12; 9836684d / 6bfe4d1b |
| 3 | p12; aad6a979 / 97b02659 | p13; 9836684d / 6bfe4d1b |

PoisonCap: E3c protocol (their FPGA kernel's options, libc revocation default 0 at boot and 1 from the run),
`reuse/launch-poisoncap-arm.sh`. Sublet: the paged virtual platform, `reuse/launch-capstone-arm.sh`.

## Result

| | allocations | reuses, temporal off | reuses, temporal on |
|---|---|---|---|
| PoisonCap | 52,146 | 51,499 (98.8 %); 95 % of them within a distance of 7 | 35,316 (67.7 %); 92 % of them at distances 512 to 65,535 |
| Sublet | 52,146 | 51,499 (98.8 %) | 51,499, every counter and bin identical to off |

- PoisonCap's quarantine delays reuse at this size; it does not prevent it. At `--size 1` (1,180 allocations)
  its temporal-on arm reused nothing because the quarantine never filled.
- Sublet's revocation moves no address: the two arms' reports are identical.
- The two controls issue the same number of blocks and reuse as many, but their bins differ slightly (as at
  `--size 1`); each protected curve is read against its own control.
- All three sets of each system are identical in every counter and bin (12 runs, all admissible), PoisonCap's
  set 1 with its earlier wrapper included.
- Run times: PoisonCap 16,656-16,871 s on p13 and 20,239-20,380 s on p12; Sublet 31,510-31,960 s on p13 and 35,465 / 36,247 s on p12.
  Not a cost comparison (different platforms, shared hosts).
