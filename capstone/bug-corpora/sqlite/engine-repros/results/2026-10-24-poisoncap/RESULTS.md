# Results: the temporal cases on PoisonCap's published memsys5 (2026-10-24)

Read with `PREDICTIONS.md` (committed 3cd43d008369, before the run) and `PREDICTIONS-REVIEW.md`
(committed 378c8ea8f8bf, before any result was read). Platform: PoisonCap's published artifact, built
from its pins in the Ubuntu 24.04 container (`ports/sqlite/poisoncap/`), booted single-user with
`runtime_revocation_default=0` (`platform-smoke.json`: their four cheribsdtest poison tests PASS).
SQLite: their `sqlite/` tree's amalgamation (`prepare-sqlite.json`). Each run is one boot; the gate
(controls) passed in every run. Every case was run twice per row (`runs/published-*` and
`runs/published-*-rep2`); the two readings agree in every case, and the second records failing
`rc=` lines, which reclassified case 23 from "returned" to "returned-with-error". Result lines:
`runs/*/results.jsonl`; scoring: `verdicts/*/verdicts.md` (`ports/sqlite/poisoncap/verdicts.py`).

## Their published memsys5, 19 qualified temporal cases

| reading | lookaside on | lookaside off |
|---|---|---|
| fault at site, mechanism-signed (SIGPROT code 3 + poison line) | 12: 03 05 06 10 14 16 17 18 19 20 22 24 | 11: the same without 19 |
| fault at site, not the poison report | 2: 00 (code 1, bounds), 02 (code 2, tag) | 2: the same |
| no reading, heap exhausted (their quarantine keeps freed blocks) | 07 ("some iterations could not set up"), 23 (`optimize rc=7`, NOMEM) | same |
| no reading, trigger not set up on their SQLite tree | 21 (`scan_before=0`, "no node was written then freed") | same |
| no reading, an earlier CHERI-port fault | — | 19: bounds fault in `sqlite3RowSetInit`; their tree still allocates a RowSet as 64 bytes (`sqlite3.c:72323`), which `ports/sqlite/cheribsd/adapt-sqlite-cheri.py` fixes for our CheriBSD arm |
| defect path reached, no fault | 15 (`instance_rows=6`, the count at which host ASan reports the use-after-free) | same |
| unresolved | 12 (scan completes, 5 rows; whether its reallocation moved the array is not established) | same |
| fault at site / signed | **14 / 12** | **13 / 11** |

## Ours, recorded

Our Sublet memsys5 (`virtual-nested-pools`, lookaside on; `results/2026-10-11-virtual-request-bounds`,
equal to `results/2026-10-20-q15-repair`): fault at site 19 of 19, signed (cause 25) 17; 00 is a bounds
fault (cause 28) and 02 an untagged one (cause 24), as on theirs. Every case above that reads "no
reading" or "no fault" on theirs (07, 12, 15, 21, 23) is a cause-25 catch on ours.

## Against the predictions

Predicted lookaside on 18 or 19 at site, 18 signed; lookaside off 19 and 19. Measured 14/12 and 13/11.
The review's corrections held where they could be checked: 02 faults at its site without the poison
report; 24 is caught in both rows; 07 and 23 run out of heap. Not foreseen: 15's silent stale read, 21's
setup on their tree, 19's RowSet fault with lookaside off.

## Limits

- QEMU only; their poison checks are their QEMU's model, as ours are our QEMU's.
- Function-level sites: 16 and 17 fault in `memcmp`, which matches by name only (their console poison
  bounds are in `results.jsonl`).
- Three of the five cases theirs does not catch are no readings rather than misses; two of those (07,
  23) are their design (quarantine against a 1 MiB heap), one (21) is their SQLite tree. 15 is a
  reached path without a report, not yet explained; 12 is open.
