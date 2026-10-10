# Whole-corpus audit and cross-program comparison, 2026-10-10

The second audit of the day. The first (`10-10-2026_17-20-00_bug-corpus-audit-three-programs.md`)
traced every cell of FFmpeg's, tshark's and memcached's tables to its record. This one reads all 28
corpora, compares the other six programs' bugs and methods with those three, and fixes what the
comparison showed needed fixing. It also adds virtual-Capstone arms to the three programs and hunts for
the nested-allocator defect shapes they lack. Lane branch `audit2-corpora`.

## Pre-registered runs (each section committed and pushed before its run)

### R1. memcached allocator-repros on stock CheriBSD: a fixed arm for every case, and case 8 attributed

The runner (`runners/cheribsd/run-defects.py`) had no fixed run, so case 8's CHERI catch -- the whole
of "memcached nested spatial CHERI 1/4" -- had no control and no attributed site. The fault was
recorded only as `pc=0x105530`. The runner now runs FIXED then buggy on the same program in one boot
(`corpus.h`: event value 1 selects the fixed sequence), and judges a buggy arm with declared
`fault_sites` by resolving the fault pc against the program's own symbols (`llvm-nm` from the SDK,
the supervisor's resolved `mc_defect_read` as the load anchor). Case 8 declares `mc_case_body` (its
`case.json`, `fault_sites_why`, from the source).

Platform: stock CheriBSD purecap, image `~/cheri/output` (0cb16209), libc revocation on, all 9
cases, one boot, with the revocation control and the platform controls in the same boot.

Predictions:
- fixed arm, all 9: completes (exit 0, completed=1, no fault).
- buggy 00-07: completes, as recorded on 2026-10-06 (the cache keeps the object; nothing reaches free()).
- buggy 08: SIGPROT, bounds (si_code 1), at a pc inside `mc_case_body`.
- revocation control: faults at its labelled load.
If 08's fault is elsewhere, or its fixed arm does not complete, the cell becomes NO-READING and
memcached's nested spatial CHERI cell drops from 1/4 to 0/3.
