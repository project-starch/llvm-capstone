# Results

33 defects in SQLite 3.22.0, each measured on three arms. Every cell is either
detected or not detected: cases that produced a hang, a verdict the harness
could not score, or a probe showing the defect site was never reached are not
in this corpus.

| arm | detected | not detected | temporal | spatial |
|---|---:|---:|---:|---:|
| `spatial` | **8** | 25 | 6/25 | 2/8 |
| `sublet` | **31** | 2 | 24/25 | 7/8 |
| `cheribsd-revocation` | **5** | 28 | 5/25 | 0/8 |

> **CORRECTION (2026-10-20): case 24's CAUGHT in verdicts.tsv on the memsys5 malloc arms is an incidental NULL
> dereference, not a capability detection.** The affected rows are `virtual-malloc` (line 82) and `vp2-malloc` (line 84,
> the board lane's silicon fold). Both read CAUGHT, cause 24 in sqlite3OpenTable, credited "by function" because host ASan
> reports the defect there.
> - **Method:** an LDC trace (capstone-qemu cap-window-counter 043e0dbf23, `CAPSTONE_TRACE_LDC`, kit qkit7-trace = nf2's
>   files) of the old image 7c8bfc885ea3914a, read by the board lane.
> - **The trace:** in sqlite3PrimaryKeyIndex the stale Table capability loads TAGGED (va 0x55034, with 1 MiB bounds, not the
>   Table's own), and pTab->pIndex, at that pointer + 0x20, loads UNTAGGED 0 (va 0x55038). The NULL pPk then faults at
>   sqlite3OpenTable+648 (`lw 0xb0(a3)`).
> - So the use-after-free itself is not detected: this arm does not revoke, and the stale capability is valid. The fault
>   came from what the freed memory happened to hold.
> - The image rebuilt without `+movc-keeps-integer-source` has the same instructions and different frame sizes. It reads
>   MISSED on QEMU and on silicon (2026-10-20), and its trace shows no load in sqlite3PrimaryKeyIndex.
> - **Not affected:**
>   - the headline matrix, whose spatial row already scores case 24 not-detected;
>   - the nested-pools row (cause 25, revocation).
> - **Not assessed:** matrix.tsv's sublet detection of case 24, a kernel-pc halt with cause 24 under an allocator that
>   does revoke.
> - The tsv rows are generated (derive-verdicts.py, fold-silicon.py), so they are corrected by regeneration once the oracle
>   classifies near-zero untagged faults, not by hand.
> - A census of every pass reference (the compiler lane, null-census.py) finds 6 other CAUGHT cells whose faulting address
>   is below 0x1000: sqlite 02, 05 and 19, sq1q15 02, cpython lt0 10 and 15. They are CANDIDATES only, each to be read the
>   same way.

All 33 are clients of **memsys5**, SQLite's arena allocator, so all 33 are
nested by the allocator that serves them. That is the point of the comparison:
the three arms run the same defects from the same sources with the same
triggers, and differ only in what sits underneath.

**The temporal row is the result.** Sublet reports 24 of 25 where base
Capstone reports 6 and CheriBSD 5. A use-after-free between two memsys5 chunks
never crosses a `malloc` boundary, so a mechanism that knows only `malloc`
blocks has nothing to check; Sublet bounds each sub-allocation, and sees them.

## What produced these numbers

| arm | runner | platform |
|---|---|---|
| `spatial` | `ports/sqlite/repro322/corpus322.sh` | Capstone domain, base |
| `sublet` | the same script with `CORPUS_SUBLET=1`, the lent-pool memsys5 port (both removed; the current Sublet arm is `virtual-nested-pools`, `repro322/build-virtual.py --sublet`) | Capstone domain, Sublet discipline on memsys5 |
| `cheribsd-revocation` | `ports/sqlite/cheribsd/run-corpus-cheri.sh` | CheriBSD 15.0-CURRENT riscv64-purecap, QEMU |

The two Capstone arms are one image each of the same source, differing only in
whether the allocator carries the Sublet patch. The CheriBSD column was
measured in a single boot on 2026-10-06, with `poscontrol.c` run first as a
gate: it is a use-after-free through the system allocator and must fault, or
the run is void and nothing from it is recorded.

Both the build and the runner take their case list from
`ports/sqlite/repro322/tags.tsv`, which maps every runner tag either to its
case directory or to the reason it is not a case. `ls out/` answers what was
built, which is a different question, and the two have differed before.

## Two cells that are corrections rather than measurements

`29_fz02_btree_get4byte_page_overread` and `31_fz10_dbstat_decodepage_overflow`
were previously recorded as detected on `cheribsd-revocation`. Those verdicts
came from the **probe build** -- the instrumented amalgamation, built with
`-DLB_PROBE_BUILD=1` into a separate tree -- while the other 31 cases in that
column came from the plain build, so the column was not comparable with
itself. The plain build does not fault on either. Whether the instrumentation
changes behaviour or merely reaches further is not established, and each
case's oracle says so.
