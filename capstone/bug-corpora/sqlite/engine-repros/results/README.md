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
| `sublet` | the same script with `CORPUS_SUBLET=1` | Capstone domain, Sublet discipline on memsys5 |
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
