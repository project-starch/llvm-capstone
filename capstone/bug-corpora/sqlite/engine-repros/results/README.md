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

> **CORRECTION (2026-10-20): case 24's CAUGHT in verdicts.tsv on the vp2-malloc arm (line 84, the board lane's silicon
> fold) is an incidental NULL dereference, not a capability detection.** It reads CAUGHT, cause 24 in sqlite3OpenTable,
> credited "by function" because host ASan reports the defect there.
> - **RETRACTED the same evening (an adversarial audit): this box first named the `virtual-malloc` row (line 82) as
>   established too.** That row comes from a DIFFERENT image, 6fbbd27ef45c7ac8 (results/2026-10-11-virtual-request-bounds,
>   another compiler build), which has not been traced. The trace below is of 7c8bfc885ea3914a, the vp2-malloc row's own
>   image. The virtual-malloc row has the same symbol and offset (sqlite3OpenTable+648), so it is a CANDIDATE only.
> - **The vp2-malloc row is established by QEMU on the same image bytes.** The board records carry no fault address.
> - **Method:** an LDC trace (capstone-qemu cap-window-counter 043e0dbf23, `CAPSTONE_TRACE_LDC`, kit qkit7-trace = nf2's
>   files) of the old image 7c8bfc885ea3914a, read by the board lane.
> - **The trace:** in sqlite3PrimaryKeyIndex the stale Table capability loads TAGGED (va 0x55034, with 1 MiB bounds, not the
>   Table's own), and pTab->pIndex, at that pointer + 0x20, loads UNTAGGED 0 (va 0x55038). The NULL pPk then faults at
>   sqlite3OpenTable+648 (`lw 0xb0(a3)`).
> - So the use-after-free itself is not detected: this arm does not revoke, and the stale capability is valid. The fault
>   came from what the freed memory happened to hold.
> - The image rebuilt without `+movc-keeps-integer-source` has the same instructions and different frame sizes. It reads
>   MISSED on QEMU and on silicon (2026-10-20), and its trace shows no load in sqlite3PrimaryKeyIndex.
> - **Not affected by case 24 (THIS case only):**
>   - the headline matrix, whose spatial row already scores case 24 not-detected. The census below can affect the
>     headline matrix through 02, 05 and 19;
>   - the nested-pools row (cause 25, revocation).
> - **Not assessed:** matrix.tsv's sublet detection of case 24, a kernel-pc halt with cause 24 under an allocator that
>   does revoke.
> - The tsv rows are generated (derive-verdicts.py, fold-silicon.py), so they are corrected by regeneration once the oracle
>   classifies near-zero untagged faults, not by hand.
> - A census of every pass reference (the compiler lane, null-census.py) finds 6 other CAUGHT cells whose faulting address
>   is below 0x1000: sqlite 02, 05 and 19, sq1q15 02, cpython lt0 10 and 15. They are CANDIDATES only, each to be read the
>   same way.

> **CORRECTION (2026-10-20, continued): cases 02, 05 and 19 on vp2-malloc are CAUGHT on an untagged NULL or -1, not on
> a capability check of the defect.** These are verdicts.tsv's vp2-malloc rows, the board lane's silicon fold of
> results/2026-10-18-vp2-rdx0. Each was traced by LDC on QEMU on the row's own image: 02 c190938388045b7b,
> 05 95262bfccec3f7ac, 19 0a7a279444bdc8cb. The traces are sqnull-{02,05,19}old, and the rebuilt images agree. Then an
> adversarial audit.
> - **02** (sqlite3BtreeUpdateMeta+180):
>   - pBt loads tagged, and pBt->pPage1 (+0x30) loads an untagged 0, which the next ldc dereferences (fault address 0x80).
>   - Host ASan reports this case as a SEGV, NOT a use-after-free (2026-10-11-host-asan/results.json), so on the host the
>     objects were LIVE and pPage1 a genuine NULL.
>   - The case's own PROVENANCE ("Host ASan flags heap-use-after-free") disagrees with that record, so its UAF
>     classification is itself unverified.
> - **05** (sqlite3DropTriggerPtr+592):
>   - the faulting base is tableOfTrigger's return value, an untagged 0 reloaded from its stack slot (fault address 0x0);
>   - tableOfTrigger RETURNED, so its read of the freed Schema (ASan's use-after-free) went through a valid capability;
>   - the 0 is sqlite3HashFind's not-found result (a static nullElement), not a value read from freed memory.
> - **19** (nodeRelease+144):
>   - a node pointer arrives as an untagged -1;
>   - on the rebuilt image the trace shows pNode->pParent read through the stale node's tagged capability as -1. That is
>     memsys5's free-list link, written at block offset 0 when the block was freed.
>   - On silicon this row reads **cause 4 (load address misaligned)**, not 24 (Q-31's order): the trap that fired on the
>     board was the alignment check of -1 + 0x18.
>   - The counter-reading has the most force here: an integer store cleared a pointer's tag. But the value is -1, which
>     faults on any machine.
> - **In all three the capability through which the defect's own access went was VALID and tagged,** with bounds of about
>   1 MiB seen in the traces (not the object's). This memsys5 arm never revokes, so the use-after-free is not detected; the
>   fault is incidental.
>   - That a non-capability build would also fault is REASONING, except for 02, where the host SEGV is direct. Its offsets
>     would differ (8-byte pointers).
> - **Candidates, NOT corrected (not traced on their own images):**
>   - virtual-malloc 02, 05, 19 (results/2026-10-11-virtual-request-bounds, other images; 05 faults there at +600);
>   - silicon-sqlite-stock 02 and 05 (the physical harness; offsets 72 and 340);
>   - **matrix.tsv's spatial rows for 02, 05 and 19** (the physical-profile harness). If they share the mechanism, the headline's
>     "base Capstone reports 6" temporal detections drops by up to three. Their traces are requested.
>   - matrix.tsv's CheriBSD rows (SIGPROT, which a purecap NULL dereference also raises).
>   - matrix.tsv's sublet rows (Sublet revokes, so a real detection is possible).
> - **Not affected:** virtual-nested-pools (cause 25, revocation).

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
