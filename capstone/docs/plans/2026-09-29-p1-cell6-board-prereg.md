# Pre-registration: the P1 cell-6 board readings, written BEFORE the boot

Both cells are pre-registered here: cell 6 first, cell 5 in its own section below. Cell 5 was
added after the first version of this document, which said cell 5 was excluded because it did
not build; the last section records why it did not and what changed.

## The image this pre-registration is about

    sha256      6f94d9891918d0b56c1670b40943fd1104a88d17232002b242aaacbb7e5cda11
    size        1,685,984 bytes
    entry VA    0x10000        <- sits at 0x10000, so the relinked k800 is the one to use
    built from  08ff5d0702c3b6547da8d64cf66a9ca2f4bd75e3 (dev), carrying D-prime and C-50
    toolchain   clang 22.0.0git @ 08ff5d0702c3, with #94 CONFIRMED ABSENT
                (llc --help-hidden reports no capstone-recover-provenance flag)
    built via   run-speedtest1-measure.sh, SPEEDTEST1_SUBLET=1, SQLITE_LOOKASIDE=1200,40
    workload    --testset main --size 1 --verify

ONE IMAGE, TWO ARENAS. arena and tables are HOST RUN-TIME arguments, not compile-time defines, so
the two arena configurations below are the SAME image run twice -- identical sha256. That is what
board-c6var.sh needs, since it gates on QEMU records at both.

    geometry A (the boot's denominator)  arena 2097152, tables 1750285
    geometry B (the default)             arena 1419584, tables 1750285

Tables stay at 1,750,285 in BOTH. That is deliberate and is the trap the runner documents at
:167-177: reaching a 2 MiB arena by raising SPEEDTEST1_POOL instead drags tables to 2,523,136 and
STILL satisfies the SPEEDTEST1-CYCLES gate, because that line does not carry tables at all. The
arena and tables were therefore set as separate overrides. Arithmetic checked: 1441792/65 = 22181
atoms, 22181*57 + 22181*16 + 131072 = 1,750,285.

## The emulator readings this image produced

    geometry A   SPEEDTEST1-CYCLES 697366227 HIGHWATER n/a HEAP 1344064
                 sublet: split=5568 mrev=37966 delin=32565 revoke=37966 init=5401
    geometry B   SPEEDTEST1-CYCLES 695015338 HIGHWATER n/a HEAP 911104
                 sublet: split=5481 mrev=37874 delin=32565 revoke=37874 init=5309

Both HEAP values are what board-c6var.sh expects (1344064 at the 2 MiB arena, 911104 at the
default). HEAP is sublet_heap_len() derived from the GRANTED ARENA, not from the static heap.

LOOKASIDE IS ON, proven from the run and not from the build flag, which is the rule that block of
the runner was written about:

    Successful lookasides:   25010
    Lookaside Slots Used:    1

## The predictions

1. The board's counters for this image at geometry A EQUAL the emulator's:
   split=5568 mrev=37966 delin=32565 revoke=37966 init=5401.
2. At geometry B they equal split=5481 mrev=37874 delin=32565 revoke=37874 init=5309.
3. A --stats run on the board shows lookaside ON, "Successful lookasides" non-zero.
4. Cycle counts will NOT match the emulator's and are not predicted: icount is not silicon timing.

CAVEAT THAT WOULD OTHERWISE LOOK LIKE A DEFECT. The counters above are from runs WITHOUT --stats.
Adding --stats shifts them, because --stats itself allocates: the same image at geometry A with
--stats gave split=5568 mrev=37970 delin=32569 revoke=37970 init=5401 -- mrev, delin and revoke each
+4. So compare like with like: a board run with --stats must be compared against the --stats
numbers, not against prediction 1. The image hash is unchanged either way; verified.

## Falsifiers

- Counters differing from the matching configuration above -- that is the claim, and it fails.
- Lookaside reported off, or "Successful lookasides" zero: the pool was compiled in but never used,
  which is exactly the two-variable failure the runner's comment records for the sixth cell.
- HEAP reading something other than 1344064 / 911104 for the respective arena: the granted arena is
  not what this pre-registration assumes and the denominator is wrong.

## The static check, and its control

movc-cfg-scan.py on the LINKED image: 6810 movc(rd!=rs), 3004 with the source read again,
**0 INT-ONLY, 0 mixed, no setupLookaside site**. 554 sites classified as call-return-or-argument.

THAT ZERO HAS NO SURVIVORS, so the image alone cannot show the scan would report a site if one
existed -- unlike the earlier bring-up image, which retained relocatePage and
whereLoopAddBtreeIndex as in-situ controls. The control is therefore external and was run: the SAME
scanner and the SAME compiler over the PRE-D-prime source report

    [INT-ONLY]      setupLookaside+0x3a8: movc a0, s3 -> read @0x3fc mv a0, s3
    [MIXED cap,int] setupLookaside+0x720: movc a0, s3 -> read @0x774 movc a0, s3

So the scanner does find that site when it is there, and its absence from the shipped image is a
property of the image rather than of the instrument.

A static scan is necessary, not sufficient: the scanner's own header says taggedness is dynamic.

## Cell 5

Built on the ESTABLISHED recipe, not a new geometry: SPEEDTEST1_HEAP=2097152 with
SPEEDTEST1_STACK=385024, which is what lets it load under the module's one-region rule
(measurements doc :1701). The stack asymmetry against cell 6 is ALREADY RECORDED at :4758 --
385,024 declared for the memsys5 arm against cell 6's 1,048,576 -- so it is not introduced here.

    sha256      ff577d44e3ee8dd3b356a0e9ce8ffa89da11a1884ae6e9df8b7095e4bb5b5891  (1,684,216 bytes)
    entry VA    0x10000
    emulator    SPEEDTEST1-CYCLES 678534902 HIGHWATER n/a HEAP 2097152   (canonical, no --stats)
                SPEEDTEST1-CYCLES 678681249 HIGHWATER n/a HEAP 2097152   (the same image WITH --stats)
                An earlier version of this document quoted 678681249 as the figure. That is the
                --stats run: the --stats runs reused the canonical tmp roots and overwrote
                sqlite-speedtest1.log, so the contaminated number was the one to hand. Canonical
                is 678534902, reproduced from a clean root. Which one a boot compares against
                depends on the boot: board-b80s.sh runs the cell WITH --stats and must use
                678681249; a run without --stats uses 678534902.
    lookaside   Successful lookasides 25010   (from the run, not the build flag)
    no sublet line, as it must be: this arm is memsys5, not Sublet
    budget      declares dom_data >= 2,701,968 (carve 2,316,944 + stack 385,024) -- FITS
                (the doc records 2,701,904 for the archived build; +64 bytes of carve here)

PREDICTIONS for cell 5 on the board: lookaside ON with a non-zero "Successful lookasides", and NO
sublet counter line at all. A sublet line appearing in this arm would mean the wrong image booted.

ONE TREE, stated precisely because the two cells report different build commits. Cell 6 was built
from 08ff5d0702c3 and cell 5 from fd6da67aea96. The ONLY difference between those trees is
capstone/ports/sqlite/fetch-sqlite-src.sh -- zero files under llvm/ or clang/, and no other file
under capstone/. That script only chooses which SQLite checkin to download, and both cells consumed
the SAME extracted tree (manifest d4c0e51e4aeb...), so the compiled inputs are identical rather than
merely equivalent. fd6da67aea96's content is also identical to dev df3a8e57df42, where it landed.

## Historical note: why cell 5 first failed to build

cell 5 did not build at first. With no explicit stack declaration the domain budget refused it:

    pass 2  declared nothing              order 10, FITS, but stack only 392,064
    pass 3  declared dom_data >= 3365520  order 11 -- exceeds the kernel maximum 10, DOES NOT FIT
            largest code_len that still fits: 1,376,256; this image is 1.08x that

Diagnosed against cell 6 rather than guessed:

    cell 5   code_len 1,483,752   storage 2,206,112   DOES NOT FIT
    cell 6   code_len 1,483,896   storage   109,280   fits

The code sizes are identical to within 144 bytes, so this is NOT code growth and specifically NOT
C-50, whose cost on this translation unit was measured at ~144 bytes. It is cell 5's 2 MiB STATIC
memsys5 heap in .bss: once a 1 MiB stack is declared on top of it, dom_data needs order 11.
Cell 6 escapes it because its arena comes from a region rather than .bss.

It was resolved without a new geometry and without a decision: declaring SPEEDTEST1_STACK=385024
is the recipe the measurements doc already records at :1701, and the resulting asymmetry against
cell 6's 1,048,576 is already tabulated at :4758. So the pair compares what it always compared.
The alternative that WOULD have needed the lead -- picking some new stack size to make it fit --
was not taken.

## Where the artifacts are

    ~/capstone-artifacts/p1-O2-2026-09-29/

Both images, every QEMU log behind the figures above, a README mapping each board driver's
environment variables onto the files that satisfy its gates, and a self-verified SHA256SUMS. The
per-image qemu-pass records under ~/capstone-artifacts/qemu-pass/ point their log= into that folder.

TWO THINGS A BOARD DRIVER NEEDS THAT ARE EASY TO MISS. board-c6var.sh greps the
`== Sublet: pool <arena> bytes (arena, REV_BORROWED), tables <tables> bytes` line, which appears in
the measure run's HOST STDOUT and in no guest log; both cell-6 host stdouts are staged as
cell6-hoststdout-*.log for exactly that gate. And board-b80s.sh additionally greps a native -O2
baseline for `BASELINE-WARM CYCLES 240654449`: that log is not part of this pair and was not found
under ~/capstone-artifacts, capstone/ or /tmp/capstone -- it has to be located or re-run before a
cell-5 boot.
