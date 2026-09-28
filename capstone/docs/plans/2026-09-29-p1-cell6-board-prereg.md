# Pre-registration: the P1 cell-6 board readings, written BEFORE the boot

## RETRACTED 2026-09-29: this pair is -O0, NOT -O2

Everything in this document that says or implies -O2 is wrong, including the staged folder's name
(`p1-O2-2026-09-29`) and both image filenames (`cell5-memsys5-O2.dom`, `cell6-sublet-O2.dom`). The
two cells were built at the DEFAULT optimisation level, which build-sqlite-silicon.sh:42 sets to
-O0. No script in the build path ever set SQLITE_OPT_LEVEL; -O2 existed only in names.

Established by rebuilding each cell exactly as it was built, with QEMU stubbed out so no run and no
lock were involved:

    staged cell 6                    6f94d9891918d0b56c1670b40943fd1104a88d17232002b242aaacbb7e5cda11
    rebuilt, no OPT set              6f94d9891918d0b56c1670b40943fd1104a88d17232002b242aaacbb7e5cda11   1,685,984 B
    rebuilt, SQLITE_OPT_LEVEL=-O2    df484d98b489aeab1e5b33f1bafd4693cf5835b37da27916c15b1327993f159f   1,379,064 B

    staged cell 5                    ff577d44e3ee8dd3b356a0e9ce8ffa89da11a1884ae6e9df8b7095e4bb5b5891
    rebuilt, no OPT set              ff577d44e3ee8dd3b356a0e9ce8ffa89da11a1884ae6e9df8b7095e4bb5b5891   1,684,216 B

Both staged images reproduce byte-for-byte from the default build. The -O2 arm is the negative
control and is a different, much smaller image, so the flag is live and the pair simply never
received it -- without that arm the reproduction would only have shown the build is deterministic.

Corroboration that was available hours earlier and was misread: board-b80a.sh records the old -O2
cell 5 at 330,723,308 instructions against this one's 678,534,902, a hair over 2x. That gap was
attributed to a SQLite version difference; the version has been 3.53.3 throughout.

WHAT IS AND IS NOT AFFECTED. Every measured figure below -- cycles, HEAP, sublet counters,
lookaside, image hashes, entry VAs -- is correct as measured, and so are the predictions and
falsifiers. Only the LEVEL those artifacts were built at is misstated. Nothing has been renamed and
nothing deleted: under the resolution below the -O0 artifacts are superseded rather than corrected,
and they are worth keeping as the -O0 reference they actually are.

CONSEQUENCE FOR THE NATIVE BASELINE. build-speedtest1-baseline.sh's header requires the baseline to
be built at the domain's level, because one built at a different level "measures the optimiser, not
the ABI" -- it cites five bogus silicon failures from exactly that mismatch. So the baseline is
built at -O2 too, and not before the cells are.

IT ALSO VOIDS A CLAIM, NOT ONLY A LABEL. The static movc scan recorded below reports
**0 INT-ONLY, 0 mixed, no setupLookaside site** for cell 6. That scan ran on an -O0 image, and C-32
does not manifest at -O0 -- so the zero is a property of the optimisation level, not evidence about
D-prime. It is the clean-result shape this project keeps paying for: a check that cannot fire on the
input it was given. The scan has to be re-run on the -O2 image, with the pre-D-prime -O2 control
firing at setupLookaside, before it says anything.

RESOLUTION: REBUILD AT -O2. Settled, not a preference --
paper-nested-allocators/experiments/protocols/hardware/P1-application-cost.md:76-79 reads "Use
optimization level O2 in every capability arm and identical common flags ... The existing O0 replays
are not timing baselines for this study." Everything staged under p1-O2-2026-09-29 is therefore
superseded rather than relabelled, and is kept only as an -O0 reference. The -O2 figures land in a
later commit on this branch; both commits stay, because the trail is the evidence.


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

# ============================================================================
# THE -O2 PAIR, which supersedes everything above
# ============================================================================

Built 2026-09-30 with `SQLITE_OPT_LEVEL=-O2` passed explicitly on every arm, one tmp root per arm,
each measure run under its own `flock` on the QEMU lock. Staged at
`~/capstone-artifacts/p1-O2-2026-09-30/` with a self-verified SHA256SUMS and a
`check-driver-gates.sh` that evaluates every driver expression with a negative control on each.

## The images

    cell5-memsys5-O2.dom   5f4b44257c93347738a16e2b21596bbe7033b642b4d14b0bb5b58ae5cbaa1016
                           1,378,344 bytes   entry VA 0x10000
    cell6-sublet-O2.dom    df484d98b489aeab1e5b33f1bafd4693cf5835b37da27916c15b1327993f159f
                           1,379,064 bytes   entry VA 0x10000
    native-baseline-O2     9a80c1cd2a576ed2b7732350b9ebb206515303844b0518e4b9c34c4fa2ac1bd1
                           793,584 bytes

Both cells enter at 0x10000, so neither collides with the k800 control relinked at 0x20000.

## The emulator readings

    cell 5   canonical  SPEEDTEST1-CYCLES 330684651  HEAP 2097152
             --stats    SPEEDTEST1-CYCLES 330731081  HEAP 2097152
    cell 6   arena 2097152  SPEEDTEST1-CYCLES 341028517  HEAP 1344064
                            sublet: split=5568 mrev=37966 delin=32565 revoke=37966 init=5401
             arena 1419584  SPEEDTEST1-CYCLES 338707825  HEAP 911104
                            sublet: split=5481 mrev=37874 delin=32565 revoke=37874 init=5309
             --stats @2 MiB SPEEDTEST1-CYCLES 341075473  HEAP 1344064
    baseline BASELINE-WARM CYCLES 240714705 INSTRS 240714707, 25,122 successful lookasides

THE SUBLET COUNTERS ARE UNCHANGED FROM THE -O0 RUN, to the event. That is expected and is a
cross-check rather than a coincidence: they count allocator events, not instructions, so the
optimisation level cannot move them while the workload is the same.

## Three independent corroborations that these really are -O2

Each is a figure recorded by somebody else, before this rebuild, that the new numbers land on:

    board-b80a.sh  old -O2 cell 5      330,723,308   new 330,684,651   0.012% apart
    board-b80b.sh  old -O2 @ 911104    338,496,909   new 338,707,825   0.062% apart
    board-b80b.sh  old -O2 @ 1344064   340,817,186   new 341,028,517   0.062% apart
    board-b80a.sh  old native baseline 240,654,449   new 240,714,705   0.025% apart
    board-b80a.sh  baseline lookasides      25,122   new      25,122   EXACT

The -O0 pair read 678,534,902 for cell 5 against the same 330.7 M record -- a hair over 2x, which
is the gap that should have been read as an optimisation-level difference on the day.

## THE C-32 CHECK: lookaside is ON at -O2

    cell 5  Successful lookasides: 25010   Lookaside Slots Used: 1
    cell 6  Successful lookasides: 25010   Lookaside Slots Used: 1

On the pre-D-prime -O2 build this read 0: the pool was compiled in and never used, because
setupLookaside's pStart was nulled. This is the run-time half of the D-prime claim and it is the
half that could not be obtained at -O0 at all. The staging script refuses to write the folder if
either cell reports zero.

## The movc scan: WHAT WAS PREDICTED AND WHAT WAS FOUND

The prediction was written before the -O2 images existed, from the D-PRIME note in
`capstone/ports/sqlite/sublet/sublet-3530300.patch`, which records 4 mixed / 0 INT-ONLY before
D-prime and 2 mixed / 0 INT-ONLY after, the two removed being setupLookaside and the two remaining
being relocatePage and whereLoopAddBtreeIndex "the control that shows the scan still fires".

    PREDICTED   0 INT-ONLY, 2 mixed = relocatePage and whereLoopAddBtreeIndex, no setupLookaside
    FOUND       1 INT-ONLY, 1 mixed, no setupLookaside -- at two DIFFERENT sites

    cell 6 -O2: 17246 movc(rd!=rs); 10201 with the source read again; 1 INT-ONLY, 1 mixed,
                1768 call-return-or-argument
       [MIXED cap,int] main + 0x3ac44   movc a1, s5  -> read @0x3acfc mv a1, s5
       [INT-ONLY] renameResolveTrigger + 0x10b9c0    movc s11, s10 -> read @0x10b9c4 jalr s10

THE PART THAT MATCHED IS THE PART THE PRE-REGISTRATION WAS ABOUT: no setupLookaside site, on an
image where the scan demonstrably fires (it returned two sites, and it reproduces the -O0 image's
6810/3004/0/0/554 on demand). Combined with 25,010 lookasides at run time, D-prime holds at -O2.

THE PART THAT DID NOT MATCH WAS THE PREDICTION, NOT THE SCAN — see the correction below, which
supersedes this paragraph. Kept as written because the pre-registration is only worth anything if it
records what was predicted before the result. Two things were established about the two sites:

- The scanner prints `<function> + 0x<ABSOLUTE ADDRESS>`, not an offset within the function. The
  INT-ONLY site is at absolute 0x10b9c0, which is renameResolveTrigger + 0x654; the disassembly
  there is `movc s11, s10` / `jalr s10`, exactly as reported. The location is real.
- **Both sites appear identically in cell 5**, which is memsys5 without Sublet and therefore
  carries no D-prime at all (1 INT-ONLY, 1 mixed, same two functions). So they are common code and
  are not a D-prime residual.

Both statements above are correct. The conclusion drawn from them — that the sites were unexplained
and needed their own investigation — was not, and is withdrawn in the next section.

## Predictions for the board

1. Cell 6 at the 2 MiB arena reproduces split=5568 mrev=37966 delin=32565 revoke=37966 init=5401.
2. Cell 6 at the default arena reproduces split=5481 mrev=37874 delin=32565 revoke=37874 init=5309.
3. HEAP reads 1344064 and 911104 respectively; cell 5 reads 2097152.
4. Both cells report "Successful lookasides" non-zero under --stats. A zero is C-32 returning.
5. Cell 5 shows NO sublet counter line. One appearing means the wrong image booted.
6. Cycle counts are NOT predicted: icount is not silicon timing.

## Falsifiers

- Any counter differing from its configuration above.
- Lookaside zero on either cell.
- HEAP other than the values above: the granted arena is not what this assumes.
- A sublet line in cell 5, or none in cell 6.

## Re-pinning the drivers

board-b80s.sh and board-b80a.sh hard-code the OLD images and figures (d61c8bf784f2bbd1,
b36eb3814c3cefce, 330723308, 240654449), and board-b80b.sh its own. All of those images are gone
from disk. A boot on this pair runs a copy of the driver with every constant re-pinned to the
records above; `check-driver-gates.sh` in the staged folder evaluates the re-pinned values.

## CORRECTION: the scan matched the -O2 record exactly; the PREDICTION was drawn from the wrong image

Neither site is new, and neither is unexplained. Both are C-32's documented dormant residue, and the
prior art says so in three places that were not searched before the prediction was written:

- `docs/ref/ISSUES.md`, in the C-32 box, the bullet beginning "**Left as they are:**" —
  "`renameResolveTrigger` and `main` are dormant under the P1 workload and are present in the native
  cell 5 as well." (:1730 as of 08ff5d07 and dev; the number moves with every registry edit, so the
  quote is the anchor.)
- `docs/ref/fpga-silicon-measurements-for-paper.md:4554-4562` records the site census per image:
  "Sublet -O2 **4 sites** (`setupLookaside` x2 — the live one and its free path; `main`+0x3aab4,
  mixed; `renameResolveTrigger`+0x10b5b8, a self-loop on the ALTER TABLE path)", and "cell 5 -O2
  **2**". D-prime removes the two `setupLookaside` sites. What remains is exactly 1 mixed in `main`
  plus 1 INT-ONLY in `renameResolveTrigger` — which is what this scan found, in both cells, with the
  offsets shifted by the rebuild.
- `docs/plans/2026-09-15-consolidated-board-queue.md:283` already carried the reachability argument:
  "`renameResolveTrigger` is reached per trigger and the main testset defines none".

WHY THE PREDICTION WAS WRONG. It was taken from the D-PRIME note in the sublet patch, which names
`relocatePage` and `whereLoopAddBtreeIndex` as the two survivors. That note describes a DIFFERENT
image: the same measurements table lists `whereLoopAddBtreeIndex` under **Sublet -O1**, not -O2. A
prediction sourced from one build's census cannot be checked against another's, and the mismatch was
a property of the source, not of the result.

The ALTER-path proof recorded above is therefore not a new finding but a stronger form of the
:283 sentence: per-trigger reachability, argued from the four callers of `renameResolveTrigger`
being the `sqlite_rename_*` SQL functions, which ADD COLUMN never emits.

The lesson is the cheap one: the function name was in the registry and in the measurements doc the
whole time, and neither was grepped for it before the site was raised as a possible live defect.

## The nulling-mode verification, and a one-instruction difference

Both cells re-run under a QEMU that implements Q-04's movc-nulls-untagged-source semantics, which is
what silicon does. The default binary does not implement the switch at all — it contains neither
"MOVC-NULL-SCALAR" nor "CAPSTONE_MOVC_NULL_SCALAR" — so the earlier runs could not have observed it.

    positive control  pre-D-prime -O2, image 5627abb5: lookasides 0 (against the pair's 25,010),
                      hash 112006 38bb59fd. The mode demonstrably reaches the domain.
    cell 5 --stats    330,731,082 against 330,731,081 in default mode      delta +1
    cell 6 --stats    341,075,474 against 341,075,473 in default mode      delta +1

Everything else is identical in both cells: verification hash, 25,010 successful lookasides,
1 lookaside slot used, 16,448 schema heap, HEAP, and cell 6's sublet counters
5568/37970/32569/37970/5401.

THE +1 IS ATTRIBUTED TO THE MONITOR, NOT PROVEN TO BE IT:

- the measured window counts all privilege levels (`csrr mcycle`) and the domain traps to the
  monitor throughout, so monitor instructions are inside it;
- the emulator reported an actual nulling event in the monitor at pc=0x80020a50, priv=3, on the
  `read_cpmp` path — `ld s1, 0x0(s1)` then `movc a0, s1`;
- `sqlite_host.user` cannot contribute: it contains no `movc` at all, checked against the raw
  disassembly and not only via the scanner;
- the domain image's own scan is clean of reachable integer-sourced sites — but only MODULO the
  scanner's documented blind spot — `ISSUES.md`, the C-32 audit bullet "a bridged value stored with
  `stc` and reloaded with `ldc` is invisible to it" (:1726 as of 08ff5d07 and dev) — so that leg is a
  static argument with a known hole rather than a proof.

A null domain bracketing `mcycle` around no work would close the hole. It was not run: it would
re-derive a conclusion the existing binaries already support, and the GO does not rest on it.

THE GO DOES NOT REST ON THE +1. This pre-registration predicts no cycle count — prediction 6 says so
explicitly — and every result field it does predict is identical under nulling. A nulled jump target
would have crashed or changed a result; nothing changed.

## An instrument defect found while doing this

`movc-cfg-scan.py` reported 0 movc for `fw_jump.elf`, a file with 94 of them. `llvm-objdump`
right-aligns the address column, so an image at 0x80000000 starts at column 0 while a domain at
0x10000 is indented, and the instruction regex required the indent — every image linked above
0x0fffffff scanned as perfectly clean. Fixed on `lane/compiler-movcscan-wide-addresses` (641749752f67),
together with making an empty parse exit non-zero instead of printing zeros. No recorded claim rests
on it: every documented scan is of a domain image linked at 0x10000 or 0x410000, and the monitor had
never been scanned.

### Citation note

The ISSUES.md line numbers in the section above were briefly given as :1652 and :1648. Those are
true only in the shared checkout at the repository root, which was 53 commits behind
dev; in this branch's own tree and on dev they are :1730 and :1726. The grep had been run in the
shell's working directory rather than in the tree the document belongs to, and `ISSUES.md` is
precisely the file that churned in between. The two other citations here — the measurements doc's
site census and the consolidated queue's reachability sentence — were re-checked in all three trees
and are identical in each, so only the registry ones moved.

Both are now anchored by quote. A line number into a registry that is edited daily is a citation
with an expiry date on it.
