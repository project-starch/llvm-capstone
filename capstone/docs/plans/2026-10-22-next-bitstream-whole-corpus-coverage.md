# What the batch bitstream will not read, and the live-record demand behind it (2026-10-22)

This is the input for sizing the NEXT bitstream, which the lead wants to read the whole corpus, for the RTL lane's
sizing and the runtime lane's module change. It is an enumeration over the batch kit's QEMU references, plus one new
measurement: the live-record peak and lifetime allocation count of every cell that the 64,001-record flat table stops.

## What was searched

- **The batch references:** 27 arms on the batch kit qkit7-batch2, which pairs the QEMU binary ecaf50bbd5ec44aa
  (capstone-qemu batch-v7-r60sw a36498f352: R-59, R-60, generations, compact rounding) with the flat module
  a05c198bfcc5f55c.
  - The arms: fca, fcs, fph, fpt, fso, lt0, lt1, mph, mpt, pb9, pb10, pb12, pb13q15, pg-pools, pgc, pgm0, pgm1, pgm1q15,
    pym0, pym1, sq0, sq1q15, sql0, wm0, wm1q15, wph, wpt.
  - Each arm's `verdicts.jsonl` was read cell by cell.
  - The CPython images are lt1 916e14f9b3f622c9 and lt0 23a2f0cbd80ddfbe, on the r60 cw SDK.
- **mruby and perl, stock and Sublet:** the same kit's MP references (images f8594e11, ba42d357, 76f1141d, bec05c86).
  - Every cell reads in all four arms.
  - Only Sublet mruby's control smoke fails (under (a) below).
- **The Juliet temporal set and the 755-case spatial sample:** both judged on QEMU (the silicon-mimic kit), and run on
  both vp2 trees by the board lane (dev 3ceba9bad107).
- **Corpus coverage:** `bug-corpora/index.json` (28 corpora, 327 cases), matched case name by case name against the
  arms' cells.
  - Every arm's cells fall in exactly one corpus, and none is unmatched.
  - Every corpus that has an arm has its whole case list in at least one arm.

**Not searched:**
- the board's own batch run plan (whether it schedules every arm above);
- any arm outside these references.

## (a) Record capacity: live records above 64,001

**Ten CPython cells**, all from lt1 (the pymalloc-pools arm), are stopped by the flat table on the batch kit:
- six with cause 30 in `_PyObject_Calloc`: 02, 13, 23, 27, 28 and 30;
- three that surface the refusal as a Python `MemoryError` (they read NO-READING:infra): 10, 19 and 31;
- one that surfaces it as `SystemError: error return without exception set`, during an import (NO-READING:infra): 18.

(Corrected 2026-10-22: the first version said nine. lt1 18 was classed by its error text, as an import failure. The
module's high_water at that failure, 64,023 against capacity 65,536, places it at the limit. The 16-/18-bit twin pair
and the paged kit confirm it: capstone/tests/corpus-cell-probes/README.md, pre-registered at ab3ea2c6a4d3. A capacity
census must classify by the table's reading at the failure, not by the error a program prints.)

The three `MemoryError` cells were classified by the measurement below. On a table without the cap the `MemoryError`
disappears, and each case reaches a capability fault in its own code, with a peak above 64,001.

### The measurement

- **The instrument:** capstone-qemu `measure/live-peak` (ea7bcd757e, then 730a8b2287), a36498f352 plus host-side
  counters in `cap_rev_table.c`.
  - live: +1 when a record's VALID bit turns on, −1 when it turns off. The peak is its maximum.
  - minted: one per allocation (the lifetime count).
  - Measurement only: off unless `CAPSTONE_LIVE_PEAK` is set, and it writes nothing to guest memory.
- **The kit:** qkit7-batch2-paged-lp, which is the batch kit with:
  - the PAGED twin module ed8565728d0fdabd (the same source as a05c198b, without the flat table, so there is no cap);
  - the test CSRs on, which the paged module needs;
  - only the QEMU binary swapped, for the counter build.
- **The runs:** one boot per cell, replaying the r60 lt1/lt0 session's calls for that cell (the case and its
  negative control), on QEMU domains A and B.
- **The readings are brackets, not points.** The staged runner stops reading QEMU at the gate's end marker, so the
  counters print as they go (`capstone/tests/live-peak-sizing/lp-parse.py`):
  - peak: within 64 once live falls from it, and below the next 1,024 step that was never printed;
  - minted: to within 4,096.
  - The module's own allocation counter is exact, and is the independent channel. Every one of the 20 calls' module
    counts lies inside the host bracket.
- **The control, pre-registered in ea7bcd757e, held in every check** (`control.txt`):
  - a program holding 100,000 objects read a peak of 105,569, exactly equal to the module's allocations;
  - 200,000 rounds of malloc/free read a peak between 31 and 1,023, with 806,147 allocations: about 4 records per
    round, because each free unmaps and re-maps the group;
  - holding 150,000 objects read [158,303, 158,719].
  - The gate's negative test: the first, void boot's log (no MINT lines) fails its allocation check.

### Results, per call (`capstone/tests/live-peak-sizing/cells.tsv`)

| cell (lt1) | call | result on the uncapped table | live peak | lifetime allocations (module, exact) |
|---|---|---|---:|---:|
| 02 | case | cause 28 in memcpy | 106,198–106,495 | 280,614 |
| 02 | control | exit 0 | 74,766–75,775 | 281,183 |
| 13 | case | cause 28 in unicode_fromformat_write_str | 106,272–106,495 | 281,516 |
| 23 | case | cause 25 in memchr | 74,972–75,775 | 267,453 |
| 23 | control | exit 0 | 72,869–73,727 | 264,737 |
| 27 | case | cause 25 in slot_nb_add | 71,693–72,703 | 278,149 |
| 27 | control | exit 0 | 71,765–72,703 | 281,162 |
| 28 | case | cause 25 in _PyEval_Vector | 76,506–76,799 | 313,843 |
| 28 | control | exit 0 | 76,562–76,799 | 316,941 |
| 30 | case | cause 25 in PyObject_RichCompare | 76,529–76,799 | 314,758 |
| 30 | control | exit 0 | 76,532–76,799 | 316,373 |
| 10 | case | cause 28 in l_divmod (MemoryError on the flat kit) | 80,083–80,895 | 219,402 |
| 10 | control | exit 0 | 80,090–80,895 | 220,865 |
| 19 | case | cause 25 in fnv (MemoryError on the flat kit) | 65,177–65,535 | 174,939 |
| 19 | probe | cause 25 in fnv | 64,603–65,535 | 173,796 |
| 19 | control | exit 0 | 64,690–65,535 | 177,214 |
| 31 | case | cause 25 in odict_copy (MemoryError on the flat kit) | 74,613–74,751 | 233,845 |
| 31 | control | exit 0 | 82,364–82,943 | 348,197 |
| 18 | case, with application mremap relinked in | cause 28 in mmap_ass_subscript (SystemError on the flat kit) | 66,682–67,583 | (module high_water 66,724 on the 18-bit twin) |

Reference row: mruby Sublet's control smoke at M6=20000 (not a corpus cell) peaks at 122,868–122,879 live records,
with 164,800 lifetime allocations. Its records are unswept garbage, which mruby's GC frees only on its own heap pressure.
`smoke-fit.rb` on dev (e4291ea0ce13) runs M6 at 6,000, which fits 64,001.

**Not judged here:** whether each case's fault is at its declared site. That is the arm's judge, run on the read pass.

### What it means for the index width

Audited before sending (claim-auditor, 2026-10-22). Its corrections are folded in below.

- **The largest corpus demand is exactly 106,272 records (lt1 13), then 106,198 (lt1 02).**
  - Both are pinned by the last MINT line plus the module's exact count. For 13: live was 72,019 at 278,528 allocations,
    with 2,988 to go, so it cannot pass the printed peak afterwards.
  - The 139,780-143,877 figures were the paged table's EXTENT.
  - The lifetime counts reproduce exactly across two boots on two binaries (paged-sizing and this run): 280,614,
    281,516, 267,453 and 278,149 for 02, 13, 23 and 27. The peak itself is one boot per cell.
- **On a flat table with generations, occupancy = the VALID count + at most lifetime/2^g last-generation slots.**
  - That is from the source. Every invalidating write pushes the record: `cap_rev_tree.c:262-265` and `:343-346`.
    Every allocation is followed at once by a VALID put, and the module never writes a record.
  - **Confirmed on the flat kit** (discriminator below). At the cause-30 refusal, the counter's live records equal the
    module's own count of issued records: 64,256 and 64,256, with no collection.
- **Usable records, not index space.** The module keeps a 1,024-record floor and holds back a reserve. Measured on the
  flat kit, the table stops at two counts, depending on the path:
  - a cell that derives directly (the pools' `_PyObject_Calloc`) faults with cause 30 when 256 records are left: 64,256;
  - the malloc path refuses at 511 left (64,001), as mruby's smoke did on 2026-10-21. That one surfaces as NULL. It is
    presumably how the `MemoryError` cells stop, but their flat-kit counts were not read.
  The figures below use the larger overhead, 1,535.
  - 17-bit index: 131,072 − 1,535 = 129,537 usable. That is 1.22x over 106,272, with a leak of at most 43 slots at 13
    generation bits.
  - 18-bit index: 262,144 − 1,535 = 260,609 usable. That is 2.45x, with a leak of at most 85 at 12 bits.
  - The mruby M6=20000 reference smoke (122,879) fits 17 bits at 1.05x.
- **Scope.** This covers every cell the batch kit read (each peaks at or below 64,001, or it would have stopped there),
  plus these ten.
  - No demand reading exists for cells that stopped before their body: lt0/lt1 11, lt0 18, pb13q15 05 and 08.
  - lt1 18 is read with application mremap relinked in. RETRACTED: the earlier version called its stopped run's peak
    (67,875–68,607, batch image) "a lower bound". It is not one. That run's peak is CPython's error path after the
    failed resize, which allocates more than the defect path, which peaks at 66,682–67,583.
  - Nor for the corpora with no arm, in (c).
  - sq0 23 is a genuine MISSED, not a capacity stop. It prints `VERDICT len_changed=0 sum_changed=1
    first4_changed=1` and then `NOTRAP`: the stale text was read and no trap fired. The `out of memory` comes after
    that, from the integrity check on the corrupted table. (Corrected 2026-10-22: the first version presumed it was
    memsys5's 1 MB heap.)

### The flat-kit discriminator (pre-registered at dfc80198307c, before its boot)

- **Run:** the same counter build on the FLAT batch kit (qkit7-batch2-lp: module a05c198b, QEMU bb96f695), cells
  lt1 02 and 13, one boot each.
- **Prediction for each case:**
  - it faults with cause 30 in `_PyObject_Calloc`, as on the batch reference;
  - its peak bracket lies inside [63,000, 64,001];
  - its lifetime allocations at the fault are in [130,000, 145,000]. The paged trajectory crosses 64,001 live at about
    137,600 allocations.
- **If the peak reads below 63,000:** the flat table holds more than 1,000 records beyond the VALID count. The sizing
  above must then add that excess, and the "occupancy = VALID" line is withdrawn.

**Result** (QEMU bb96f69552135b93, module a05c198bfcc5f55c):

| cell (lt1) | call | result | live peak (counter) | module high_water | lifetime allocations (module) |
|---|---|---|---:|---:|---:|
| 02 | case | cause 30 in `_PyObject_Calloc` | 64,256 (exact, closed by a RESET) | 64,256 | 137,902 |
| 02 | control | cause 30 in `_PyObject_Calloc` | 64,256–64,511 | 64,256 | 126,359 (from the running total) |
| 13 | case | cause 30 in `_PyObject_Calloc` | 64,256–64,511 | 64,256 | 137,904 |

- **HELD:**
  - cause 30 in `_PyObject_Calloc` on every case;
  - lifetime allocations in [130,000, 145,000]: 137,902 and 137,904, where the paged trajectory predicted about
    137,600.
  - The discriminating reading holds too: live equals the module's issued count exactly, with collections 0, so the
    flat table holds nothing beyond the VALID records.
- **NOT HELD: the band's upper edge (≤ 64,001).** The peak is 64,256.
  - The band had used the malloc path's limit; this path faults at 256 records left.
  - That is a wrong constant in the prediction, not a sign of excess occupancy. An excess would push the peak BELOW the
    band.

## (b) Every other cell the batch references do not read

The causes below are what the records show. "Not investigated" means exactly that.

### Infrastructure, not-reached and control-failed

| arm | case | reading | cause, as recorded |
|---|---|---|---|
| lt0, lt1 | 11 gh-143005 (ctypes array) | NO-READING:not-reached | `ModuleNotFoundError: No module named '_ctypes'`. The CPython port is built without `_ctypes`, so the trigger stops before its case. A port gap, not a platform one. |
| lt1 | 18 gh-157335 | NO-READING:infra | `SystemError: error return without exception set`, raised by `import select` at subprocess.py:125, while upstream_test.py imports test.support.script_helper. RECORD CAPACITY, the tenth cell in (a): the module's high_water reads 64,023 at the failure, and the error is gone on the 18-bit twin and on the paged kit. Beyond it the case needs application mremap, as lt0 18 does. |
| lt0 | 18 gh-157335 | MISSED, but NOT REACHED | The judge reads MISSED, yet the defect never runs. In the test's `self.m.resize(new_size)`, `mmap.resize` raises `OSError: [Errno 38] Function not implemented`, and the trigger exits 1. Public application `mremap` is unsupported on the virtual profile (runtime/virtual/README.md: "Address hints/fixed mappings and public application `mremap` remain unsupported"), and CPython resizes an anonymous map only through mremap. So the case needs that runtime feature before either arm can read it. With the runtime lane's mremap (runtime/app-mremap c26b60a61f5b) relinked into the lt0 image, it reads CAUGHT: cause 28 in `mmap_ass_subscript`, a declared site (capstone/tests/corpus-cell-probes/README.md). |
| lt0 | 13 gh-144169 | NO-READING:infra | `MemoryError` in the upstream test's `ast.Name(**{object(): 'y'})`. NOT record capacity: on the uncapped table it persists at a live peak of 7,753–8,191 and 20,810 allocations, with no refusal printed. On lt1, with per-object pool bounds, the same case faults with cause 28 in unicode_fromformat_write_str. The likely reading, NOT measured: on lt0's single-capability pymalloc arena the defect reads a non-str key as a str and requests an oversized buffer. |
| pb13q15 | 05 2d61f18 | NO-READING:control-failed | The FIXED run faults with cause 24 (address 0) in `capstone_cap_revoke_child` (image offset 0x24588), at a free. The case's canary does it: `memset((unsigned char *)next, 0x5a, 1)` writes byte 0 of the NEXT item. Under patch 0006 that byte is the item's `sublet_parent` capability, the first field of `struct _stritem`. The patch's own text says "Overwriting the slot clears its tag and the next free faults." pb12 (virtual-malloc, no slot) runs the fixed twin clean. The arm is right and the case is wrong for it: a canary that reads the byte instead of writing it would not clobber the slot. That fix belongs to the corpus owner. The buggy run's catch is correct: cause 28 at the chunk's end. MEASURED 2026-10-22: the case rebuilt with the canary read instead of written reads CAUGHT (cause 28 at `write_probe`, FIXED run clean), while the prebuilt re-reads this row (capstone/tests/corpus-cell-probes/README.md). |
| pb13q15 | 08 e8364b5 | NO-READING:control-failed | The BUGGY run's own precondition `CHECK(ptr > rcurr + rbytes, 785)` fails. On the q15 arm (`Q15_PAD padded=2 pad_bytes_max=16`) the space scan stops at the first pad byte past rbuf, which is not a space. So the overrun this case needs never leaves the padded object, and the reduction refuses to give a verdict. pb12 CATCHES the same case. An arm property, not a defect. A variant that fills the allocation's usable extent with spaces is the corpus owner's call. |
| sq0 | 06 0c8c9a64b3 | NO-READING:inconclusive | Reached, and the defect ran undetected. `expertrem destroyed; calling rem() over freed ctx` is followed by `Assertion failed: iSlot<=p->nSlot` in idxRemFunc and abort 134. So the stale read went through. On this memsys5 arm (27 MISSED; its uaf-mem5 and bounds-mem5 controls complete rather than fault) that is a miss, and only the judge's "completed" rule leaves it unread. |

### Unattributed: a fault was read, but not at a declared site

| arm | case | fault | why it is unattributed |
|---|---|---|---|
| lt0 | 32 gh-149449 | cause 12 at a pc in no symbol | An instruction fetch from a non-code address. The ASan site is PyCodec_NameReplaceErrors. On lt1 the same case reads cause 25 and is unattributed too. |
| lt1 | 16 gh-156075 | cause 28 in advance_with_locations | The ASan sites are read_byte/read_varint, which are probably inlined into it. |
| pg-pools | 07 1af08af694 | cause 28 in pg_popcount_optimized | The case declares no fault sites. |
| pg-pools | 09 7a7d9693c7 | cause 28 in memcpy | The same: no declared sites. |
| sq0 | 20 d21bd37c7c | cause 24 in nodeRelease | The ASan sites are nodeAcquire/nodeHashLookup. On silicon this load reports the address exception rather than 24 (Q-31's order), so its cause will differ from QEMU's. |
| wm1q15 | 22 c702b44a01 | cause 25 in capstone_cap_revoke_child | The double free is refused inside the allocator's own revoke, not at a probe. |

### Arm-level

- **mruby Sublet's control smoke** fails at the flat limit (the reference row in (a)); its 23 cells read.
- **Juliet:** `judge.py`'s CAUGHT needs QEMU's own access-violation line, which silicon never prints.
  - So on a board, every catch reads NO-READING:unattributed under its STRICT rule: the 716 spatial-sample rows, and
    every temporal catch. The fault cells themselves equal QEMU's.
  - The uaf control's occasional cause 13 voids a batch on either tree, 6 of 320 runs each (dev aea753c6b15e).

## (c) Not on the batch bitstream at all

| corpus | cases | why there is no batch arm |
|---|---:|---|
| ffmpeg/plane-repros | 1 | (2026-10-22: landed on dev at fd8c28583642, with the batch-kit QEMU references for both vp2 arms (lane `compiler/plane-virtual` 32f334d52471), libavutil rebuilt on the batch recipe; both MISSED with every control as pre-registered at 878fcf649efe.) No virtual arm on dev: `bug-corpora/tools/run-virtual-cases.py:66` lists four FFmpeg corpora and not this one. A virtual run exists on the unlanded branch `plane-virtual-carve` (b03dc5ebd1eb pre-registration, 0512bd175160 result): both arms MISSED as predicted, and the carve's bound and revocation controls fired in the same boot. The read lands inside upstream's own plane extent. (Corrected 2026-10-22: the first version said FFmpeg's corpora ran on the classic line, yet five of them have vp2 virtual results.) |
| httpd/apr-pool-repros | 1 | PAUSED in the 2026-10-16 board campaign plan: it needs a new physical-profile kind, and APR's lend API was never moved to CDERIVE/CREVOKE. |
| httpd/bucket-repros | 8 | PAUSED, with apr-pool. |
| sqlite/capi-repros | 19 | (2026-10-22: a virtual-malloc arm on dev 75e077151800, path-independent images at 76054751cd4c; reference run 3 reads CAUGHT 12 and 7 out-of-denominator by the lead's decisions below; vp2-malloc queued on the board.) Before that it needed a port; its only required arm was host ASan. |
| xlang/repro | 15 | (2026-10-22: a virtual-malloc arm on dev da4cb338b37f / 76054751cd4c: CAUGHT 3, MISSED 1 (the stack use-after-return), the 11 mruby rows out of the denominator by the lead's decision; vp2-malloc queued.) Before that it needed a port: shim domains on the old domain interface. |
| xlang/lua-cdp | 13 | (2026-10-22: the same arm, CAUGHT 13.) Before that, the same. |
| xlang/reuse-not-free | 2 | No Capstone column is defined. |
| mruby/gc-slot-repros, xlang/toctou-double-fetch | 0 | Nothing to run. |
| Juliet spatial, beyond the 755-case sample | the rest of the census | Only the stratified sample (every functional variant of CWE 121/122/124/126/127, at three flow roles) ran on silicon. |

### Decisions for capi and xlang (the project lead, relayed by the runtime lane, 2026-10-22)

The lead's words: "capi и xlang как рекомендовано" ("capi and xlang as recommended"). Concretely:

**sqlite/capi-repros:**
- **(a) Version:** SQLite 3.53.3, through ports/sqlite's six rewrites plus
  `-DHAVE_MALLOC_H=1 -DHAVE_MALLOC_USABLE_SIZE=1`.
- **(b) Controls:** the generic malloc controls suffice. There is no per-row FIXED twin.
- **(c) Verdicts by oracle:**
  - a heap-use-after-free row that reads cause 25 is CAUGHT;
  - the bad-free row, refused by the allocator (cause 2 in `cap_malloc_invalid_free`), is CAUGHT;
  - a null-deref row (cause 24 or 12) is its own null-deref class, not counted as a temporal catch;
  - a none row that exits 0 is correct: no false alarm.

**xlang:**
- **(a) Attribution:** the malloc controls plus per-row fault-site attribution, with no test-only no-revoke runtime.
- **(b) Rows:** the 12 mruby shim rows are dropped, because the real-interpreter arms cover mruby.

The arms (sites, a verdicts-v1 runner, arms.json and corpus.json entries) are being built by the compiler lane, for
handoff to the board lane.
