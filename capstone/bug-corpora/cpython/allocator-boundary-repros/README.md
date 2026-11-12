# CPython allocator-boundary defect corpus

21 memory-safety defects live in CPython 3.13.7, in the three cells
[`pymalloc-repros`](../pymalloc-repros) does not cover. That corpus holds
consumer-side **temporal** defects whose memory comes from **pymalloc** and
states it is the whole reachable set at the pin; these are the others.

Together the two corpora are 41 cases and all four cells:

| | nested (pymalloc) | non-nested (libc malloc) | total |
|---|---:|---:|---:|
| **temporal** | 20 — `pymalloc-repros` | **3** — here | 23 |
| **spatial** | **13** — here | **5** — here | 18 |
| **total** | **33** | **8** | **41** |

## What nested and non-nested mean here

CPython takes a 1 MiB arena from the OS, cuts it into 16 KiB pools and hands out
blocks of 512 bytes or less (`SMALL_REQUEST_THRESHOLD`,
`Include/internal/pycore_obmalloc.h:157`). A use-after-free or an overflow
between two of those blocks never crosses the boundary of that one `mmap`, and
`free()` never reaches `malloc`, so a malloc-level tool has no event to see.
That is **nested**. **Non-nested** means the object came straight from libc
`malloc`, with no second allocator in between.

Two routing facts decide which applies, both read off the pinned source:

* `Objects/obmalloc.c:351` sets `PYMEM_ALLOC = PYMALLOC_ALLOC`, so the MEM
  domain is **not** a separate allocator. Only `PyMem_RawMalloc` bypasses
  pymalloc.
* Requests over 512 bytes go to libc `malloc` even through the OBJ and MEM
  domains.

## The side is measured, not read off the size

The obvious shortcut -- "ASan reported an N-byte region, N ≤ 512, so it is
nested" -- is wrong, and case 07 is why. `gh-140594` overflows a **100-byte**
region and is non-nested: `PyOS_StdioReadline`'s buffer is `PyMem_RawMalloc`'d
(`Parser/myreadline.c:207,223,325,347`), so libc owns it at any size. Under
`PYTHONMALLOC=malloc` every allocation is a libc malloc, so ASan reports a
region for everything and the size cannot tell the domains apart.

So each trigger is run **twice** and the question is whether the heap report
survives the nested allocator:

| under `PYTHONMALLOC=pymalloc` | verdict |
|---|---|
| ASan still reports a **heap** violation | the object is not a pymalloc block → **non-nested** |
| no heap report | the damage stayed inside the arena → **nested** |

A crash without a heap report in the pymalloc arm is **not** a missed
measurement. The stale read returns the freelist link pymalloc writes *inside*
the freed block (`Objects/obmalloc.c:2494`), so the program dies on that
garbage while ASan stays quiet about the heap. That is the mechanism evidence
for "consumed inside the arena", and it is what `allocator_consumed` records.

Both halves of axis 2 are present on every case: `allocator_layer` says which
allocator the memory came from, `allocator_consumed` whether the damage stayed
inside one of its blocks.

## Liveness

Every case is live at the pin and the proof is a measurement, never an
assertion: the trigger is upstream's own reproducer from the issue, and running
it on the pinned 3.13.7 ASan build produces the sanitizer report quoted in
`live_proof`. All 36 reproduce from their case directory with the oracle
recorded for them -- the 21 filling the three cells `pymalloc-repros`
does not cover, and the 15 temporal/nested cases that re-measure its
issues through the interpreter (22-32, and 33-36 added 2026-10-29, whose
record and the five of its issues that cannot run whole are in
`results/2026-10-29-asan/README.md`).

Three things that had to be got right for those runs to mean anything:

* `ASAN_OPTIONS=detect_leaks=0`. With leak checking on, the leak summary buries
  the report and a reproducing case reads as silent.
* A positive control before each batch. An earlier run of this corpus capped
  address space with `ulimit -v`, which stops ASan reserving its ~15 TB shadow
  region; 84 cases came back "no violation" when the instrument had never
  started.
* A negative control, because a positive one only proves the instrument
  started. Two exist and they answer different questions: a weak one that
  replaces every trigger with a stub, which the three arms of 2026-10-06 (the
  physical `spatial` and `sublet`, and `cheribsd-revocation`) pass, and a strong
  per-case one that keeps the allocation traffic and makes only the offending
  access valid, which exists for 20 of the 32 cases, 18 of them when the three
  arms of 2026-10-06 were run -- this change adds the controls for 06 and 32, which have
  therefore not been run on those arms. It qualifies every detection the two physical Capstone arms report
  on those cases -- base 11 of 11, Sublet 17 of 17 -- and 2 of the 6 on CheriBSD, where
  the other 4 were measured on an earlier binary than the control could be run against.
  For the remaining 12 cases the defect is not one invalid access in an otherwise
  ordinary program, and three of those 14 are the cases no arm delivers. What each
  control settles, and what neither does, is in `results/20261006/README.md`. On the
  virtual arms `runners/virtual/run-virtual.py` runs a case's strong control when the
  case faults outside its `fault_sites`, and the fault counts only if the control then
  runs clean on the same image (case 19's control is paired with a probe; see
  `runners/virtual/README.md`).

## Arms

CPython has one heap on Capstone, musl's mallocng on the virtual profile
(`capstone/runtime/virtual/heap.c`): exact bounds per object, lifetime retired on free. The two
Capstone arms differ only in pymalloc, and the arm names are the ones every corpus on the board
uses:

| arm | configuration (`tools/arms.json`) | what it is |
|---|---|---|
| `virtual-malloc` | `virtual-cpython` | the interpreter with pymalloc stock (patch 0009). mallocng bounds and retires what it hands out -- `PyMem_Raw` memory, objects over 512 bytes, pymalloc's arenas -- but not the blocks pymalloc carves inside an arena |
| `virtual-nested-pools` | `virtual-cpython-pools` | the same with `CPY_SUBLET=1`: patch 0014 makes obmalloc hand out every block as a child lifetime of its arena, bounded to the request (`CDERIVE`), and revoke it on free (`CREVOKE`). Requests over 512 bytes stay with mallocng, as in the stock arm |
| `cheribsd-revocation` | -- | CheriBSD purecap, libc revocation as the platform ships it |

How to build and run the two Capstone arms is in
[`runners/virtual/README.md`](runners/virtual/README.md). A fault counts for a case only when it
lies in one of the case's `fault_sites`, which come from a host ASan run recorded before any arm
ran (`probe/asan-sites.py`, `results/2026-10-10-asan/sites.tsv`). That run locates 31 of the 32
cases; case 17 produced no report there, although the 2026-10-07 ASan build recorded one, so a
Capstone fault in case 17 cannot be attributed. Cases 33-35 take theirs from the same tool's run of
2026-10-29 (`results/2026-10-29-asan/sites.tsv`), which locates all three; case 28 rode along in
that run as a control and gave exactly its 2026-10-10 sites.

**Superseded: the physical arms.** Until 2026-10-10 the Capstone arms ran on the physical
application domain, `spatial` (the level0 heap with per-object bounds, pymalloc stock) and `sublet`
(level0 plus patch 0014). CPython no longer builds against those heaps. Their results, the
expectation pre-registered for them and how it held up are in
[`results/20261006/README.md`](results/20261006/README.md); `probe/counts.py --check` still
re-derives that page's tables from its matrix.

## The declared bundles: 2026-11-12, hash seed 0

`corpus.json` names `results/2026-11-12-seed0/virtual-malloc` and
`results/2026-11-12-seed0/virtual-nested-pools`. They ran the images and the platform of
`results/2026-11-11-t1800` (patch 0014 with the one-operand `CREVOKE`, `--timeout 1800`) with one
change to the invocation, the hash seed. `results/2026-11-12-seed0/README.md` compares every row
with the two earlier records.

    virtual-malloc        CAUGHT 17  MISSED 14  NO-READING 1
    virtual-nested-pools  CAUGHT 28  MISSED  3  NO-READING 1
    cheribsd-revocation   unchanged, the record of 2026-10-20

**The hash seed is pinned.** The declared bundles are measured with
`run-virtual.py --guest-env PYTHONHASHSEED=0`, which sets the variable for every guest process of
the run. Nothing else about the run changes. Seed 0 was fixed before the first seeded run, and no
other seed was run. Without the seed, rows 02, 03 and 04 read a property of one process's hash key
and not of the image. Their triggers hand a `bytes` field name to a `%U` format, and
`_copy_characters` reads the `bytes` object through the `str` layout. On this ABI
`PyBytesObject.ob_shash` and `PyASCIIObject.state` share offset 48, so the word that tells the copy
how wide a character is and where the characters lie is the cached hash of the field name, which
CPython keys per process. One process copies one-byte units from inside the object, the next
two-byte units past its end, the next takes its data pointer from the payload. Unseeded, rows 02
and 04 took both verdicts from run to run, on the converted and on the unconverted tree: the spread
table is in `results/2026-11-11-t1800/README.md` (ten runs of `virtual-nested-pools`, 02 CAUGHT in
six and 04 in four). Under seed 0 the three rows read the same in five of five runs per arm
(`results/2026-11-12-seed0/repeats/`), although the address space layout still differs from process
to process.

**Rows 02, 03, 04, 13 and 18 as they now read:**

| case | `virtual-malloc` | `virtual-nested-pools` | why it differs from 2026-10-20 |
|---|---|---|---|
| 02 | MISSED: the copy ends inside the block's 128-byte size class | CAUGHT, `cause=28` in `memcpy` called from `_copy_characters`, on the first byte past the 121-byte block; attributed by the negative control | the seed selects a one-byte copy of 64 bytes from offset 64 |
| 03 | MISSED, as 02 | CAUGHT, as 02 | the same field name and so the same copy as 02 |
| 04 | MISSED | MISSED `◇` | the seed selects a copy that reads no character: the run performs no out-of-bounds access |
| 13 | MISSED: `MemoryError`, declared the defect's own uncaught read | CAUGHT, `cause=28` in `unicode_fromformat_write_str` | `memory_error_is_the_defect`, declared 2026-10-22, now measured |
| 18 | CAUGHT, `cause=28` in `mmap_ass_subscript` | CAUGHT, `cause=28` in `mmap_ass_subscript` | the runtime serves `mremap` since 2026-10-22, so `mmap.resize` succeeds and the write meets the mapping's bounds |

**`04` joins `06` and `17` as a `◇` row: counted, and the measured run performs no out-of-bounds
access.** Under seed 0 the hash of `b'\xff' * 524288` says "not compact", the data pointer comes
from 16 payload bytes of `0xff`, and the copy loop ends before its first read. Both arms print a
field name of 524288 NUL characters. The row is counted as a miss on both arms, the conservative
reading, and it says nothing about the bounds of the object. The unseeded `CAUGHT` readings of `04`
came from processes whose hash selected a wider copy.

**One more cell differs, and the seed does not explain it: `09` on `virtual-malloc`.** The stock
arm does not bound a frame, the defective stores succeed, and the row reads what the interpreter
does afterwards. The declared bundle's run faulted in a dictionary lookup past the end of an arena
(`cause=28` in `unicodekeys_lookup_unicode`, counted by the negative control). Six further seeded
runs of the same image aborted as the earlier records did (`NO-READING:inconclusive`), so the
`CAUGHT` is 1 of 7. `results/2026-11-12-seed0/README.md` has the runs and what they exclude.

**The arms read 17 / 31, 28 / 31 and 12 / 31**, over the denominator the next section defines:
every case both Capstone arms reached, which leaves out `11` alone. With the review's withdrawal
rule of the next section the `virtual-malloc` figure is 11 / 31: the same six cells carry a code
that is not their axis's mechanism (`01`, `05`, `10`, `15` and `31` with `cause=24`, `32` with
`cause=12`). It is 10 / 31 when `09` is also read as the consequence it is. `virtual-nested-pools`
stays at 28 / 31, every catch with its axis's code. These two figures are this review's reading and
not a measurement, as before.

The case-by-case review that follows was written against the 2026-10-20 pair. For the rows equal in
verdict, cause and function, 28 of 32 on `virtual-nested-pools` and 29 of 32 on `virtual-malloc`,
it holds for the declared bundles too.

## Measured 2026-10-20, case by case (the declared bundles until 2026-11-12)

Bundle `results/2026-10-20-review-t1800`, on the emulator this repo pins
(`capstone/capstone-qemu d35af47b163c`) with a compiler carrying the C-80 fix,
against images from `ports/common/application/build-virtual.sh cpython` built
`CPY_TEST_CAPI=1` -- an image without it fails every case
`NO-READING:control-failed`, because both controls need `_testinternalcapi`.
`--timeout 1800`, and the budget is in the bundle's directory name because
`runners/virtual/run-virtual.py` does not record it in `inputs.json`: two bundles
measured at different budgets are otherwise indistinguishable from their own records.
The 2026-10-19 attempt at `--timeout 900` lost case `21` to a runner timeout, and a
timeout is a failed measurement and not a verdict. At 1800 s case `21` completes and
reads `CAUGHT cause=25` in `little2_toUtf8`, so it was never a hang: it simply needed
longer than 900 s. Both arms were re-run together at the one budget, because an arm
measured at 1800 s beside an arm measured at 900 s would put two runner configurations
into one declared bundle set.

    virtual-malloc        CAUGHT 15  MISSED 14  NO-READING 3
    virtual-nested-pools  CAUGHT 27  MISSED  4  NO-READING 1
    cheribsd-revocation   32 of 32 rows, and a negative-control pass in which
                          nothing faulted

**Fourteen of those cells are withdrawn by review**, as the per-arm table below adds up to
(6 + 0 + 8). A cell counts when the
mechanism REJECTED the defective access, not when the access succeeded and the
program faulted afterwards on what it fetched. Which code that is depends on
the axis, and the guest's own `sicode` selftest establishes the CheriBSD pair
on this build (`bounds si_code=1`, `revoked si_code=2`):

| axis | Capstone, the mechanism | CheriBSD, the mechanism |
|---|---|---|
| spatial | `cause=28` `CAP_OOB` | `si_code=1` `PROT_CHERI_BOUNDS` |
| temporal | `cause=25` `INVALID_CAP` | `si_code=2` `PROT_CHERI_TAG` (revocation clears the tag) |

`cause=24` `UNEXP_OP_TYPE` -- an untagged value used where a capability was
required -- is a consequence on either axis.

| arm | withdrawn | cases |
|---|---:|---|
| `virtual-malloc` | 6 | 01, 05, 10, 15 (spatial, `cause=24`), 31 (temporal, `cause=24`), 32 (temporal, `cause=12`, an instruction page fault -- not a capability rule at all) |
| `virtual-nested-pools` | **0** | every one of its 27 catches carries its axis's code |
| `cheribsd-revocation` | 8 | 01, 05, 09, 10, 13, 14 (spatial, `si_code=2`), 31 (temporal, `si_code=2`, decided by `si_addr` -- see below), 32 (`si_code=5` `PROT_CHERI_PERM`, neither code) |

**The arms read 15 / 31, 27 / 31 and 12 / 31.** That is what the 2026-10-20 bundles and the
CheriBSD record say, every cell of it derived by the judge or, for ③, taken from that run's
record. It was the figure to quote until 2026-11-12. The declared bundles now read 17 / 31 and
28 / 31 (previous section), and ③ is unchanged.

With the fourteen withdrawals above applied the same cells read 9 / 31, 27 / 31 and 4 / 31. **That
is this review's reading, not a measurement**: no tool enforces the withdrawal rule, so nothing in
the tree recomputes those three numbers. `catch_causes` (`SCHEMA.md`) is the key that would, for the
two Capstone arms, and no configuration declares it; for ③ it would not be enough either, because
those cells come from a run's record and prose rather than from a judge. On the **nested** rows, 20
of the 31, that reading puts ③ at 0 in both cells and ① at 0 in the spatial one.

**On the temporal axis the fault code alone cannot decide a CheriBSD cell.**
Revocation clearing a tag and dereferencing a recycled non-capability value both
report `si_code=2`; only on the spatial axis do the mechanism and the consequence
carry different codes. `si_addr` is the tiebreaker, against two anchors: case
`09`, a known consequence, faults at `0x69c7ca`, and case `21`, a known
mechanism, at `0x426656a8`, a heap address. Case `31` re-ran at `0x5e5f06` --
`09`'s shape -- so it is withdrawn, and with it the arena-return explanation its
oracle offered, since that would have put the fault at the stale access on a
heap address inside the freed block. Each case's own `note` records which fault was which and
why.

**The denominator is every case both Capstone arms REACHED** -- `reached` in the
observation, the runner's own record that the trigger got to the defect before its
defective access. Reaching it is the arm's chance at the defect; what the arm then
did or failed to do with that chance is the measurement. So a case an arm reached
and then could not read stays in and scores as no detection, while a case an arm
never reached is out, because the arm was never given the chance. On this bundle
that is **31 of 32**: `11` alone is out, not reached on either arm (the trigger
stops on `ModuleNotFoundError` before the case).

`13` is **in**, on `dev`'s rule rather than this bundle's reading. `case.json`
declares `memory_error_is_the_defect`, which records that the `MemoryError`
`virtual-malloc` hits is the defect's own uncaught read and not the table
refusing; under that rule the case is reached and the arm missed it.
`runners/virtual/run-virtual.py` applies the rule, so it takes effect on a run,
and the 2026-10-20 bundle predates it: its cell reads
`NO-READING:infra`. The arm was re-measured on 2026-11-11 and 2026-11-12, and the declared cell
reads MISSED.
`virtual-nested-pools` detects it, `cause=28` in `unicode_fromformat_write_str`.
`18` is **in**: both arms reached it and both MISSED on the 2026-10-20 bundles. Since the runtime
serves `mremap` both arms catch it (previous section). It had been held out on `NOTIMPL` from the `spatial` and `sublet` arms
retired on 2026-10-11, which is a verdict from arms this table no longer reports.
Including it is the conservative direction -- it is a case both unprotected arms and
Sublet missed, so holding it out removed evidence against Sublet.

This earlier read "live upstream and reproduces on the reference ASan build", which
is not the criterion the table uses: all 32 cases appear in
`results/2026-10-10-asan/sites.tsv`, so that wording yields 32 and is silent on the
third condition actually applied.

*Known imprecision, not fixed here.* One denominator is shared by all three arms,
while `SCHEMA.md` has a NO-READING "count as neither". Strictly each arm's figure
should be taken over the cases THAT arm has a reading for, which is why ③'s
non-nested/spatial figure below rests on four readings and not five. Moving to
per-arm denominators would change every table in all three programs, so it belongs
in a change made on its own terms rather than as a side effect of this one.

Four changes moved two withdrawals, recovered one reading, and put a
caveat on two rows:

* **`06` and `17` are marked `◇`: counted, and the measured binary contains
  no out-of-bounds access anyway.** On `06`, `SIZEOF_VOID_P` is 16 on both
  capability targets, so `sizeof(PyFloatObject)` is 48 against the host's 24 and
  `offsetof(PyASCIIObject, hash)` is 40 against 24 -- both read from the target
  binaries' DWARF -- and the defective read lands 8 bytes *inside* the object.
  On `17`, all three targets sink the defective load below the length check that
  the source puts after it, so with `inleft == 1` it never executes; the plain
  x86-64 build at the same `-O3` does perform it, so this is instruction
  scheduling, not a flag. Both rows are counted as misses, which is the
  conservative reading and needs no argument about where to draw a line. Each
  gains a probe that marks the access -- `probe-140551.py`, `probe-157325.py` --
  because upstream's reproducers mark nothing, and `06` gains a
  `negative_control.py`.
* **`02`, `03` and `04` read from the declared bundle like every other row.**
  They were re-measured under `PYTHONHASHSEED=0` with five repeats per arm
  during the review; those runs were not committed, so they were not
  what the table reported and the claims they carried were dropped from the
  notes. Since 2026-11-12 the declared bundles are themselves measured under that
  seed, and `results/2026-11-12-seed0/repeats/` commits five repeats per arm.
* **`16` on Sublet was `NO-READING` and is a detection, shown `D 28‖`.** It
  had faulted `cause=28` in `advance_with_locations`, a symbol the case did not
  declare. That symbol is `read_byte` inlined: `read_byte` is one statement,
  `return *bounds->opaque.lo_next++;` at `codeobject.c:1077`, the exact line
  host ASan names, and the target binary's DWARF shows every
  `DW_TAG_inlined_subroutine` in `advance_with_locations` resolving to that
  line, with 1077 the only line in the function that dereferences the linetable
  pointer. `advance_with_locations` was added to `fault_sites` with that written
  into `fault_sites_why`, and the case was **re-run** rather than re-scored, so
  the verdict is the judge's. Attribution by symbol name is fragile across build
  configurations -- the ASan host build and the optimised cross build inline
  differently -- and the source line is what the comparison wants.
* **`08` on CheriBSD is restored to a detection, shown `D 2†`.** Its
  withdrawal had been reasoned from size-class rounding, which cannot absorb
  `READ 1048576` out of an 8193-byte region. Re-run in the guest with only the
  claimed length varied: `malloc(8193)` gets a capability of length 10240, so
  10240 is the last length that does not fault, 10241 through 24576 fault
  `si_code=1` PROT_CHERI_BOUNDS, and 32768 upward fault `si_code=2`
  PROT_CHERI_TAG -- the flip is exactly where `tools/cheribsd/capbounds` finds
  this block's tag going. Every fault lands at one of two addresses 118 bytes
  apart, so the copy is stopped at the object's edge whatever is claimed. Past
  the representable range the arithmetic clears the tag and the same rejection
  arrives under the tag code, so **`si_code=2` is the spatial mechanism here**.
  A capability system has three regimes, not two.

**The function name cannot make this call, which is why the code has to.**
Every withdrawn fault resolves to the case's declared `fault_sites`, because
the defective read and the dereference of what it returned are commonly two
lines of one function -- `set_dealloc` reads `entry->key` at `setobject.c:505`
and dereferences it at `:507`. Disassembled at the faulting pc, the two are
different instructions: `ldc a0, 0x0(s3)`, the capability load of the entry,
against `ld a1, 0x0(a0)` at `address=0x7`, a plain load through what that
returned.

**`tools/verdicts.py` could enforce the Capstone half automatically.** Its
optional `catch_causes` key, proposed in PR #259, would score a `cause=24` fault
MISSED where a configuration declared it as `[25, 28]`. No configuration
declares it in any tree -- `tools/arms.json` has no such key on `dev` or on any
open branch -- and the key itself exists only in that pull request. Two things would have to happen in the same
commit, and neither is this review's to do alone: the key lands in
`tools/verdicts.py` **with** its entry in `SCHEMA.md` and a test, and this
corpus's bundle is re-folded, because `derive-verdicts.py --check` re-judges
every stored observation against the current judge and would otherwise fail on
every cell the key re-scores. `virtual-cpython` and `virtual-cpython-pools` are
used by this corpus alone, so no other corpus's numbers move -- but "no other
corpus moves" is not the same as "safe to declare", and an earlier revision of
this paragraph ran the two together.

### One cell that moved for a reason worth recording

**Case 02 and case 04 depend on the process's hash key.** 02 was
`D 5` on the pools arm in `results/2026-10-10-sublet-isa` and MISSED in the
first full run of 2026-10-18; 04 was MISSED in that run and `cause=24` in the
re-run, from the same image and the same trigger. An earlier revision of this
paragraph put that down to the contents of neighbouring memory. The layout shows
the cause: `_copy_characters` takes the character width and the location of the
characters from the `state` word of the `str` layout, and on this ABI that word
is the `bytes` object's cached hash (`ob_shash`, offset 48), which CPython keys
per process. The length is `ob_size` in every process. The seed rule of
2026-11-12 fixes the key (section "The declared bundles").

In the 2026-10-20 bundles 02 is MISSED on both virtual arms, 04 is MISSED on
`virtual-malloc` and `CAUGHT` `cause=28` in `_copy_characters` on
`virtual-nested-pools`. The `PYTHONHASHSEED=0` repeats run during that review
were not committed. The declared bundles of 2026-11-12 read 02 `CAUGHT` on
`virtual-nested-pools` and MISSED on `virtual-malloc`, and 04 MISSED on both.

**Case 16's fault was real and unattributable, and the reason was a symbol
name.** The pools arm faults `cause=28` in `advance_with_locations`, which was
not one of the ASan sites its `case.json` declared (`read_byte`), so the judge
returned NO-READING. Naming a site after the run WOULD fit the instrument to the
reading, which is why the site was added only on evidence that is a property of
the binary and not of any run: `read_byte` is one statement at
`codeobject.c:1077`, the line host ASan itself names, and the target binary's
DWARF puts every `DW_TAG_inlined_subroutine` inside `advance_with_locations` at
that line, which is also the only line in the function that dereferences the
linetable pointer. `advance_with_locations` IS `read_byte`, inlined. The site
was added with that written into `fault_sites_why`, and the case was **re-run**
rather than re-scored, so the verdict is the judge's.

The general point outlives the cell: **attribution by symbol name is fragile
across build configurations**, because the ASan host build that produces
`fault_sites` and the optimised cross build that produces the fault inline
differently.

## Fidelity, stated as a limitation

The triggers are upstream reproducers driven through the real interpreter, not
C reductions against the allocator the way `pymalloc-repros` does it. They
establish that the defect is live and which allocator owns the object. They do
**not** establish which arm ought to catch it.
