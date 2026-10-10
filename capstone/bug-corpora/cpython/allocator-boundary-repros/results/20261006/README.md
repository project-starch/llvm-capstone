# Three arms over 29 cases per arm, 2026-10-06 to 2026-10-09

| | | corpus | spatial | sublet | cheribsd-revocation |
|---|---|---:|---:|---:|---:|
| spatial | nested | 13 | 7/11 | **9/11** | 5/11 |
| spatial | non-nested | 5 | 4/4 | **4/4** | 3/4 |
| temporal | nested | 11 | 3/11 | **11/11** | 2/11 |
| temporal | non-nested | 3 | 0/3 | **3/3** | 1/3 |
| **total** | | **32** | **14/29** | **27/29** | **11/29** |

Each cell is `detected / measured`. A row is a measurement only if the trigger
reached the defect site. A case that timed out, exhausted the application heap,
died at `import`, had its selected test skipped, or hit a syscall the guest does
not implement is **not** a measurement (SCHEMA rule 4) and is out of the
denominator rather than counted as silence. Nor is a row whose trigger has since
been replaced.

`../../probe/counts.py` derives this table from `matrix.tsv`, and
`counts.py --check` fails if this copy differs.

## Corrections made at review, 2026-10-10

Three statements in this corpus were wrong after the author's own review pass.
They are corrected where they stand, each marked *(Corrected at review)*:

1. **The base arm does not revoke heap memory.** This file said that
   `CAPSTONE_REVOCATION_ENFORCE` makes it revoke and that its `cause=24` faults
   are revoked capabilities.
   - The level0 heap's `free` only marks the block free
     (`capstone/ports/musl-capstone/runtime/level0.c`, `l0_free`).
   - Patch 0009 contains no revoke.
   - The runtime's only revoke sites (`musl-capstone/runtime/context.c`) release
     a thread's context area, and they are reached only through `__clone`, which
     no trigger uses.

   At the pinned QEMU, `1a6dd207`, `CAPSTONE_REVOCATION_ENFORCE` is defined at
   `op_helper.c:1635`. It acts only on a capability whose revocation node a
   `csrevoke` has invalidated:
   - a still-tagged one raises 25 (`:1712-1718`);
   - one reloaded from memory comes back untagged (`:2006-2010`) and raises 24.

   The line numbers cited before (`:1420`, `:1641-1642`) are not those lines in
   that tree. Both paths need a revoke first, and nothing on this arm issues one.
   So a `cause=24` here is an access through a register that held no tag: a
   pointer slot that later data overwrote (**tag integrity**, which the hardware
   gives every arm; SCHEMA.md `sysalloc-bounds`, "no revocation"), or an integer
   or NULL used as a pointer. The matrix records no faulting value, so these are
   not told apart; each case log's `value =` line (`op_helper.c:1706-1708`) would
   do it. The 31 `spatial` oracles that assume the arm revokes keep their text
   and carry an `oracle_premise` note.
2. **No per-case oracle was written before the runs.** This file said cases
   `01`-`21` carry theirs from the commit that added them. That commit has every
   arm as `{"status": "not written"}`, and so does the next one. The per-case
   oracles for all 32 cases first appear in the commit that also adds
   `matrix.tsv` (2026-10-06 15:41 UTC, after runs from 10:51 UTC). The only
   predictions that predate the runs are the three cell-level expectations in
   the top-level README.
3. **The CheriBSD hashes were established after the runs, not by them.** The
   runs' own `run.meta` named `fba7e8d8…`, which is now known to be a different
   build. `3172023c…` is recorded as a 16-digit prefix, and `df124719…`
   predates the runner's guest-copy check. Both are reconstructions. Record the
   full digest of the guest binary, and re-run if it is no longer available,
   before citing the cheribsd column by hash.

## The delivered 29

**All three arms are reported over the same 29 cases**, and the matrix's
`delivered` column carries that per row. Every column is therefore a per-cell
pair with every other: in the nested/spatial cell, `7/11`, `9/11` and `5/11` are
the same eleven cases.

Three rows are measured and held out rather than deleted, because the arm that
measured them is being compared against arms that cannot run them:

| case | arm | measured | why it is held out |
|---|---|---|---|
| `13` | `sublet` | DETECTED `cause=5` | base cannot run it at all |
| `13` | `cheribsd-revocation` | DETECTED `si_code=2` | the same |
| `18` | `cheribsd-revocation` | SILENT | neither Capstone arm can run it |

Each stays in `matrix.tsv` as `delivered = held-out`, distinguishable from a row
that was never a measurement. Counted, Sublet would be 28 of 30 and CheriBSD 12
of 31, over sets the other arms do not share.

### What is outside the 29

| arm | delivered | out of it | cases |
|---|---:|---:|---|
| `spatial` | 29 | 3 | **CAPACITY** `13`; **NOMODULE** `11`; **NOTIMPL** `18` |
| `sublet` | 29 | 3 | **HELD-OUT DETECTED** `13`; **NOMODULE** `11`; **NOTIMPL** `18` |
| `cheribsd-revocation` | 29 | 3 | **HELD-OUT DETECTED** `13`; **HELD-OUT SILENT** `18`; **NOMODULE** `11` |

An earlier version of this file read "that sublet loses twice as many as spatial
is itself a result: revocation's cost is what pushes those cases over the limit".
That is withdrawn. Two of the four it lost, `07` and `32`, were not over any
limit: they reached a child process the domain could not give a second
application heap. Of what remains, `08` turned out not to be capacity-bound at
all and now runs on both arms, and `13` exhausts the heap on the base arm while
sublet measures it and reports `cause=5`. So these exclusions show revocation
costing more on no case at all, and on the one case left they point the other
way.

## What the measurements say about the oracles

The oracles were derived from the mechanism, per cell. **They were not written
before the runs, and the earlier claim that they were is withdrawn.**
*(Corrected at review.)* `git log` is the record. The commit that added cases
`01`-`21`, and the one after it, declare every arm `{"status": "not written"}`.
The per-case oracles for all 32 cases first appear in the same commit as
`matrix.tsv`. Only the top-level README's three cell-level expectations were
recorded before the result. That is why the figure this file leans on is not `oracle_met`
but how many cases the classification *predicts* the outcome for -- a property
of the mechanism and the case, not of when someone wrote a sentence. 33 measurements contradicted them. **The spatial arm's oracles have since
been rewritten**, and because rewriting an oracle after seeing the results is
the easiest way to launder a disagreement into an agreement, the change is set
out here in full. No verdict was changed; only the `oracle_met` column was
recomputed.

### What was wrong, and it was the oracle

Both spatial oracle texts described a bounds-only baseline: "complete: the stale
access stays inside one arena the system allocator still holds" for the nested
cases, "fault: the heap bounds each allocation" for the non-nested ones. The arm
is not bounds-only, because tag integrity is in the hardware and cannot be
switched off. *(Corrected at review.)* This used to say the arm "enforces
revocation as well". It does not: nothing on it revokes heap memory (correction
1 above).

The measurements say how much this matters. Of the arm's 14 detections, **11
carry `cause=24`**, an access through an untagged register. That is an
overwritten pointer slot, or an integer or NULL used as a pointer, but not a
revoked capability. Three are bounds faults: `cause=5` on `07` and `08`, and
`cause=7` on `12`. The arm detects mostly by tag integrity, not by bounds. No
cell in this corpus isolates spatial safety.

### The rewrite, and why it is not an improvement

| | before | after |
|---|---|---|
| nested | completion | completion **or** `cause=24`; a `cause=5` or `7` falsifies |
| non-nested, spatial class | a fault | `cause=5`/`7`, **or** `cause=24` where the access reaches freed and revoked memory |
| non-nested, temporal class | a fault | `cause=24`; a bounds fault or completion falsifies |
| `18_gh-157335` | a fault | `cause=7` at the write past the mapping's end |

`oracle_met` on this arm went from 16/30 to 25/29 as the denominator grew. **Nine rows flipped, every one
of them from False to True, and none the other way.** That direction is exactly
what a results-fitted rewrite produces, so read the number for what it is: the
new nested oracle is *weaker* than the old one. It admits two outcomes where the
old admitted one. A weaker oracle is met more often and says less.

The honest statement is not "25 of 30 oracles now hold". It is that **the base
arm has no defect-determined outcome for the 24 nested cases at all**: whether it
faults depends on whether pymalloc has returned the block's emptied arena to the
system allocator (`insert_to_freepool`, when `nf == ao->ntotalpools && ao->nextarena
!= NULL`), which is a property of arena occupancy and not of the defect. The arm
is uninformative for them, and the corrected oracle now says so instead of
predicting a completion it cannot guarantee.

### What the arm genuinely misses

Four rows remain unmet on the base arm, and they are the informative ones:

- `19_gh-142664`, `20_gh-143308`, `21_gh-146169` -- temporal, non-nested,
  silent. *(Corrected at review.)* That silence is what this arm predicts. Its
  heap does not revoke, so a use-after-free on a system-allocator block is
  invisible to it unless a later owner overwrites a pointer slot. Their oracles
  demanded a revocation fault, so `oracle_met = False` records a wrong oracle,
  not a miss. Sublet catches all three, with `cause=24`.
- `32_gh-149449` -- unmet not for silence but for the kind of fault: `cause=2`,
  an ordinary illegal-instruction trap rather than the arm's mechanism. Sublet
  faults on the same case with `cause=24`.

An earlier version of this list also carried `07`, `08` and `18`. `07` and `08`
are now detections -- `07` once its trigger stopped depending on a child
process, `08` once its lie stopped demanding 2 GiB -- and `18` is outside the
denominator rather than a miss, because the guest kernel answers `mmap` with
`Errno 38`.

## The finding that was worth following, resolved

`../pymalloc-repros` holds 11 of these defects as C model consumers, and on its
`spatial` arm all 20 of its cases complete -- 0 faults, which is that corpus's
declared oracle, not a failure. The same defects reached through the
**interpreter** fault on the same arm in 3 of 11 cases.

Those 3 all carry `cause=24`. *(Corrected at review.)* This used to be
"resolved" as revocation, which this arm does not do. What `cause=24` does
establish is an access through a register that held no tag. The likely
difference between the two corpora is reuse. The interpreter runs a real object
lifetime, so a freed block is reused by a new owner that writes data over the
old object's pointer fields. A C model that frees back to its own pool, with
nothing else allocating, produces no such overwrite. That reuse has not been
traced case by case. The two corpora's `spatial` arms still do not measure the
same thing, and their numbers must not be pooled.

## What the re-runs settled, and what is left

Thirteen rows were re-measured on 2026-10-07: `07` and `32` on all three arms
with their new single-process triggers, `02` and `13` on cheribsd with the
vendored stdlib test package, `08` and `13` against the raised image, and `21`
with 1800 s where it had had 600 s. Five of the ten module losses remain, plus
`08` and `13` on the Capstone arms for a reason that turned out not to be
capacity at all.

**`21` is measured now, and the earlier reading of it was wrong.** It had been
recorded as TIMEOUT on both Capstone arms, and the judgement here was that
raising the budget would not help because the cancel path had timed out too and
the guest had stopped answering. It helped: `21` is SILENT on the base arm and
`DETECTED cause=24` on sublet, which is the sharpest single row in the table --
a non-nested use-after-free in expat that revocation catches and the baseline
does not. One caveat is recorded rather than smoothed over: it completed in
roughly 430 s, under the 600 s it had previously exceeded, but in a three-case
subset where the timeout had happened in a full 32-case pass. Subset and full
runs are therefore not freely comparable, and that is a property of the harness,
not of the defect.

**`08` and `13` were never capacity-bound, and the raised images were built on a
wrong premise.** Two images were built at 192 MiB of application heap against
48 MiB, on the belief that these two cases were short of room. Neither was
recovered by them, and the logs say why:

- `08` has its `readinto` return `2147483647`, and the defect IS that
  `BufferedReader` trusts that number. The real region is 8193 bytes; the
  overflow is the difference. The domain is asked for 2 GiB and answers
  `MemoryError`, so the overflow never happens. **This one was then fixed in the
  trigger instead**, and `08` is now a detection on both arms -- see below.
- `13` passes `object()` as a keyword name and the pin uses a pointer as a
  length: the host ASan build reports `failed to allocate 0x9fad41ee2fbf bytes`,
  about 175 TB. No heap in any configuration satisfies that.

So both are a declared boundary on the Capstone arms, for a property of the
defect rather than a tunable. What the raised images did establish is that
capacity was not the explanation -- a negative result, and the reason the
hypothesis is not still open.

`13` also gives the cleanest three-way contrast in the corpus. The base arm
cannot run it: the allocator refuses the bogus length and the defect's read
never happens. Sublet reports `cause=5`, catching the access before the
allocation path is reached. CheriBSD reports `si_code=2`. The nested-allocator
discipline turns a case the baseline cannot even measure into a bounds fault.

**Still a declared boundary, five rows, needing a different image.** `11`
(`_ctypes`), `15` (`_testinternalcapi`), `23` (`_testlimitedcapi`) and `24`
(`zoneinfo` tzdata) on Capstone; `11`, `12` (`_interpreters`), `21` and `22`
(`pyexpat`) and `23` on cheribsd. Adding a module changes the binary, so every
row would have to be re-measured against a new hash rather than only the
recovered ones, and that is deliberately out of scope here. `_ctypes` needs
libffi and may not be reachable in a freestanding musl domain at all.

**Probably permanent.** `18` on both Capstone arms: the guest kernel answers
`mmap` with `Errno 38`, and the case's buffer IS an mmap mapping.

Two consistency checks worth recording. `07` and `32` kept their signatures
across the trigger rewrite: `07` still faults with `si_code=1` on cheribsd and
`32` still with `si_code=5`, exactly as their child processes had, so running
them in one process changed the harness and not the defect. And `13` on cheribsd
is a detection that had been hidden as NOMODULE -- the vendored package did not
merely let the case run, it uncovered a fault.

## The exclusions are mostly missing modules, and that was found late

Four cases were thought to be short of the denominator. Re-reading every
remaining row's own evidence -- the Capstone arms' per-case logs, and for
cheribsd the `last` field of its `verdicts.tsv`, since that run keeps no per-case
log -- found sixteen more rows that are not measurements at all. A case whose
trigger dies at `import` never reaches the defect site, and recording that as the
arm staying quiet is the same error as recording a capacity failure that way.

| arm | missing, and what it costs |
|---|---|
| Capstone, both arms | `_ctypes` (`11`), `_testinternalcapi` (`15`, the test is skipped), `_testlimitedcapi` (`23`), and on the base arm `zoneinfo` tzdata (`24`). `18` needs `mmap`, which the guest kernel answers with `Errno 38` |
| `cheribsd-revocation` | `test.test_ast` (`02`, `13`), `_ctypes` (`11`), `_interpreters` (`12`), `pyexpat` (`21`, `22`), `_testlimitedcapi` (`23`) -- seven of its thirty-two rows |

So the denominators are not short by four. They are short by ten, eight and nine,
and the dominant cause is the images rather than the runs: five of the seven
cheribsd losses and three of the Capstone ones are a module that was never built
in, not a property of any defect.

### A Python exception is not evidence the defect did not run

The first pass at this flagged any output containing a traceback or a skip, and
it was wrong twice over. Cases `19` and `31` select no method and run their whole
upstream file -- 157 tests on `19` -- so a skip there belongs to some other test.
And case `22` shows the deeper error: on the base arm it raises `ValueError:
unknown event ''`, while on sublet the same code path faults with `cause=24`. The
`ValueError` IS the use-after-free's uncaught consequence -- the stale read
returned an empty string and the interpreter carried on with it. Case `16` is the
same shape, `ValueError: x` on the base arm against `cause=5` on sublet. Those
rows are measurements, and the most interesting kind the corpus holds: the arm
did not catch the defect and the program kept running on corrupted data. Only an
import failure, a skipped selected test, an unimplemented syscall or `Ran 0
tests` is excluded.

## Two rows recovered by fixing triggers rather than images

`08` and `24` were both excluded, and neither needed a different image.

`08`'s replacement value was chosen by measuring **both** host builds, because
they answer different questions and only one of them is the domain:

| lie | ASan build | plain build, which is what the domain runs |
|---|---|---|
| `2147483647` | heap-buffer-overflow | SIGSEGV |
| `16777216` | heap-buffer-overflow | SIGSEGV |
| `1048576` | heap-buffer-overflow | SIGSEGV |
| `65536` | heap-buffer-overflow | HANGS |
| `8194` | heap-buffer-overflow | HANGS |

Every value overflows, and every value terminates under ASan, because ASan
aborts at the first overflowing byte. Without bounds checking the loop only ends
if the claimed copy runs off into unmapped memory, which needs about a megabyte.
`8194` was tried first on the ASan evidence alone and would have traded CAPACITY
for TIMEOUT. `1048576` overflows the same 8193-byte region, terminates either
way, and asks the domain for 1 MiB of a 48 MiB heap.

`24` died in `zoneinfo/_common.py` `load_tzdata`: the image has `_zoneinfo`
built in but no tzdata. The UTC TZif is 127 bytes and is carried in the trigger
as base64, used only when the real tzdata is missing, so the host and CheriBSD
take the path they took before. Both branches were measured rather than
inferred, with a negative control confirming UTC really does not resolve first,
and both reach the same fault: SEGV on address `0x8` in
`clear_weakref_lock_held`.

Both now detect: `cause=5` on `08` on both Capstone arms -- which is a **bounds**
fault, taking the base arm's bounds column from two to three -- and `cause=24`
on `24` on base. Neither row's `oracle_met` needed a rule change to agree: the
recomputation flipped nothing.

## Not every fault is the mechanism

| arm | detections | by the capability mechanism | an ordinary trap |
|---|---:|---:|---|
| `spatial` | 14 | **14** | none |
| `sublet` | 27 | **27** | none |
| `cheribsd-revocation` | 11 | **11** | none |

The last column is not a detection and is not in the first. Case `32` does fault
on `spatial`, with `cause=2`, and that fault is excluded by this corpus's own
rule: a fault whose cause is not one the capability hardware raises is the CPU
trapping, not the arm's mechanism. An earlier version of this file counted it
and reported base as 15 of 29; it is 14.

Case `32` is why this column exists. Its dangling `_ucnhash_CAPI` pointer is
**called**, and on the base arm control lands on non-code at `0x87d0` and the CPU
raises `cause=2`, `RISCV_EXCP_ILLEGAL_INST` -- the way any CPU traps that, with no
capability mechanism involved. The same case on sublet faults with `cause=24`,
`UNEXP_OP_TYPE`, an untagged capability used as a pointer (on sublet, a revoked
one reloads untagged), which IS the mechanism.
Counting both as a detection would credit the base arm with a protection that
did not act, so the spatial oracles now name their falsifier: a fault whose cause
is anything other than 24 falsifies them, `cause=2` included. That tightening
moved exactly one row, `32` on `spatial`, and it moved it from met to unmet --
the opposite direction from the two earlier oracle corrections, which is the
check that it was not fitted to the numbers.

The capability causes are `5` and `7` (an access outside a capability's bounds),
and `24`, `25`, `26` (`UNEXP_OP_TYPE`, `INVALID_CAP`, `UNEXP_CAP_TYPE`), from
`capstone-qemu/target/riscv/cpu_bits.h`. On cheribsd the mechanism announces
itself with a `si_code`, so every fault there is one.

## How much each arm discriminates, which is not what `oracle_met` measures

Both the spatial and the cheribsd oracles have now been corrected, and both
corrections moved `oracle_met` up and only up: spatial 16/30 -> 25/29,
cheribsd 18/30 -> 29/29, with 9 and 11 rows flipping from False to True and
none the other way. Read on its own that looks like the corpus improving. It is
not, and the number that says what actually happened is this one:

| arm | delivered | `oracle_met` | cases the classification PREDICTS | of those, met |
|---|---:|---:|---:|---:|
| `spatial` | 29 | 25/29 | **7/29** | 4 |
| `sublet` | 29 | 27/29 | **29/29** | 27 |
| `cheribsd-revocation` | 29 | 29/29 | **0/29** | 0 |

The corrected oracles stopped predicting outcomes the mechanism does not
determine, so they are met more often and say less. On the base arm, whether a
nested defect faults depends on whether pymalloc has returned the block's
emptied arena to the system allocator (`insert_to_freepool`, when `nf ==
ao->ntotalpools && ao->nextarena != NULL`). On CheriBSD it depends on that and on
sweep timing as well, because the run records
`security.cheri.runtime_revocation_every_free_default` as 0 and
`runtime_revocation_async` as 1 -- revocation there is sweep-based and
asynchronous, so a stale access reached before the next sweep still holds a
tagged capability.

Those outcomes are reproducible: the triggers are deterministic. They are simply
not derivable from the case's class and side, which is the whole claim a
four-cell table makes. **`sublet` is the only arm whose outcome follows from the
defect's classification for every case it measured**, and it is therefore the
only arm whose misses are informative: two of them, `06` and `17`, where the
oracle names a fault and the arm was silent.

CheriBSD's figure is 0 of 29 for a reason worth stating plainly: every one of
its oracles now admits more than one outcome, because that guest's revocation is
sweep-based and asynchronous. The arm is measured and its verdicts are
reproducible; what it is not is predictable from a case's class and side.

The base arm's four unmet rows are `19`, `20`, `21`, `32`. `19`, `20` and
`21` are non-nested use-after-free on system-allocator blocks. *(Corrected at
review.)* This arm's heap does not revoke, so its silence there is the expected
outcome and the oracle was wrong. Sublet catches all three, with `cause=24`. `32` is unmet because its fault is `cause=2`, an ordinary
illegal-instruction trap rather than the arm's mechanism.

The one case the cheribsd oracle did predict is `18_gh-157335`, and it was not
met. Its buffer is an mmap mapping, so the prediction was a bounds fault, and the
arm was silent. *(Corrected at review.)* That row is now held out, which is why
the table reads 0 of 29. It is not measured on the Capstone arms: both read
NOTIMPL, because the guest kernel has no `mmap`.

## A cause 24 carries no faulting value, and the one that exists was discarded

The review that corrected the base arm's premise ends on an open question: a
cause 24 on that arm is an access through an untagged base register, which is
either a pointer slot that later data overwrote -- tag integrity -- or an
integer or NULL that was never a pointer, and "the matrix records no faulting
value to tell those apart". Two things turned out to be true about that, and
they point in opposite directions.

**The fault report cannot answer it.** `RISCV_EXCP_UNEXP_OP_TYPE` is raised by
`riscv_raise_exception(env, ..., GETPC())`, which takes no address and sets no
`badaddr`, so the `address=` field on a cause-24 line is an unset field rather
than the faulting value. The run logs agree: every surviving cause-24 line
reads `address=0x0`, at 9 distinct pcs across as many defects, while every
surviving bounds line reads a distinct non-zero address.

| cause | sites | `address=0x0` | non-zero |
|---|---:|---:|---:|
| 5 | 13 | 0 | 13 |
| 7 | 1 | 0 | 1 |
| 24 | 11 | 11 | 0 |

So this is not a recording gap in the matrix. At this pin the hardware reports
no faulting value for an operand-type fault, and no amount of care with the
matrix would have produced one.

**The emulator does print the value, and the harness was throwing it away.**
`_helper_access_with_cap` prints the untagged register's word on exactly this
path, and the comment beside it says why: without it "telling a pointer that
lost its tag from an integer that was never one costs a disassembly session".
`CAPSTONE_DEBUG_PRINT` is an unguarded `fprintf` to stderr, and `capstone_vm`
already redirects QEMU's stderr to `<state>/qemu.log`. But that file is opened
`"wb"` on every `restart`, and the run directory did not keep a copy -- so the
one field that answers the question was produced on every run and retained on
none.

`run-arm.sh` now keeps the per-case slice as `<case>.qemu.log` and records
`pc`, `address` and `untagged_value` in `verdicts.tsv`. Classifying the base
arm's cause-24 faults needs a re-run under that harness; it is not derivable
from what is on disk.

**What is on disk.** `pc` was reported all along and was being dropped;
`inputs.json`'s `fault_sites` now carries every fault line that survives, 25
of them. Read each as evidence about its own run: the 2026-10-06 and most
2026-10-07 directories were under `/tmp` and were lost to a reboot, so only 6
of these sites come from the run the matrix cites for that row, and
`is_cited_run` says which. Two checks fall out of them and both hold: case 24's
pc differs between the base and the module image, as two different links
should, and is identical across the three sublet images, which differ only in
heap capacity.

## The negative controls, and the two questions they answer

Rule 5 of the SCHEMA requires every oracle to have a negative control:
a suite whose oracles cannot say FAIL proves nothing by saying PASS.
Two controls are run here and they answer different questions. Keeping
them apart matters, because the weaker one is cheap and passing it is
not evidence that any particular detection depended on the defect.

Rule 5 is written for the `case-json` corpora, where the control
corrupts a C fixture the program then refuses. A `script-trigger`
corpus has no fixture to corrupt, so the control replaces the trigger
instead. The requirement it has to meet is the same one: the oracles
must be able to say FAIL.

### The weak control: nothing faults when no defect is performed

Every case directory is staged exactly as in a measurement run -- same
image, same interpreter, same stage, same harness -- but `trigger.py`
is replaced by a stub that prints a marker and exits. An arm that
faults here faults on the harness rather than on the defect, and every
detection it reports would be worthless. All three arms pass.

| arm | run | cases | executed | faults | case timeout | status |
|---|---|---|---|---|---|---|
| cheribsd-revocation | `boundary-negctl-20261010-074922` | 32 | 32 | 0 | n/a | pass |
| base | `spatial-negctl-20261010-065110` | 32 | 32 | 0 | 300s | pass |
| sublet | `sublet-negctl-20261010-062057` | 32 | 32 | 0 | 300s | pass |

`executed` is counted from the marker, not from the exit status, and
that distinction is the whole value of the control. An earlier base
run, `spatial-negctl-20261010-061543`, reported 32 silent cases and 0
faults and would have been recorded as a pass; the marker appears in
only 3 of its 32 logs. The VM had died after the third
case and the remaining invocations returned "VM is not running" with
a nonzero status, which the classifier scored as silence. That run is
kept in the record as void. The harness now aborts the whole run on
that message, and a control that cannot show positive evidence that
every case it claims to have run actually started is void rather than
passing.

| arm | run | cases | executed | faults | case timeout | status |
|---|---|---|---|---|---|---|
| base | `spatial-negctl-20261010-061543` | 32 | 3 | 0 | 300s | void |

### The strong control: the same traffic, with the one access made valid

Passing the weak control only shows that an arm does not fault on an
empty program. It says nothing about whether a given detection depended
on the defect or merely on the allocation pattern the case happens to
perform. For that, a case needs a variant that performs the same
allocations and frees and differs only in that the access the defect
makes out-of-bounds or stale is in bounds and live. Where such a variant
exists it is committed next to the trigger as `negative_control.py`, and
`--negative-control` prefers it over the stub; `control-kind.tsv` in each
run records which kind each case received.

18 of the 32 cases have such a variant: `01`, `02`, `03`, `04`, `05`, `09`, `10`, `14`, `15`, `17`, `19`, `23`, `24`, `25`, `27`, `28`, `30`, `31`.
11 of those are the ones whose trigger runs an upstream test file
through `runpy`. For those the variant is the upstream file itself,
committed as `negative_control_test.py`, with the offending literal or
the reentrant call removed and any assertion that no longer applies
replaced by one that checks the now-valid result -- a diff of one to four
hunks against the file the trigger runs, with the `runpy`, `unittest` and
import traffic identical. Every variant is first run on the pinned host
interpreter, where it has to print its marker, exit 0 and produce no
ASan report under either allocator, before it is run in a domain.

The remaining 14 cases receive the stub: `06`, `07`, `08`, `11`, `12`, `13`, `16`, `18`, `20`, `21`, `22`, `26`, `29`, `32`.
`11`, `13`, `18` are the three no arm delivers, so 11 scored cases are qualified by
the mechanism table above rather than by a control. For those the defect
is not one invalid access in an otherwise ordinary program -- the
reentrant comparison, the deallocation ordering or the C-level cache *is*
the defect, and removing it removes the case.

The strong control has been run on the base arm only. Sublet's 27 and
cheribsd-revocation's 11 are qualified by the weak control and by the
fault mechanism each reports, not case by case. `run-cheribsd.sh` could
not run a variant at all until now -- only the stub path was implemented
there -- so that arm's per-case control is new code waiting on a run,
not a measurement that came back empty.

| arm | run | cases | executed | faults | case timeout | status |
|---|---|---|---|---|---|---|
| base | `spatial-negctl-20261010-081957` | 9 | 8 | 0 | 300s | void -- case 24 produced no marker, see below |
| base | `spatial-negctl-20261010-083844` | 9 | 8 | 1 | 300s | failed -- re-run of the above; case 24's first variant faulted, see below |
| base | `spatial-negctl-20261010-090107` | 1 | 0 | 0 | 120s | no measurement -- case 24's second variant under the 120s default |
| base | `spatial-negctl-20261010-090622` | 5 | 5 | 0 | 900s | ran, self-test failed -- case 31 ran 299 tests with no fault, see below |

What the strong control settles, and what it does not:

- Clean on the base arm, so these detections are defect-dependent: `10`, `15`, `24`, `30`.
- Measured only inside runs that were voided or failed for case 24's
  sake: `01`, `03`, `04`, `05`, `09`, `14`. Each was silent with its marker
  present in `spatial-negctl-20261010-081957` and again in
  `spatial-negctl-20261010-083844`, so the per-case evidence is there,
  but neither run carries a clean run-level verdict. A re-run under the
  fixed evidence gate is outstanding.
- Built and verified on the pinned host interpreter, never yet run in a
  domain: `02`, `17`, `19`, `23`, `25`, `27`, `28`. One of them, `19`, carries
  an allowed-failure list: `test_hex_use_after_free` is the same
  reentrancy through `memoryview.hex()`, a sibling defect unfixed at this
  pin, and the trigger's own run of that file fails it in all six
  parametrised classes. The control removes exactly the six
  `test_hash_use_after_free` failures and no others, which is evidence it
  edited the method the case is about.
- `31` ran its full 299 tests with no fault, and its control's own
  self-test then failed on two `test_sizeof_exact` assertions. Those
  assert exact `sys.getsizeof` values, which capability pointers widen,
  so they fail on this substrate for reasons unrelated to the control;
  the trigger's own run of the file fails them too. The control now
  allows those two by name, and a re-run to record the clean verdict is
  outstanding.

#### Case 24 took two variants, and the first one was wrong

The first variant for `24` kept the defect's descriptor on
`_weak_cache` and made it return one stable `Cache()` instead of a fresh
one. On the base arm it faulted -- but with cause 7, a store bounds
fault, where the real trigger produces cause 24. A control that faults by
a different mechanism than the trigger is not evidence that the arm
faults on ordinary traffic; it is evidence that the variant introduced a
second defect of its own, since it still installs a non-dict descriptor
where the interpreter expects a mapping.

The second variant removes the descriptor entirely: same `ZoneInfo`
subclass, same construction, same vendored tzdata, nothing abnormal on
the class. It is silent on the base arm. So the fault belonged to the
first variant, and `24`'s detection on base is defect-dependent and
stands. The first variant is kept next to the trigger as
`negative_control.variantA.py` -- a control that is itself defective is
worth keeping visible, because it is the failure mode a reader should
expect from this kind of control.

The same case also shows why the case timeout is part of the control's
declaration rather than a detail. Variant B first ran under the 120 s
default and returned TIMEOUT, which is not a measurement either way
(rule 4). It completes well inside 900 s. A trigger that faults exits
early; a control has to run the case to the end, so a control needs more
time than the measurement it qualifies, not the same.

Neither variant changes any number in the tables above: `24`'s detection
on base stands, so base is still 14 of 29.

## Provenance

`inputs.json` carries every run directory. *(Corrected at review.)* It has a
`run.meta` only from the later runs: the earliest 2026-10-06 run directories are
listed by name, and one `image_sha256` per arm stands for them. A `run.meta`
records the arm, the image
sha256 for the Capstone arms, `CAPSTONE_REV_NODES`, and for CheriBSD the whole
`security.cheri` subtree read inside the guest before any case ran and again
afterwards. Both Capstone arms refuse to boot an image whose hash is not the one
recorded for the arm they were asked for; the CheriBSD arm refuses to start if
`runtime_revocation_default` is not 1.

Two controls ran before every batch: `objects.py` (the interpreter is qualified
on this image at all) and, on CheriBSD, `mech-control S_OOB` (the si_code
handler reports — without it a fault is only "it crashed").
