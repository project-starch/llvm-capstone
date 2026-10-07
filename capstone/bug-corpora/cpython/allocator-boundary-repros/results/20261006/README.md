# Three arms over 32 cases, 2026-10-06 and 2026-10-07

| | | spatial | sublet | cheribsd-revocation |
|---|---|---:|---:|---:|
| spatial | nested | 6/10 | **9/11** | 6/12 |
| spatial | non-nested | 3/3 | **3/3** | 2/4 |
| temporal | nested | 3/9 | **10/10** | 2/9 |
| temporal | non-nested | 0/3 | **3/3** | 0/2 |
| **total** | | **12/25** | **25/27** | **10/27** |

Each cell is `detected / measured`. A row is a measurement only if the trigger
reached the defect site. A case that timed out, exhausted the application heap,
died at `import`, had its selected test skipped, or hit a syscall the guest does
not implement is **not** a measurement (SCHEMA rule 4) and is out of the
denominator rather than counted as silence. Nor is a row whose trigger has since
been replaced.

## Why no arm's denominator is 32

| arm | measured | excluded | cases |
|---|---:|---:|---|
| `spatial` | 25/32 | 7 | **CAPACITY** `08` `13`; **NOMODULE** `11` `23` `24`; **NOTIMPL** `18`; **SKIPPED** `15` |
| `sublet` | 27/32 | 5 | **CAPACITY** `08`; **NOMODULE** `11` `23`; **NOTIMPL** `18`; **SKIPPED** `15` |
| `cheribsd-revocation` | 27/32 | 5 | **NOMODULE** `11` `12` `21` `22` `23` |

An earlier version of this file read "that sublet loses twice as many as spatial
is itself a result: revocation's cost is what pushes those cases over the limit".
That is withdrawn. Two of the four it lost, `07` and `32`, were not over any
limit: they reached a child process the domain could not give a second
application heap. Of what remains, `08` exhausts the heap on BOTH
arms, and `13` exhausts it on the base arm while sublet measures it and reports
`cause=5`. So these exclusions show revocation costing more on no case at all,
and on one case they point the other way.

## What the measurements say about the oracles

The oracles were derived from the mechanism, per cell, and written before these
runs. 33 measurements contradicted them. **The spatial arm's oracles have since
been rewritten**, and because rewriting an oracle after seeing the results is
the easiest way to launder a disagreement into an agreement, the change is set
out here in full. No verdict was changed; only the `oracle_met` column was
recomputed.

### What was wrong, and it was the oracle

Both spatial oracle texts described a bounds-only baseline: "complete: the stale
access stays inside one arena the system allocator still holds" for the nested
cases, "fault: the heap bounds each allocation" for the non-nested ones. The arm
is not bounds-only. `CAPSTONE_REVOCATION_ENFORCE` defaults to 1
(`capstone-qemu/target/riscv/op_helper.c:1420`) and is independent of the sublet
heap discipline, so the base arm enforces revocation as well.

The measurements say how much this matters. Of the arm's 11 detections, **10
carry `cause=24`**, a dereference of a revoked capability, and exactly **one**
(`12_gh-143377`) carries `cause=7`, a store bounds fault. The arm detects almost
entirely by revocation. No cell in this corpus isolates spatial safety.

### The rewrite, and why it is not an improvement

| | before | after |
|---|---|---|
| nested | completion | completion **or** `cause=24`; a `cause=5` or `7` falsifies |
| non-nested, spatial class | a fault | `cause=5`/`7`, **or** `cause=24` where the access reaches freed and revoked memory |
| non-nested, temporal class | a fault | `cause=24`; a bounds fault or completion falsifies |
| `18_gh-157335` | a fault | `cause=7` at the write past the mapping's end |

`oracle_met` on this arm went from 16/30 to 25/30. **Nine rows flipped, every one
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

Five rows remain unmet, and these are the informative ones -- all non-nested,
all system-allocator blocks the arm had an event to act on:

- `07_gh-140594`, `08_gh-140607` -- silent. 07's measurement is void anyway: its
  trigger reached a child that never got a heap, and it has since been rebuilt to
  run in one process.
- `18_gh-157335` -- silent on all three arms. Its buffer is an mmap mapping, not
  an allocator block, so no allocator-level mechanism has an event.
- `19_gh-142664`, `20_gh-143308` -- temporal, non-nested, silent. A
  use-after-free on a system-allocator block is what revocation exists to catch,
  and this arm did not catch it.

## The finding that was worth following, resolved

`../pymalloc-repros` holds 11 of these defects as C model consumers, and on its
`spatial` arm all 20 of its cases complete -- 0 faults, which is that corpus's
declared oracle, not a failure. The same defects reached through the
**interpreter** fault on the same arm in 3 of 11 cases.

Those 3 all carry `cause=24`. That resolves it. A C model frees its object back
to its own pool and so never produces an allocator-level free, which is the only
thing a revoked-capability fault can come from. The interpreter runs a real
object lifetime and does reach the path where pymalloc returns an emptied arena
to the system allocator. The two corpora's `spatial` arms are therefore not
measuring the same thing, and their numbers still must not be pooled -- but the
reason is now known rather than open.

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
recovered, and the logs say why:

- `08` has its `readinto` return `2147483647`, and the defect IS that
  `BufferedReader` trusts that number. The real region is 8193 bytes; the
  overflow is the difference. The domain is asked for 2 GiB and answers
  `MemoryError`, so the overflow never happens.
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

## Not every fault is the mechanism

| arm | detections | by the capability mechanism | an ordinary trap |
|---|---:|---:|---|
| `spatial` | 12 | **11** | `32` cause 2 |
| `sublet` | 25 | **25** | none |
| `cheribsd-revocation` | 10 | **10** | none |

Case `32` is why this column exists. Its dangling `_ucnhash_CAPI` pointer is
**called**, and on the base arm control lands on non-code at `0x87d0` and the CPU
raises `cause=2`, `RISCV_EXCP_ILLEGAL_INST` -- the way any CPU traps that, with no
capability mechanism involved. The same case on sublet faults with `cause=24`,
`UNEXP_OP_TYPE`, a revoked capability used as a pointer, which IS the mechanism.
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
corrections moved `oracle_met` up and only up: spatial 16/30 -> 24/27,
cheribsd 18/30 -> 29/30, with 9 and 11 rows flipping from False to True and
none the other way. Read on its own that looks like the corpus improving. It is
not, and the number that says what actually happened is this one:

| arm | measured | `oracle_met` | cases the classification PREDICTS | of those, met |
|---|---:|---:|---:|---:|
| `spatial` | 25 | 21/25 | **6/25** | 3 |
| `sublet` | 27 | 25/27 | **27/27** | 25 |
| `cheribsd-revocation` | 27 | 26/27 | **1/27** | 0 |

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
only arm whose misses are informative: two of them, `06`, `17`,
where the oracle names a fault and the arm was silent.

The base arm's four unmet rows are `19`, `20`, `21`, `32`. `19`, `20` and
`21` are non-nested use-after-free on system-allocator blocks, which is what
revocation exists to catch and this arm did not; `21` is the one sublet does
catch, with `cause=24`. `32` is unmet because its fault is `cause=2`, an ordinary
illegal-instruction trap rather than the arm's mechanism.

The one case the cheribsd oracle does predict is `18_gh-157335`, and it is not
met. Its buffer is an mmap mapping, so a bounds fault was the prediction and the
arm was silent -- the same outcome it has on the other two arms.

## Provenance

`inputs.json` carries every run directory and its `run.meta`: the arm, the image
sha256 for the Capstone arms, `CAPSTONE_REV_NODES`, and for CheriBSD the whole
`security.cheri` subtree read inside the guest before any case ran and again
afterwards. Both Capstone arms refuse to boot an image whose hash is not the one
recorded for the arm they were asked for; the CheriBSD arm refuses to start if
`runtime_revocation_default` is not 1.

Two controls ran before every batch: `objects.py` (the interpreter is qualified
on this image at all) and, on CheriBSD, `mech-control S_OOB` (the si_code
handler reports — without it a fault is only "it crashed").
