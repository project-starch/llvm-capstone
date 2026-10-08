# Why each arm is silent: dispositions, two retractions, and what is still open

2026-10-08. Lane `virtual-capstone-bug-corpora`, on `origin/virtual-musl-local`.
Commits `50d56a1704b1`, `758e6084d0ce`, `b474f81138d5`, `78a1dd8c386b`, plus the
parking of one case.

The question this day started from: for every bug in the tree, does each of the
three protection arms report? The table answered it for the cells that were
measured and shrugged at the rest. By the end a `caught` cell still means what it
meant, a `missed` cell has to say WHY where a run can establish it, and eight
cells that said `caught` do not any more.

## The record to read

* [`capstone/bug-corpora/PROTECTION.md`](../../bug-corpora/PROTECTION.md) --
  generated, the whole table: totals, a ledger of what each arm's silences are
  made of, the arms compared only where all three were measured and split by who
  allocated the object, then one row per bug per application.
* [`capstone/bug-corpora/protection.json`](../../bug-corpora/protection.json) --
  the same with the bundle and the detail behind every cell.
* [`capstone/bug-corpora/SCHEMA.md`](../../bug-corpora/SCHEMA.md) -- the rules:
  `disposition`, `detection_unproven`, `allocator_boundary`, and what each one
  obliges a declaration to say.
* [`capstone/bug-corpora/arms.json`](../../bug-corpora/arms.json) -- the three
  arms and the verdict vocabulary, including the quarantine rule below.
* `tools/check-corpus.py` (clean on 13 corpora, 192 declared cases; its self-test
  rejects 19 corruptions) and `tools/protection-matrix.py --check` (generated
  files fresh).

## Where it stands

| arm | caught | of those, by quarantine | missed | of those, explained | not run | of 145 | ignored |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cheribsd` | 35 | 4 | 75 | 49 | 35 | 145 | 17 |
| `capstone-sysalloc` | 56 | 0 | 62 | 0 | 27 | 145 | 17 |
| `capstone-sublet` | 122 | 0 | 0 | 0 | 23 | 145 | 17 |

Split on who allocated the object the defect crosses, over the cases where all
three arms have a verdict:

| boundary | cases | `cheribsd` | `capstone-sysalloc` | `capstone-sublet` |
|---|---:|---:|---:|---:|
| `system` -- malloc handed the object out directly | 29 | 24 (83%) | 29 (100%) | 29 (100%) |
| `nested` -- the program's own allocator carved it | 66 | 8 (12%) | 8 (12%) | 66 (100%) |

**At the nested boundary the two malloc-boundary arms are indistinguishable, and
it is the same eight cases.** What they catch there, they catch because the object
reaches libc after all. The entire measured difference between them is at the
malloc boundary, 24 against 29, and all five of the cells that differ are
`not-temporal`: CheriBSD bounds an allocation to its SIZE CLASS (7 -> 16, 20 ->
32, 100 -> 112, 1000 -> 1024, 4097 -> 5120, all measured), so a short overrun
lands inside the round-up, while ours narrows to the request.

## IN THE QUARANTINE COUNTS AS CAUGHT

The user's rule, and deliberately generous to the arm. Where a run measures the
object inside CheriBSD's revocation quarantine and no sweep completing while the
case ran, the cell is a CATCH, marked `C`-q and counted in a column of its own
because the evidence is membership in a shadow bitmap rather than a reported
fault. The arm itself is untouched -- revocation on, asynchronous, batched, as the
platform ships it. Four cells: mruby 3, 9, 10, 17.

## The instrument, and how to read it

`capstone/ports/common/host/cheribsd/quarantine-probe.c`, shared because the
question belongs to every corpus. It reads the kernel's revocation shadow bitmap
(one bit per 16-byte granule, set while quarantined) and the dequeue epoch, whose
advance counts completed sweeps, and forces no sweep. Two entry styles: a shared
object to `LD_PRELOAD`, and `-DQPROBE_WRAP` for a static binary, because mruby's
arm must be static -- dynamically linked it dies in the loader with "Traditional
TLS not supported". The shared CheriBSD runner passes a case's own `env` through,
which is how the first style gets in front of a program.

**Read the sweep counter first.** With `sweeps=0` nothing freed during the case
was ever cleared, so a stale read succeeds WITHOUT the block being handed out
again, and a zero in `reused_while_quarantined` then says nothing about
quarantine membership. The counter has a positive control: 400,811 allocations
churned give 22 sweeps, a quiet run gives 0.

What separates the two silences is the program's own account of where its memory
went, and in every case it is the SIZE of the frees column:

| group | frees in the whole process | disposition |
|---|---|---|
| postgres/mmgr | **0** | `never-freed` |
| cpython/pymalloc | 5, with 4 allocations | `never-freed` |
| memcached/allocator | 4 to 6 | `never-freed` |
| httpd bucket + apr-pool | 7 to 10 | `never-freed` |
| perl 5 | 2,983 | `never-freed`, on the code path (`PL_sv_root`) |
| mruby 3, 9, 10, 17 | 1,238 to 1,994 | `quarantined-unswept` -- mruby frees through libc |

mruby is why the expectation cannot be assumed: four cases that look like every
other nested row came out the other way, and they are the first measured instance
of the asynchronous window losing a race it could have won.

## Two retractions

**Eight `capstone-sysalloc` catches are withdrawn.** Every one had
`attributed=false` in its own run -- the runner resolves the case's labelled probe
in the image that ran and says whether the faulting pc lies inside it -- and
nobody read the column. Rule 2 of the contract is exactly that: a fault is
accepted at the labelled probe and nowhere else. A control now runs each case
beside its own upstream-FIXED sequence in the same image, and the fault fires on
both: memcached 02-07 at `slabs.c:533` and httpd/bucket 00, 01, 04, 06 at
`apr_buckets_alloc.c:113`, which is allocator CREATION, where there is no "after
free" to detect. Declared `detection_unproven`; the arm drops from 64 to 56.

That control did not exist on this vehicle: `MC_CASE` and `APRB_CASE` called their
case body with a hardcoded zero, so the fixed arm was unreachable. The fixture's
spare event word now selects it, with no header layout change, because the hosted
runners parse those structs by offset.

**"CheriBSD catches one nested case more than we do" is withdrawn.**
memcached/allocator 8 is now parked. The two cells are not a comparison: in the
CheriBSD build `mcp_object_backing` takes each cache object with its own `malloc`
(`src/cheribsd/malloc-leases.c:161`), so libc's per-allocation bound covers the
object and does the nested allocator's work for free, while the virtual build
carves the objects out of one `MCP_OBJECTS` region where `capstone-sysalloc`
leaves the nested allocator unprotected by definition. `capstone-sublet` does
catch it, cause 28 (CAP_OOB). The row returns when the virtual build takes cache
objects from the platform allocator the way the CheriBSD build does -- and should
then go the other way on a shorter overrun, since our bound is the request and
CheriBSD's is the class.

## An open port defect

The memcached fault above is not understood, only located, and that is written up
in [the port's README](../../ports/memcached/allocators/README.md): every case
that releases a slab chunk faults at `slabs.c:533`, the first read of `ptr` on
entry to `do_slabs_free`, cause 24 (untagged), on the fixed sequence as well. A
pointer spilled through a `shrink`-narrowed stack slot reloads without its tag.
The first hypothesis is already refuted there -- a silent image of the same
campaign has 187 `shrink`->`stc` pairs against this one's 317 -- so the next step
is the register state at the trap, not more reading of the disassembly. Six cases
of that corpus are unmeasurable on the virtual vehicle until it is fixed.

## Two arms that were never built, and why the name hid it

`cheribsd` x `cpython/pymalloc` read "not run" for twenty cells because
`shared/build-cases.sh` had a target called `cheribsd` that invoked
`host/cheribsd/poisoncap/build.sh`. PoisonCap is our adapter for a competitor's
platform and `arms.json` excludes it on purpose, so the arm looked measurable
while the only CheriBSD images were of something else. And it could not have
compiled: `shared/corpus.h` guarded its capability-operand probe on
`PYMALLOC_POISONCAP`, a PLATFORM, where what decides is the ABI -- on any purecap
target a pointer is a capability and an `"r"` operand fails with "couldn't
allocate input reg for constraint 'r'". Both fixed; the guard is now
`__CHERI_PURE_CAPABILITY__` as well, checked against both toolchains.

## Four instrument errors, recorded rather than quietly fixed

Two of them are the shape that reads clean without checking anything:

* memcached's supervisor used for cpython's cases, built with
  `-DPROBE_SYMBOL=mc_defect_read`, printing "expect mc_defect_read unavailable" --
  an oracle that could never have fired;
* a mode argument passed to a build that takes `argc 3`, so all twenty cases
  returned exit 2 before doing anything;
* no arguments passed to postgres/mmgr's driver, which takes the mode and checks
  the case number against its own, so five programs returned 75 -- an
  infrastructure failure is never a verdict, and the convention did its job;
* an empty-fixture control that proved nothing, because the case refuses a fixture
  with no events (`APRP failed=700`) before reaching the line under suspicion.

## What is still open

| | cells | what it needs |
|---|---:|---|
| wireshark/wmem disposition | 22 | a tshark CheriBSD build, which does not exist |
| SQLite + postgres/sql silences | 52 | a `defect_marker` in 41 cases, then one run. The largest single lever in the tree, and hand work |
| memcached + httpd withdrawn cells | 8 | the open port defect above |
| ffmpeg/pool disposition | 4 | no `build-cases.sh` and no CheriBSD runner; the path has to be built, as cpython's was |
| postgres/sql sublet | 9 | the server image is built, the run is ~41 min |
| wireshark/wmem sublet | 4 | `WM_PAYLOAD_BYTES` is 256 MiB and mallocng refuses it; measured peak need is about 2.4 MiB |
| mruby case 11 | 2 | unstable on three vehicles (abort / INST_PAGE_FAULT / timeout); diagnose, do not repeat |

One thing to build before the next campaign: a **staleness query**. The runner
already records `platform_sha256` and a per-row `image_sha256` and nothing asks
them, which is why three full campaigns ran in one night for at most 56 open
cells -- the base moved twice and everything was measured again each time.
