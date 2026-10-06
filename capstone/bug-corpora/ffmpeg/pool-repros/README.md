# FFmpeg pool defect corpus

Consumer-side temporal defects whose stale storage is memory an `AVBufferPool`
handed out. `pool_release_buffer` (`libavutil/buffer.c:344`) pushes a payload
onto a LIFO freelist and `av_buffer_pool_get` (`:390`) hands the identical
`buf->data` back, so nothing reaches `malloc` and same-address reuse is a
property of the allocator rather than of a run.

    00_461fb22053_af_join_dedup_bound/            a reference is never taken; stale read
    01_1886c3269d_h264_refs_partial_clear/        reset bounded by the count, not the array
    02_316531e61c_vidstab_parked_plane_pointer/   pointer parked in a library; stale write
    03_5c66a3ab51_vvc_nonref_output_releases_tabs/ non-ref frame output; side tables returned

| shape | cases |
|---|---|
| reference never taken / reuse / stale read | 0 |
| partial clear / reuse / stale read | 1 |
| parked pointer / reuse / stale write | 2 |
| premature return to the pool / reuse / stale read | 3 |

Four cases, four shapes. **Case 3 is the corpus's first `AVRefStructPool` case**; 0-2 are all
`AVBufferPool`. That matters because `AVRefStructPool` is the second of the two FFmpeg pool
allocators this work ports, and it is a genuine recycling pool rather than a wrapper: a release
pushes the entry onto `pool->available_entries` (`libavutil/refstruct.c:230-231`) and the next
`av_refstruct_pool_get` pops the same one back (`:258-261`).

**Case 3 has run natively and on the component port's capability pair.** Native 2026-10-03
(`results/2026-10-03-native-four-cases/`): both arms as predicted, and `run-native.sh` exited 0 over all
four cases, so extending `shared/driver.c` for the refstruct pool did not disturb 0-2. Capability pair
2026-10-04 as probe case 39
(`../../../ports/ffmpeg/buffer-pool/results/measurements/20261004-vvc-case3-probe39/`): mode 0 completes,
mode 2 faults with cause 24 at the labelled access, one binary for both. Cases 0-2 carry N=3 on both the
native pair and the `poolsublet` arm.

Case 3's `poolstock`/`poolsublet` rows (fixtures 46/47) were **measured 2026-10-04**
(`../../../ports/ffmpeg/app/results/20261004-qemu-pool-corpus-40-47/`): `poolsublet` 46 faults with
cause 24 at `case.c:104` and 47 completes FIXED, against a `poolstock` control that completes both —
16 of 16 cells across 40-47 on both arms. **All four cases now carry a discriminating pair on FFmpeg's
own ported pools.** They were predictions until then **because nobody had run them, not because they
could not be run.** The earlier wording here said the FFmpeg app port's "SDK gate
correctly refuses both toolchains on this host"; **that was withdrawn on 2026-10-04 (`5208789e4b9e`)**. A
qualified toolchain IS present — the `llvm-capstone-cc` sibling worktree's `build-release`, commit
`7d01722aab88` — `check_toolchain` accepts it end to end, and on 2026-10-04 all sixteen corpus images
(40-47 on both arms) were built through that gate with it. Read the per-case status, not the corpus
status, when counting what is measured.

Case 3's first native run **failed**, and the reason is recorded because it generalises: the pool's free
list is LIFO, so with both side tables released the entry handed back is `rpl_tab` (the last released),
not `tab_dmvr_mvf`. The reduction watched only the latter and reported `reuse_same_address=0` — a correct
allocator misread by the instrument, and indistinguishable from a defect that does not exist. The
reduction now tracks both and names which came back. The inventory and triage that selected them, and the
further pool-backed specimens it found that are not built here, are in
[`docs/ref/ffmpeg-pool-consumer-defects.md`](../../../docs/ref/ffmpeg-pool-consumer-defects.md).

## The contract

The layout and the `case.json` fields are the corpus contract in
[`SCHEMA.md`](../../SCHEMA.md),
which is the authority; it is referenced rather than copied, because a contract
that exists twice is two contracts. One directory per case,
`NN_<upstream-fix>_<slug>/` holding `case.c`, `case.json` and `PROVENANCE.md`;
case numbers dense from 0; `case.c` declares the number its directory carries
and the driver refuses a fixture that names another.

Where this corpus differs, and why:

* **An extra arm, `native-fix-differential`.** The other corpora's arms differ
  by *protection*, the defect being present in both. Here the native pair
  differs by whether the **upstream fix** is applied, which is a different axis
  and is named rather than folded into `spatial`/`sublet`. The protected arms
  are the port's probe cases 36–38 and do differ by protection only.
* **`native-detect` is declared and not written**, and not merely unwritten but
  tautological here: the port's payload arena is one allocation, so ASan's
  silence would measure the fixture rather than FFmpeg.

## Running

    bash runners/run-native.sh [outdir]

One program per case, built from its `case.c` plus `shared/driver.c`, each run
twice. The control arm runs first and an infrastructure failure exits 75 with no
verdict. The protected arms live with the port, because they need a toolchain and a
guest a per-case script would have to reinvent. A Capstone domain:

    bash ../../../ports/ffmpeg/buffer-pool/security-tests/qemu/run.sh <out> \
      --cases 36,37,38 --modes 0,2 --rounds 1

and against the Sublet port of FFmpeg's own pools, each `case.c` unchanged, in the FFmpeg app
port's domain (build first with `FFAPP_HEAP=sublet FFAPP_POOL=sublet|stock
FFAPP_CORPUS_DIR=<this corpus>` `ports/ffmpeg/app/host/build-domain.sh`):

    CAPSTONE_VM_STATE=<running VM> FFAPP_CORPUS_OUT=<new dir> \
      bash runners/run-sublet-port.sh <poolsublet|poolstock> [rounds]

(the delegated application ABI: each image is a `capstone-exec` application, and the verdict
reads its exit status, the launcher's fault record and the QEMU log over that run;
`results/20260929-qemu-sublet-port/` was taken on the earlier HostCall transport; the same
predictions re-run on this transport, 36/36 as registered, are in
`ports/common/application/results/20260929-dev-merge.json`).

and CheriBSD with PoisonCap, where the same three cases are registered as
`pool-<mode>-<case>`:

    python3 ../../../ports/ffmpeg/buffer-pool/host/cheribsd/poisoncap/run.py \
      <build> <out> --stage pool --disable-default-revocation \
      --case poison-live --case poison-read --case poison-write \
      --case poison-reuse --case poison-reused-read \
      --case pool-0-36 --case pool-2-36 --case pool-0-37 --case pool-2-37 \
      --case pool-0-38 --case pool-2-38 --sdk ... --rootfs ... --image ...

Keep the five `poison-*` controls in that selection. Without them the run shows
only that mode 2 ends differently from mode 0, which is a differential and not
evidence that poisoning was active; with them the platform is demonstrated
independently of these cases. `selection.json` records `complete_suite: false`
for any subset, so a partial run cannot later read as a full one.

## Four systems

| system | what it acts on | af_join | h264_refs | vidstab |
|---|---|:--:|:--:|:--:|
| **Capstone** | bounds and tags; no lifetime event | — | — | — |
| **Sublet** | the last return to the pool | **fault**, cause 24 | **fault**, cause 24 | **fault**, cause 24 |
| **CHERI default** *(PREDICTED, NOT MEASURED — retracted 2026-10-06)* | `free()` → quarantine → sweep | *—* | *—* | *—* |
| **PoisonCap** | the lease return: poison, then sweep before reissue | **SIGPROT** 162 | **SIGPROT** 162 | **SIGPROT** 162 |
| **Sublet port of FFmpeg's own pools** (2026-09-29) | FFmpeg's own `buffer.c`: the return to the pool is a revoke | **fault**, cause 24, at `case.c:85` | **fault**, cause 24, at `case.c:48` | **fault**, cause 24, at `case.c:33` |

> **RETRACTION, 2026-10-06 — the CHERI-default row is a PREDICTION, not a measurement.** Its three
> dashes were asserted as measured by `18a16640682b` ("Measure the stock-CheriBSD arm instead of
> deriving it") and carried forward by `3f3abf676e58`. No such run is evidenced: the cited case names
> `pool-0-36/37/38` are built at exactly one place,
> `../../../ports/ffmpeg/buffer-pool/host/cheribsd/poisoncap/run.py:70`, and that runner's only call
> site passes the unconditional literal `"--runtime-revocation", "off"` at `:129-130` — unchanged at
> every revision of the file — so those names belong to a revocation-**off** run. The precise scope
> of what is and is not provable from the tree is in each case's `cheribsd-revocation.retraction`
> field. The predicted *outcome* still stands on its mechanism and on the measured siblings
> (`wmem-repros` 13/13, httpd 9/9); it is the **measurement** that is withdrawn.
>
> **A separate, weaker gap — flagged, not retracted.** The **PoisonCap** row cites the same
> `pool-{0,2}-{36,37,38}` names, and no committed bundle holds those either: cases 36-38 entered the
> runner in `6891936e52f3`, after the only pilot that recorded pool cases
> (`results/measurements/20260919-poisoncap-pilot/pool-isolated-1.json`, whose case list stops at
> `pool-2-13`). This is weaker because `--case pool-2-36` *is* a valid selector, so the run is
> possible and may simply be uncommitted, whereas the CheriBSD arm had no code path at all. It needs
> its bundle committed or its own retraction; until then this row must not be contrasted against as
> "the measured one".
>
> **UNRESOLVED, and not silently corrected either way:**
> `../../../docs/ref/cheribsd-denominator-audit.md` counts **22** measured CheriBSD temporal rows
> across the corpora (13 + 1 + 8), while the inventory's table (c) reports **18**. The two have not
> been reconciled, and neither figure is relied on here.

The last row is the Sublet port of FFmpeg's own pools (`ports/ffmpeg/sublet`), not the
buffer-pool port's substitute that the `Sublet` row measures. Each `case.c` runs unchanged against
it, in the FFmpeg app port's domain, with upstream's pools as the one-macro control:
[`results/20260929-qemu-sublet-port/`](results/20260929-qemu-sublet-port/README.md), N = 3 per cell.

It catches whoever listens for the moment the nested allocator takes the storage
back. The other two listen for an event that never happens here: Capstone
spatial has none at all, and CHERI's deployed revoker sweeps what `free()` put
in its quarantine — a buffer returned to an `AVBufferPool` never reaches
`malloc`, so it never enters that quarantine and the sweep has nothing to find.

The CHERI-default row is measured, not derived, and carries its own control in
the same guest immediately before the cases: the `cheribsd-abi` probe calls
CheriBSD's `malloc_revoke_enabled()` and the runner requires
`runtime_revocation=1`. Its summary records `guest_default_revocation:
preserved`. **The argument was always available; what was missing was the run.**

PoisonCap is not blind here, and that is the expected result rather than a
setback: [the taxonomy](../../../docs/design/sharing-bug-taxonomy-and-novelty.md)
files free-ended UAF as **1-timing** and puts it in the *Performance* column,
noting that CHERI can match it with `eager` and that the claim is what that
costs. These three cases are that class. Two systems catching them is the
model and the measurement agreeing.

### The controls, and one intermediate

Three further arms ran and are not systems in the table above:

| arm | role |
|---|---|
| Capstone mode 0 | the matched control for Sublet: same binary, same guest, one difference |
| PoisonCap mode 0 | the matched control for PoisonCap: same binary, same guest, one difference |
| Capstone mode 1, backing lifetime | an intermediate, and the informative negative |

The two mode-0 arms are controls and not comparisons. They isolate one variable
each; the CHERI-default row cannot do that job for PoisonCap, because it differs
from it in kernel, emulator, libc, build and revocation setting at once.

Mode 1 is the one negative that says something on its own. It **has** revocation,
at the block — and the pool never frees its block, it recycles
out of it. In the port's twelve-case suite that arm catches exactly one defect,
`buffer-read-after-backing-free`, and none of these three. That separates "no
mechanism" from "a mechanism at the wrong point", which is the distinction the
whole argument turns on.

The protected oracles are not interchangeable. A domain halts and publishes a
fault PC, which the runner compares against the address that boot printed for
its probe. A CheriBSD process reports only a status, so there the setup marker
carries the weight: it is printed only once reuse at the same address has been
checked. No timing comparison is made or implied — the arms run on different
emulators.

## Where this corpus deviates from the contract, and why

[`../../tools/check-corpus.py`](../../tools/check-corpus.py) enforces
[SCHEMA.md](../../SCHEMA.md) over every corpus, this one included, reading the
`corpus.json` beside these cases. The deviations this section used to list are
now part of the contract instead of complaints: the extra arms are declared
arms, and the case macro is declared (`FF2`) rather than assumed to be the
pymalloc corpus's `PYC`. What each one means is unchanged.

| arm | what it is |
|---|---|
| `native-fix-differential` | the contract's arms differ by **protection**, the defect present in both. This pair differs by whether the **upstream fix** is applied. Folding it into `spatial`/`sublet` would misname it |
| `cheribsd-revocation` | stock CheriBSD with `libc` revocation enabled: a fourth system. **Declared, NOT measured** — this line previously read "the arm that makes the blindness claim a measurement", which was retracted on 2026-10-06; see the retraction note above |

`poisoncap-protected` carries `si_code: null` with a note. The runner records a
process exit status; 162 is 128+34 so the signal is derived, but `si_code` is not
recoverable from an exit status and is not guessed.

Extending the checker to know these is a change to the pymalloc corpus and
belongs in a conversation with it, not a unilateral edit from here.
