# FFmpeg pool defect corpus

Consumer-side temporal defects whose stale storage is memory an `AVBufferPool`
handed out. `pool_release_buffer` (`libavutil/buffer.c:344`) pushes a payload
onto a LIFO freelist and `av_buffer_pool_get` (`:390`) hands the identical
`buf->data` back, so nothing reaches `malloc` and same-address reuse is a
property of the allocator rather than of a run.

    00_461fb22053_af_join_dedup_bound/            a reference is never taken; stale read
    01_1886c3269d_h264_refs_partial_clear/        reset bounded by the count, not the array
    02_316531e61c_vidstab_parked_plane_pointer/   pointer parked in a library; stale write

| shape | cases |
|---|---|
| reference never taken / reuse / stale read | 0 |
| partial clear / reuse / stale read | 1 |
| parked pointer / reuse / stale write | 2 |

Three cases, three shapes. The inventory and triage that selected them, and the
further pool-backed specimens it found that are not built here, are in
[`docs/ref/ffmpeg-pool-consumer-defects.md`](../../../docs/ref/ffmpeg-pool-consumer-defects.md).

## The contract

The layout and the `case.json` fields are the corpus contract in
[`cpython/pymalloc-repros/SCHEMA.md`](../../cpython/pymalloc-repros/SCHEMA.md),
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
| **CHERI default** | `free()` → quarantine → sweep | — | — | — |
| **PoisonCap** | the lease return: poison, then sweep before reissue | **SIGPROT** 162 | **SIGPROT** 162 | **SIGPROT** 162 |

It catches whoever listens for the moment the inner allocator takes the storage
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
at the backing allocation — and the pool never frees its backing, it recycles
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

`tests/check-corpus.py` in the pymalloc corpus enforces
[SCHEMA.md](../../cpython/pymalloc-repros/SCHEMA.md). Run against these cases it
reports exactly three kinds of problem, all of them deliberate. They are listed
here rather than silenced, and no copy of that checker is shipped beside them: a
fork would be a second contract, and a checker that fails by design is noise.

| what it reports | why |
|---|---|
| `arm 'native-fix-differential' is not in SCHEMA.md` | the contract's arms differ by **protection**, the defect present in both. This pair differs by whether the **upstream fix** is applied. Folding it into `spatial`/`sublet` would misname it |
| `arm 'cheribsd-revocation' is not in SCHEMA.md` | stock CheriBSD with `libc` revocation enabled is a fourth system the contract does not yet name. It is the arm that makes the blindness claim a measurement |
| `case.c declares no PYC_CASE` | the macro is the corpus's seam to its allocator; here it is `FF2_CASE`/`APR_CASE`. The rule the checker means — a case declares the number its directory carries, and the driver refuses a fixture that names another — is implemented |

`poisoncap-protected` carries `si_code: null` with a note. The runner records a
process exit status; 162 is 128+34 so the signal is derived, but `si_code` is not
recoverable from an exit status and is not guessed.

Extending the checker to know these is a change to the pymalloc corpus and
belongs in a conversation with it, not a unilateral edit from here.
