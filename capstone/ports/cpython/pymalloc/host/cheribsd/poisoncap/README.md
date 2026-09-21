# CPython pymalloc on PoisonCap

Experimental protection for the extracted CPython 3.13.7 allocator, using the
existing native recordings and lifetime hooks. The interpreter, Python object
free lists above pymalloc, garbage collector and extensions are outside this
port. The allocator is single-threaded and trusted. Its library target remains
`CPython::Pymalloc`.

The [first pilot](../../../results/20260919-poisoncap/README.md) records the
complete 33-process suite, replay measurements, failed attempt and provenance.

## Platform and build

Use the already reconstructed [PoisonCap platform](../../../../../ffmpeg/buffer-pool/host/cheribsd/poisoncap/README.md).
Its pinned LLVM, QEMU and CheriBSD sources are reused without additional
platform patches. SDKs, source archives, builds and raw guest logs stay under
external temporary storage.

From the repository root:

```sh
export CAPSTONE_LLVM_BUILD_DIR=/path/to/capstone-llvm-build
source capstone/tests/capstone-test-env.sh
export CHERI_SDK=/tmp/capstone/poisoncap-work/sdk
export CHERI_SYSROOT=/tmp/capstone/poisoncap-work/output/rootfs-riscv64-purecap
PORT=capstone/ports/cpython/pymalloc
PC="$PORT/host/cheribsd/poisoncap"
BUILD=/tmp/capstone/poisoncap-pymalloc-work/build/poisoncap
bash "$PC/build.sh" "$BUILD"
# Optional application entry linked to CPython::Pymalloc:
bash "$PC/build.sh" "$BUILD" -DPORT_CLIENT_SOURCE=/absolute/path/my-program.c
```

The [example](../../../examples/pymalloc.c) maps a 16 KiB-aligned backing
region with poison authority, initializes the metadata allocator and lifetime
adapter, explicitly selects mode 1, and calls the real pymalloc API. A direct
client must perform the same initialization. Linking alone is insufficient.
The platform controls reuse the FFmpeg integration's standalone instruction
probe. The CMake build selects GNU C for the published revocation header.

## Run

```sh
python3 "$PC/run.py" "$BUILD" /tmp/capstone/pymalloc-poison-pools-1 \
  --sdk "$CHERI_SDK" --rootfs "$CHERI_SYSROOT" \
  --image /tmp/capstone/poisoncap-work/output/cheribsd-riscv64-purecap.img \
  --disable-default-revocation
```

The Python environment needs `pexpect`. The common runner holds the QEMU lock,
uses a disposable snapshot and retains failed attempts. Output directories
must be new. Guest libc automatic revocation is explicitly off; the recommended
flag also disables the guest default before SSH starts, avoiding the known
VM-locking failure of this published platform. Explicit PoisonCap revocation
by the adapter remains active. This is not whole-process temporal protection.

Mode 0 uses bounded spatial leases; mode 1 adds lifetime invalidation. The
PoisonCap replay binary defaults to mode 1 and accepts an explicit final
argument `0` or `1`. The runner always passes it explicitly.

The full pool suite includes an example, five platform instruction controls,
five additional API checks, and nine paired spatial/protected lifetime cases,
plus the shared ABI and bounds controls. Stale free is an explicit rejection
(exit 1 with both setup and exact rejection markers), rather than a capability
fault. Other invalid loads/stores require the setup marker and SIGPROT.
The repeated-address case defaults to eight iterations; use `--reuse-rounds
2000` for the longer variant. The separate arena-turnover check allocates and
frees 2,300 512-byte objects and requires real arena release. `--case NAME`
selects a diagnostic subset and records that it is not the complete suite.

## Native recording and replay

Build the native component with `PYMALLOC_RECORD_PYTHON` naming a GIL-enabled
CPython 3.13.7. Then use the existing recorder. `--items` controls workload size;
its default of 80 preserves the original driver workload.

```sh
PYTHONMALLOC=pymalloc /path/to/python3.13 "$PORT/host/record.py" "$TRACE" \
  --module-dir "$NATIVE_BUILD/python" --rounds 1 --items 1
"$NATIVE_BUILD/bin/replay-reference" "$TRACE" "$NATIVE_REPORT"
python3 "$PC/run.py" "$BUILD" "$RESULTS" \
  --sdk "$CHERI_SDK" --rootfs "$CHERI_SYSROOT" --image "$IMAGE" \
  --disable-default-revocation --stage replay \
  --recording "$TRACE" --native-report "$NATIVE_REPORT"
```

This runs the complete control suite and replays the recording in both modes.
`replay-validation.json` checks operation counts, completion, mode and event
checksum against the native oracle. The guest also checks payload bytes before
free/realloc and after realloc; the reported checksum fingerprints events.
Arena counts, metadata watermarks and
allocation addresses are not required to match across ABIs. Trace payloads
are synthetic checks, not a reconstruction of Python object graphs. The
larger existing recording is a separate workload; success on a small recording
does not establish its success.

## The CPython defect corpus

The twenty reduced CPython consumer-defect sequences in
[`capstone/bug-corpora/cpython/pymalloc-repros`](../../../../../../bug-corpora/cpython/pymalloc-repros/README.md)
build against this adapter through the port's `PY_CORPUS_SRC` seam. The corpus
source is not copied or vendored into the port; the port supplies the hosted
entry, the real `Objects/obmalloc.c` and the adapter, and the corpus supplies
its own `pym_replay`.

```sh
BUILD=/tmp/capstone/poisoncap-pymalloc-corpus-work/build/poisoncap
CORPUS=$PWD/capstone/bug-corpora/cpython/pymalloc-repros
bash "$CORPUS/shared/build-cases.sh" cheribsd "$BUILD"
python3 "$CORPUS/runners/cheribsd/run-defects.py" "$BUILD" /tmp/capstone/pymalloc-defects-1 \
  --sdk "$CHERI_SDK" --rootfs "$CHERI_SYSROOT" \
  --image /tmp/capstone/poisoncap-work/output/cheribsd-riscv64-purecap.img \
  --disable-default-revocation
python3 "$CORPUS/runners/cheribsd/run-defects.py" "$BUILD" /tmp/capstone/pymalloc-defects-control-1 \
  --sdk "$CHERI_SDK" --rootfs "$CHERI_SYSROOT" --image "$IMAGE" \
  --disable-default-revocation --negative-control
```

That produces `bin/defects`, which takes the same `input.bin output.bin MODE`
arguments as the other hosted programs and reads one 128-byte fixture: a
12x uint64 header followed by one 4x uint64 event whose id is the case number.
Each case runs in both modes against that one binary. Mode 0 is
request-bounded spatial authority with no per-object invalidation and the
sequence must complete; mode 1 adds this adapter's lifetime invalidation and
the stale read must raise `SIGPROT` with `si_code == PROT_CHERI_TAG` at the
labelled `pyc_defect_read` instruction. Exit 162 alone is not accepted: it is
the status of every `SIGPROT`, so the program's own handler prints the trap PC
from `ucontext_t.uc_mcontext.mc_capregs.cp_sepcc` beside the probe label's
address, and the runner requires a complete
`PYC_DEFECT_FAULT case=N signal=34 code=2 pc=... expected=... exact=1` line with
the two addresses identical, the case's `ready` marker before it, and the
platform status. The handler then re-raises with the default disposition, so a
fault is never converted into a clean exit. The negative control corrupts every
fixture so the program refuses it before any case runs, executes every selected
arm anyway, and exits 0 only when every selected oracle reports a failure.

The corpus's case 5 faults at the labelled byte read, before the pointer load
that follows it can execute; a protected case-5 arm therefore shows the stale
access refused, not the stale link being followed and refused.

Scope is the corpus's own: a real allocator with reduced consumers. These are
not interpreter runs, and they cover neither the per-type free lists nor
`PyArena` above pymalloc. As everywhere else in this port, automatic guest libc
revocation is explicitly disabled for the documented platform workaround while
the adapter's explicit sweeps remain enabled.

## Storage and policy

The payload budget is split into 32 MiB small-object arenas and 32 MiB raw
fallback. Block/arena authority records, replay scratch and snapshot buffers
use the separate 16 MiB metadata heap. No new upstream CPython patch is needed.
The existing extraction, provenance and lifetime-hook patches are reused.

On free, the adapter poisons the block and finishes a sweep before clearing
poison, overwriting the stored poison capabilities with zeros, and permitting
pymalloc's in-band free-list write. Clearing access state alone leaves poison
capabilities that a later kernel sweep can mistake for a newly issued lease.
The `unwritten-reuse` check covers zero-size and raw allocations that receive
no client stores before an unrelated free triggers another sweep. Pool reassignment
and empty-arena release have their own invalidation boundaries. Pool/arena
release accepts only empty allocator state; this is not an adversarial parent
revoking live allocations owned by an untrusted child manager.

In-place realloc uses a capability-preserving snapshot, invalidates old client
aliases, then restores the payload and returns fresh bounded authority.
Failed snapshot allocation leaves the original allocation usable. Freed-object
contents do not require the persistent snapshots used by the FFmpeg pools.
Metadata allocation caches snapshot buffers, so peak requested snapshot size
is not their total retained storage. Metadata high-water includes that storage.

`PYM_POISONCAP` reports mode, sweeps, bytes poisoned/cleared/zeroed/copied, peak requested
snapshot size, pool reclassifications and pointer size. `copied_bytes` counts
only the additional snapshot save/restore traffic for protection; it excludes
ordinary moved-realloc copies, calloc/replay writes and kernel scan traffic.
`zeroed_bytes` counts the additional retired-poison payload overwrite.
The ordinary report
retains arena and metadata counters. These are operation/resource measurements,
not total process/OS memory or hardware timing. Synchronous sweeps can make
large QEMU traces expensive; optimizing quarantine is outside this first policy.
