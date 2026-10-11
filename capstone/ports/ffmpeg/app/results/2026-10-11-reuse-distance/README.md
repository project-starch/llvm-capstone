# FFmpeg 9.0.1: reuse distance at AVBufferPool and AVRefStructPool, stock against protected (2026-10-11)

The FFmpeg row of the pre-registration
[`docs/plans/2026-10-11-nested-reuse-distance.md`](../../../../../docs/plans/2026-10-11-nested-reuse-distance.md)
(16d2171a3edc, pushed before any instrumented build or run). Boundaries: **buffer-pool**
(AVBufferPool entry hand-out and return) and **refstruct-pool** (AVRefStructPool entry hand-out
and return). Workload: the decoder application on the run-of-record clip,
`capstone-vexec ffmpeg.dom input.mkv 1` (30 frames), as in the memory-behaviour run. Arms:
**stock** (FFmpeg's pools as released) and **protected** (`FFAPP_SUBLET=1`: buffer-pool patch
0003 on top of 0001), both with the same observer hooks behind `EXP_REUSE_GAP`. N = 1 per arm
(deterministic), both VMs in the same batch as the Perl row. Data: [`reuse.json`](reuse.json)
(bucket k = observer `bins[k-1]`, empty buckets omitted).

## Outcome

| | buffer-pool stock | buffer-pool protected | refstruct-pool stock | refstruct-pool protected |
|---|---|---|---|---|
| hand-outs (`attempts` = `issues`) | 90 | 90 | 240 | 240 |
| releases | 90 | 90 | 240 | 240 |
| reuses | 81 | 81 | 216 | 216 |
| distinct start addresses | 9 | 9 | 24 | 24 |
| entries freed to the system allocator (`drops`) | 9 | 9 | 24 | 24 |
| reuses at a newly allocated entry (`fresh_reuses`) | 0 | 0 | 0 | 0 |
| observer `error` | 0 | 0 | 0 | 0 |
| median bucket | 2 (distance 2-3) | 2 | 3 (distance 4-7) | 3 |

| bucket (distance) | 1 (1) | 2 (2-3) | 3 (4-7) | 4 (8-15) |
|---|---|---|---|---|
| buffer-pool, both arms | 27 | 54 | -- | -- |
| refstruct-pool, both arms | 27 | 54 | 108 | 27 |

**Share of reuses whose bucket differs between the arms** (sum over k of
|stock_k/N_s - prot_k/N_p| / 2): **0** at both boundaries. The REUSE-GAP lines of the two arms are
identical, character for character. Both arms: exit 0, 30 frames, every frame hash equal to the
native reference (native.out 3eebf311c9fa...), stdout identical (sha256 6175afea1c8c8913...).

## Verdicts

- **Prediction 1 (equal histograms), FFmpeg: holds** at both observed boundaries, bucket for bucket,
  with the same issues, releases, reuses and distinct starts.
- **Prediction 3 (the figure), FFmpeg panel:** the stock and protected curves are the same curve at
  each boundary. The figure itself is drawn by the paper script from `reuse.json`, not here.
- Prediction 2 concerns SQLite only.

Every entry is released before it is handed out again (releases = issues), and every pool frees all
its entries at the end (`drops` = distinct starts). No new entry landed on an address an earlier,
freed entry had (`fresh_reuses` 0), so every reuse counted here is the pool reissuing its own
entry.

## Hook placement

Patches and wrapper: [`experiments/study/reuse-hooks/ffmpeg/`](../../../../../experiments/study/reuse-hooks/ffmpeg/).
Buffer-pool 0001 (trace calls) and 0003 (the CDERIVE/CREVOKE lines and the sublet fields) change
the hooked functions, so there are two hook variants with identical placement relative to
upstream's own statements:

| | stock (`pools-reuse-gap-stock.patch`) | protected (`pools-reuse-gap-protected.patch`) |
|---|---|---|
| buffer-pool hand-out | `libavutil/buffer.c`: `av_buffer_pool_get`, after the pool lock is dropped, for every successful get, reissued (`pool->pool`) or new (`pool_alloc_buffer`) | the same place (after 0003's CDERIVE of the issued child) |
| buffer-pool return | `buffer.c`: `pool_release_buffer`, before the lock that pushes the entry on the free list | the same place (after 0003's CREVOKE) |
| buffer-pool address | `BufferPoolEntry.data`, the entry's own storage pointer; size `pool->size` | the same field (0003 keeps the original pointer there) |
| refstruct-pool hand-out | `libavutil/refstruct.c`: `refstruct_pool_get_ext`, right before the object is stored to the caller | the same place (after 0003's CDERIVE) |
| refstruct-pool return | `refstruct.c`: `pool_return_entry`, before the pool lock | the same place (after 0003's CREVOKE) |
| refstruct-pool address | the entry's start, its RefCount header (object - `REFCOUNT_OFFSET`); size `REFCOUNT_OFFSET + pool->size` | the same address (0003's issued child and the entry's lifetime start there) |
| drops | `buffer_pool_flush` before freeing an idle entry; `pool_free_entry` before `av_free` | the same places |

Each boundary has its own static table (2^14 slots) and `atexit` report (one `REUSE-GAP <boundary>`
line on stderr, formatted without printf). A hand-out is `reuse_gap_attempt` +
`reuse_gap_issue`; whether the entry was new is passed alongside only to count `fresh_reuses`,
which does not change the histogram. The address is taken as `(uint64_t)(uintptr_t)` of the
pointer; the compiler warns that this casts a capability to a smaller integer (four warnings per
arm, at these four casts), which is the intended reading of its address (the sv-heads adapter
does the same).

The protected objects carry the lifetime instructions and the stock ones do not: the disassembler
leaves 8 instructions in `buffer.o` and 5 in `refstruct.o` of the protected build undecoded (the
CDERIVE/CREVOKE encodings; the virtual-pools run counted the same 8 and 5), and none in the stock
objects.

## Builds

Source: the snapshot described in the Perl row's README (`git archive 16d2171a3edc capstone` plus
the uncommitted `experiments/study/reuse-hooks/{perl,ffmpeg}`, plus `compiler-rt/lib/builtins`
from 9bc6f9adf1c9; snapshot commit f21690b, which the image manifests record with
`runtime_dirty` false). Driver: the Perl row's
[`build.sh`](../../../../perl/musl/results/2026-10-11-reuse-distance/build.sh), which runs
`experiments/study/reuse-hooks/ffmpeg/build.sh ARM OUT ffmpeg-9.0.1.tar.xz` (tarball
cf38e0e28c7e5605..., checked by prepare-source.sh against upstream.json). The wrapper repeats
`common/application/build-virtual.sh`'s ffmpeg branch with the hook patch between source
preparation and build: `host/prepare-source.sh` (stock) or `--sublet` (protected), the hook patch
at fuzz 0 plus the two headers into `libavutil/`, `host/build-domain.sh` with
`FFAPP_SUBLET=0|1`, `FFAPP_EXTRA_CONFIGURE=--extra-cflags=-DEXP_REUSE_GAP`,
`FFAPP_EXTRA_FIXTURES="11 12 13 14 15"`, then `build.py --app ffmpeg --profile virtual
--nested none|ffmpeg --stack-bytes 1048576`. It checks that the define is in the libraries'
CFLAGS and that `buffer.o` and `refstruct.o` carry the hook tables.

| | stock | protected |
|---|---|---|
| image (`image/ffmpeg.dom`) | 063fc55dd001131698190bc20418bd0133f969906564cd33f8ee7418184423a3 | 1996436b89fe7cb9924d12f6f84e58221aa5565213303a73f9a743b277728010 |
| prepared source key | src-9.0.1-f7e8b586f7b4 | src-9.0.1-sublet-a08a5bd0dcb2 |
| hook patch | a1c6398854332cef... | f90634de57d720487... |
| build wall | 70 s | 70 s |

An earlier build of both images from the same inputs (before a wrapper-check fix that touched only
the Perl checks and the nm call) produced byte-identical images. Inputs: hook header
f95a0717be512aba..., observer 627b0fe3ac30d75b..., buffer-pool 0001 b0acc8d2fb146339...,
0003 5d3cd35a00ae199c....

## Runs

Script and parser (shared with the Perl row):
[`run.sh`](../../../../perl/musl/results/2026-10-11-reuse-distance/run.sh),
[`parse.py`](../../../../perl/musl/results/2026-10-11-reuse-distance/parse.py). Each VM boots
through `runtime/virtual/run-staged.py --timeout 3600` with its own `CAPSTONE_QEMU_LOCK` and a fresh
`--work`. In the guest: `insmod`, then from `/mnt/vm`

    CAPSTONE_EXEC_DIAGNOSTICS=1 /mnt/vm/capstone-vexec /mnt/vm/prog.dom /mnt/vm/input.mkv 1 > out 2> err

then `rmmod` (exit 0 on both). Frames compared on the host with the regex of `ffapp-decode.sh`
against native.out (input.mkv 7c0964440ad178a0...). stderr: the two REUSE-GAP lines and the
workload's own `MEMPHASE before-0` / `MEMPHASE released-0`, identical on both arms.

Platform: as in the Perl row (QEMU af37cc32, launcher 3975314a, module a1c6cb6b, Image e833dfb1,
run-staged.py d6770b8b, clang 4c6135fe; full hashes in `reuse.json`).

**Wall times**: `capstone-vexec` in the guest: stock 8.18 s, protected 8.09 s. Boots
(run-staged.py start to end): 16 s each; the batch of 8 VMs 56 s. Not a performance comparison.

## Cross-reading with the memory-behaviour run (not registered)

The memory-behaviour run could not check "about one lifetime per hand-out" because FFmpeg exposes
no hand-out count. Here the two pools hand out 90 + 240 = 330 entries, 33 of them newly allocated
(9 + 24 distinct starts, `fresh_reuses` 0); the protected arm there allocated 325 more lifetimes
than stock. Read from patch 0003's source, a new entry derives two lifetimes (its lifetime and
the issued child) and a reissue one, which would give 363, not 325; the difference is not
reconciled here (nothing in this run measures lifetimes). Different
images (the hooks), so this is a reading across runs, not a check.

## Deviations

- The wrapper's first build of both FFmpeg images passed; they were rebuilt after the Perl wrapper's
  check fixes so all four images come from one snapshot commit (f21690b). The rebuild is
  byte-identical to the first build.
- `CAPSTONE_EXEC_DIAGNOSTICS=1` in the guest environment, as in the memory-behaviour run.
- N = 1 per arm as registered. Raw serial logs not committed.
