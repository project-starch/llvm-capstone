# tshark: reuse distance at wmem, stock against protected (2026-10-11)

Pre-registered in `capstone/docs/plans/2026-10-11-nested-reuse-distance.md` (16d2171a3edc, pushed
before any instrumented build or run). Wireshark row: wmem block and block_fast object hand-out
and free; the capture set. Prediction: stock and protected give the same reuse histogram at every
observed boundary, bucket for bucket.

**Verdict: as predicted.** At both boundaries, `wmem-block` and `wmem-block-fast`, the two arms
give identical histograms, and identical issues, releases, reuses, distinct starts and attempts.
This holds summed over the capture set and also on each of the ten captures separately. The
observer's `error` is 0 in all 40 reports: 2 boundaries x 10 captures x 2 arms.

| boundary | arm | issues | releases | reuses | distinct starts | attempts | error | median bucket |
|---|---|---|---|---|---|---|---|---|
| wmem-block | stock | 7,221 | 7,221 | 135 | 7,086 | 7,221 | 0 | 2 |
| wmem-block | protected | 7,221 | 7,221 | 135 | 7,086 | 7,221 | 0 | 2 |
| wmem-block-fast | stock | 10,966 | 10,966 | 4,850 | 6,116 | 10,966 | 0 | 8 |
| wmem-block-fast | protected | 10,966 | 10,966 | 4,850 | 6,116 | 10,966 | 0 | 8 |

Share of reuses whose bucket differs (sum_k |s_k/N_s - p_k/N_p| / 2): **0.0** at both boundaries.

**Buckets** (bucket k holds 2^(k-1) <= d < 2^k; equal in both arms):
- wmem-block: k1 3, k2 130, k4 2;
- wmem-block-fast: k1 15, k2 30, k3 60, k4 120, k5 240, k6 380, k7 765, k8 968, k9 195, k10 1,479,
  k11 598.

The numbers are sums over the ten processes, one per capture. `distinct_starts` is the sum of
the per-process counts.

## Hook placement (`capstone/experiments/study/reuse-hooks/wireshark/`)

- `wmem-reuse-gap.patch`: everything it adds is behind `EXP_REUSE_GAP`.
- `wmem-reuse-gap.h`: the observer helpers, one static observer per boundary in the file that owns
  it (a 65,536-slot static table), and the exit-time report:
  `REUSE-GAP <boundary> arm= attempts= issues= releases= reuses= distinct_starts= error= capacity= b1= ... b32=`.
- `build-wireshark-reuse.sh`: the build wrapper.

**Placement.** The hooks are in upstream's consumer functions. Patch 0001 keeps those functions
and calls them from its wrappers with the allocator's own pointer, so in both arms the hooks
record the allocator's address, never the caller's capability.

| event | where | what |
|---|---|---|
| hand-out | `wmem_block_alloc`, `wmem_block_alloc_jumbo`, `wmem_block_fast_alloc` (both returns), just before the user's pointer is returned | `reuse_gap_attempt`, then `reuse_gap_issue(start, size)` |
| free | `wmem_block_free`, before the chunk is merged or the jumbo block freed | `reuse_gap_release` |
| reset | `free_all` of both allocators, before anything is reinitialised | release of every live start inside each block, and of each jumbo object |
| in-place realloc | `wmem_block_realloc`'s grow into the next chunk, and its shrink | `reuse_gap_resize` (no new start) |
| jumbo realloc | both allocators (through the system allocator) | resize if the start is unchanged, else release + issue |

Two cases need no hook of their own:
- a `block` realloc that moves goes through the hooked `wmem_block_alloc` and `wmem_block_free`;
- a `block_fast` grow goes through the hooked `wmem_block_fast_alloc` and leaves the old object
  live.

`block_fast`'s free is a no-op upstream, so its objects are released only by the reset that lets
the allocator reuse their addresses. That is a deliberate departure from "hook the free": hooking
the no-op would release addresses the allocator does not reuse until the reset.

**One patch for both arms.** The patch applies at fuzz 0 to the released tree. On top of wmem
patch 0001 it applies at fuzz 0 with line offsets only, adding the same lines at the same places,
so no second variant was needed. The build recompiled the two allocator objects with ninja's own
compile command plus `-DEXP_REUSE_GAP`; for the protected arm, `-DEXP_REUSE_GAP_PROTECTED` was
added as well, and it only labels the report.

**Native checks of the hooks**, with the wmem port's GLib shim (a scratch harness, not kept):
- 50 rounds of mixed alloc, realloc (jumbo included), free and `free_all` on both allocators gave
  `error=0`, with every issue released and reuses counted.
- With the `block_fast` reset release removed, the observer reports `error=2` (an overlapping
  issue). So the reset hook is what makes `block_fast` measurable, and the error check fires.

## Inputs and runs

- **Source:** p12 snapshot `ae06300ecd92`. It is 16d2171a3edc's `capstone/` tree, plus the
  uncommitted `TSAPP_SUBLET` wiring (see `../2026-10-11-same-answers/README.md`), plus the three
  hook files. The runtime and port paths are unchanged from 9bc6f9adf1c9, and the runtime archive
  is the same as the same-answers images' (`e5693f5c...e545`).
- **Hook files** (sha256):
  - `wmem-reuse-gap.patch` `b192b4c1...479a`;
  - `wmem-reuse-gap.h` `91f1e94c...f210`;
  - `reuse-gap-observer.h` `627b0fe3...1769`.
- **Images** (`build-wireshark-reuse.sh stock|protected`):
  - stock, nested=none: `7c9c894646329c96263d7870e0a1b5c8ca51efee98eba62cecbfe71bee03b44e`;
  - protected, nested=wireshark: `0b549748ae45eb828f1a5e90abaccec882637e9836040de63f307ade39df385b`.
  - Only the protected image contains the `wmem_block_sublet_*` wrappers, and both contain the
    `wmem_rg_*` hooks.
- **Platform:** QEMU af37cc32, kernel e833dfb1, launcher 3975314a, module a1c6cb6b, exact bounds,
  one hart (full hashes in `reuse.json`).
- **Runs:** one boot per arm, the two boots in parallel with separate `CAPSTONE_QEMU_LOCK` files.
  - Each capture is one `tshark -r C -V -n` process (TZ=UTC, empty configuration).
  - Every run exits 0, with no fault.
  - Every run prints the same stdout as native stock tshark, and for ntp the native minimal
    build's output, as in the same-answers run. So the instrumentation does not change the
    output.
  - Runner: `run-reuse.py` here; its sha256 and the gate's are in `reuse.json`.
- **Wall time:** the instrumented builds took 1,393 s (stock) and 1,429 s (protected), in
  parallel. The boots took 499 s and 505 s.

## Repeats, 2026-10-12

Two more runs per arm, pre-registered in
[`docs/plans/2026-10-12-nested-reuse-repeats.md`](../../../../../docs/plans/2026-10-12-nested-reuse-repeats.md)
(82de1ec3, pushed before any run). In `reuse.json`, `buckets` and the counters beside it stay run 1
(the plotted one), `repetitions` lists runs 1-3 per boundary and arm (each summed over the ten
captures as before), and `repeats` records the setup and each new run's checks.

- **Host and boots:** p12, four VMs in parallel (stock and protected, runs 2 and 3), each its own
  boot with its own `CAPSTONE_QEMU_LOCK` and a fresh work directory; boots took 456 to 458 s.
- **Images:** the 2026-10-11 images in the p12 build directories, unchanged: stock
  `7c9c8946...b44e`, protected `0b549748...385b`, equal to the record's (checked on the host before
  the boots, and against each image's manifest by the runner).
- **Runner:** the same `run-reuse.py` (sha256 `28615a93...b3fb`, equal to `runner_sha256`), same
  kit, captures and repo; only the work directory and the lock file differ. The gate script's
  sha256 equals the record's `gate_sha256` in every run.
- **Checks:** in every run, all ten captures exit 0, no domain fault, and each capture's stdout
  sha256 equals run 1's (which matched native stock tshark, and the native minimal build for ntp).
  The observer's `error` is 0 in all 80 new reports (2 boundaries x 10 captures x 4 runs).

| boundary | arm | run | issues | releases | reuses | equal to run 1 |
|---|---|---|---|---|---|---|
| wmem-block | stock | 2 | 7,221 | 7,221 | 135 | yes |
| wmem-block | stock | 3 | 7,221 | 7,221 | 135 | yes |
| wmem-block | protected | 2 | 7,221 | 7,221 | 135 | yes |
| wmem-block | protected | 3 | 7,221 | 7,221 | 135 | yes |
| wmem-block-fast | stock | 2 | 10,966 | 10,966 | 4,850 | yes |
| wmem-block-fast | stock | 3 | 10,966 | 10,966 | 4,850 | yes |
| wmem-block-fast | protected | 2 | 10,966 | 10,966 | 4,850 | yes |
| wmem-block-fast | protected | 3 | 10,966 | 10,966 | 4,850 | yes |

**Verdict: as predicted.** Every new histogram equals run 1's bucket for bucket (the buckets
listed above), every counter (issues, releases, reuses, distinct starts, attempts) equals run 1's,
and stock equals protected in every run. This also holds per capture: each new run's REUSE-GAP
fields for each of the ten captures equal run 1's.

## Also seen

- **Hand-outs against the memory row's extra lifetimes:** wmem hand-outs per process (both
  boundaries) range from 801 (arp) to 3,318 (dns_port). The protected arm's extra lifetimes in
  `../2026-10-11-memory-behaviour` are 0.76 to 1.30 times that, which fits "about one per
  hand-out". Stock's own run-to-run spread is several hundred lifetimes, so the ratio is no
  sharper than that.
- **Where reuses come from:** `wmem-block` (the file and epan scopes) almost never reuses an
  address within a run: 13 to 16 reuses per process. `wmem-block-fast` (the packet scope, rewound
  per packet) reuses on the captures with many frames per pool (dhcp, dns_port, dns-ooo), and not
  at all on http, arp and ntp.

No serial log is kept.
