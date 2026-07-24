# Cycle-accurate borrow cost on Capstone silicon (Genesys 2 / CVA6)

**Vehicle:** CapliFive CVA6 "Capstone" core, bitstream
`working-caplifive-captype-fixed.bit`, OpenSBI Capstone monitor, image
`fw_payload_fpga_up_ctl.bin`. **Timer:** `mcycle` read inside a Capstone domain.
**Method:** each operation is an `N`-iteration inner loop bracketed by two `mcycle`
reads with an empty calibration loop subtracted; figures are cycles per operation.
The measured loop bodies are byte-identical to the QEMU `borrow-cost` probe. This is
the first Capstone temporal-safety measurement to run on hardware.

## Workload

A microbenchmark, not an application. One Capstone domain runs a fixed set of tight
inner loops, each an `N`-iteration body bracketed by two `mcycle` reads with an
empty calibration loop of the same trip count subtracted; the reported figure is
cycles per operation. The loop bodies are:

- **raw** — one plain word load through an ordinary pointer.
- **borrow** — the full revoke-at-free sequence per iteration: `mrev` + `delin` +
  load + `revoke`.
- **copy@256 / copy@1024** — a word-copy loop over a 256 B / 1024 B payload (the
  defensive whole-object copy the borrow replaces).
- **primitives** — `load`, `shrink`, `mrev`, and `mrev`+`delin`+`revoke`, each in its
  own bracketed loop (the breakdown probe).

The data footprint is one scratch capability plus a single small shared region; the
instruction footprint is a handful of instructions. Loop bodies are byte-identical
to the QEMU `borrow-cost` probe, so the silicon cycles and the QEMU instruction
counts measure the same code.

## Measurement conditions

- **Warm cache, all L1 hits.** Given the tiny footprint, iteration 0 warms the L1
  I$/D$ and iterations 1…N−1 hit; the empty-loop subtraction removes loop and warm
  instruction-fetch overhead. The load-side figures carry no cold-miss or
  DRAM-latency component.
- **D$: 32 KiB, 8-way, 128-bit line (one capability per line), write-through with a
  write buffer (no-write-allocate).** Revocation-tree accesses share the D-cache
  ports, so the *stores* inside `mrev`/`revoke` (node writes + shadow-tag updates)
  stream to DRAM through the write buffer — that traffic is part of the measured
  figures, not hidden. Because stores don't allocate, they don't pollute L1; only the
  `revoke`-walk *reads* allocate (≤~2.25 KB). This is **conservative**: a write-back
  D-cache (a config this CVA6 family also ships) or a dedicated rev-node cache would
  keep the hot tree on-chip and drop the per-write DRAM traffic.
- **No tag cache.** Capability tags ride in the L1 D$ line's `user` bits (free on a
  hit), backed by a shadow tag table in DRAM (1 bit / 16 B) that is touched only on a
  D$ miss/writeback — which the warm loop does not incur.
- **Single core; `mcycle`** read in-domain (the unprivileged `cycle` counter is gated).

## Operations

**Super-operations** — the same boundary lend, protected three ways:

- **raw** — plain load of the borrowed word through an ordinary pointer (no safety).
- **borrow** — the revoke-at-free temporal-safety sequence for one lend:
  `mrev` (mint a revocation capability) + `delin` (de-linearise a working cap to
  hand out) + load + `revoke` (invalidate the delegated cap, reclaim the linear
  handle).
- **copy@256 / copy@1024** — the defensive whole-object copy the borrow replaces,
  for a 256-byte and a 1024-byte payload.

**Elementary primitives** — what the borrow decomposes into, plus the spatial-safety
primitive:

- **load** — one word read.
- **mrev** — mint a revocation capability from a linear handle (allocates a
  revocation-tree node).
- **delin** — de-linearise a linear cap into a copyable working cap.
- **revoke** — invalidate the delegated cap and reclaim the linear handle (walks the
  revocation tree). `delin` is required for `revoke` to return a reusable linear cap,
  so the two are measured together.
- **shrink** — narrow a capability's bounds to a sub-range (the per-object
  spatial-safety primitive; the `SHRINK` op the compiler emits at materialisation).

## Super-operation cost (mcycle, cyc/op)

A borrow replaces the defensive whole-object copy; the headline is that it is
several-fold cheaper than that copy and O(1) in object size. The `vs raw` column is
the mechanism measured against a bare load — its cost in isolation, **not** a program
slowdown (a borrow is paid once per lend and amortized over the object's uses).

| operation | cyc/op | vs raw | cost model |
|-----------|-------:|-------:|------------|
| raw pointer            | 8    | 1.0× | O(1) |
| **capability borrow**  | **182** | **22.8×** | **O(1) in payload** |
| copy — 256 B           | 902  | 112.8× | O(payload) |
| copy — 1024 B          | 3611 | 451.4× | O(payload) |

Copy is stable and cycle-accurate — ~900 cyc at 256 B, ~3600 at 1024 B, i.e.
O(payload). The borrow is payload-independent and 2–5× cheaper than even the 256 B
copy, ≥7.7× cheaper at 1024 B; the gap widens with payload.

## Elementary-primitive cost (mcycle, cyc/op)

| primitive | cyc/op |
|-----------|-------:|
| load           | 2   |
| shrink         | 1   |
| mrev           | 50  |
| delin + revoke | 121 |
| **mrev + delin + revoke** (reclaim) | **171** |
| **borrow** = reclaim + load          | **~173** |

The temporal-safety cost is the revocation-tree machinery: **`mrev` (mint, 50 cyc)
and `revoke` (reclaim, ~120 cyc)** account for essentially the whole borrow;
`delin`, `load`, and `shrink` are each 1–2 cyc single-cycle register ops. Spatial
narrowing (`shrink`) is effectively free. The instruction-count proxy is blind to
this — it models `mrev` and `revoke` as one instruction each, hiding that both are
multi-cycle hardware operations.

## Revocation cost grows with the live tree (O(tree))

`revoke` walks the revocation tree, whose node pool the hardware never reclaims: it
is a fixed 1024-entry **bump allocator** with a monotonic allocation head (set once
at init, only ever incremented). A `drop` instruction exists — it is decoded in the
RTL and QEMU and has an LLVM builtin — but it only *invalidates* a node (clears its
valid bit); it does **not** free the node's slot, and neither does `revoke`. So no
software sequence returns a slot to the pool, and each `mrev` consumes one for the
domain call's lifetime. In a tight single-lineage loop the per-borrow cost therefore
rises with accumulated revocations:

> **borrow(N) ≈ 75 + 3·(N/2) cyc/op** — base ≈ 75 cyc, +≈3 cyc per accumulated node.

The fit is fixed by three independent points and predicts each: the standalone
sweep (182 @64, 464 @256) and the breakdown's `mrd`@small-tree vs `full`@larger-tree
offset (Δ ≈ 3 cyc/node). This is a property of the current hardware, not the
mechanism — real workloads free distinct objects, so the live tree is bounded by
heap occupancy rather than by revoking one lineage in a loop. Adding slot
reclamation to the rev-node allocator (free the slot on `drop`/`revoke`, so the head
can be reused) would flatten the growth and yield a single-lineage O(1) constant of
≈75 cyc.

### Revocation tree size

The revocation tree is a hardware-managed array in a dedicated DRAM region
(`CAP_REVNODE_MEM_BASE = 0xBFFF_C000`): **1024 nodes × 16 B = 16 KiB**, addressed by
a 10-bit allocation head; each capability carries a 30-bit node id. `mrev` bumps the
head to allocate a node; the head is monotonic (`drop` invalidates a node but does
not free its slot, and `revoke` does not either), so it only climbs and 1024 `mrev`s
overflow it. Every figure here was taken far below that ceiling:

| run | tree size |
|-----|-----------|
| **breakdown, peak** | **144 nodes (~2.25 KiB)** = 16 + 2·64 over one domain call |
| — `mrev` loop | 0 → 16 nodes |
| — `mrev`+`delin`+`revoke` loop | 16 → 80 nodes |
| — borrow loop | 80 → 144 nodes |
| **sweep point A** | 64 nodes → borrow 182 cyc |
| **sweep point B** | 256 nodes → borrow 464 cyc |

Peak occupancy was 144 of 1024 nodes (~14%), so the per-primitive numbers are the
near-empty-tree base; the two sweep points fix the O(tree) slope.

## Discussion

**The instruction-count proxy captures the shape but not the absolute cost.** Under
QEMU `-icount` the borrow is +4 retired instructions over the raw load (2 → 6); on
silicon it is ~180 cyc. The proxy models `mrev` and `revoke` as one instruction each
and is blind to the multi-cycle revocation-tree work they do. What it gets right is
the *shape* — borrow O(1) in payload, copy O(payload) — and that holds cycle-accurately.
The silicon measurement's job is to put a real constant on the borrow, which it does:
~171 cyc of temporal-safety work over a ~2-cyc access.

**Break-even is below the smallest defended object.** At a small tree the borrow is
182 cyc, and even under pathological single-lineage growth it stays ≤464 cyc — already
2–5× cheaper than copying a 256 B object (~900 cyc) and 7.7–20× cheaper at 1 KiB.
Because the borrow is payload-independent while the copy is O(size), the margin only
widens: any object large enough to be worth bounds-checking is cheaper to borrow than
to copy. The cycle-accurate data confirms the "borrow O(1) ≪ copy O(size)" claim the
instruction proxy first suggested.

**The cost is the revocation machinery, and it is bounded in practice.** `mrev`
(50 cyc) + `revoke` (~120 cyc) are essentially the whole borrow; `delin`, `load`, and
`shrink` are 1–2 cyc, so the spatial half of the model is nearly free. The O(tree)
growth appears only when one lineage is revoked in a tight loop with no node release;
real workloads free distinct objects, so the live tree tracks heap occupancy, and our
runs sat at ≤14% of the 1024-node capacity. Releasing nodes on `revoke` in the RTL
would make even the single-lineage constant O(1) at ≈75 cyc.

**Against the mechanism it competes with, the margin is orders of magnitude.** The
revoke-at-free contract point is this fixed ~171-cyc capability operation. The only
CHERI configuration that provides the same synchronous temporal guarantee — eager
revoke-on-every-free — pays a stop-the-world quarantine sweep (in the QEMU-to-QEMU
comparison, ~14 M instructions per free) rather than a fixed contract-point op. The
silicon number confirms our side of that comparison is a cheap, constant hardware
operation, not a sweep — which is the whole point of the design.

## Reproduce

`build-borrow-cost-fpga-nogp.sh` (super-operations) and
`build-borrow-breakdown-fpga-nogp.sh` (primitives) build the gp-free/cjalr-free
domains; run under QEMU with `run-domain-smoke.py` for a functional check, then on
the board (`board_run_breakdown.py`: verify bitstream, boot, UART-transfer, harvest
the `RESULT` lines). Full steps: `ref/fpga-borrow-cost-reproduction.md`.
