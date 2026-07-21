# Cycle-accurate borrow cost on Capstone silicon (Genesys 2 / CVA6)

**Vehicle:** CapliFive CVA6 "Capstone" core, bitstream
`working-caplifive-captype-fixed.bit`, OpenSBI Capstone monitor, image
`fw_payload_fpga_up_ctl.bin`. **Timer:** `mcycle` read inside a Capstone domain.
**Method:** each operation is an `N`-iteration inner loop bracketed by two `mcycle`
reads with an empty calibration loop subtracted; figures are cycles per operation.
The measured loop bodies are byte-identical to the QEMU `borrow-cost` probe. This is
the first Capstone temporal-safety measurement to run on hardware.

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

`revoke` walks the revocation tree, which the hardware never prunes (each `mrev`
leaves a node for the domain call's lifetime, and the core exposes no prune
instruction). So in a tight single-lineage loop the per-borrow cost rises with
accumulated revocations:

> **borrow(N) ≈ 75 + 3·(N/2) cyc/op** — base ≈ 75 cyc, +≈3 cyc per accumulated node.

The fit is fixed by three independent points and predicts each: the standalone
sweep (182 @64, 464 @256) and the breakdown's `mrd`@small-tree vs `full`@larger-tree
offset (Δ ≈ 3 cyc/node). This is a property of the current hardware, not the
mechanism — real workloads free distinct objects, so the live tree is bounded by
heap occupancy rather than by revoking one lineage in a loop. Bounding it in
hardware (release nodes on `revoke`, or add a prune op) would flatten the growth and
yield a single-lineage O(1) constant of ≈75 cyc.

## Reproduce

`build-borrow-cost-fpga-nogp.sh` (super-operations) and
`build-borrow-breakdown-fpga-nogp.sh` (primitives) build the gp-free/cjalr-free
domains; run under QEMU with `run-domain-smoke.py` for a functional check, then on
the board (`board_run_breakdown.py`: verify bitstream, boot, UART-transfer, harvest
the `RESULT` lines). Full steps: `ref/fpga-borrow-cost-reproduction.md`.
