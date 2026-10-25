# httpd (APR pools + apr-util buckets) on the virtual profile: two arms, pre-registered

Pre-registered 2026-10-26 on lane `compiler/httpd-virtual`, before any adapter code, build or run.
Taken from the lead's "resume all experiments and all work" (relayed by the runtime lane, queue item 2).

## What is built

Both corpora, `bug-corpora/httpd/apr-pool-repros` (1 case) and `bug-corpora/httpd/bucket-repros` (8 cases), on
the pools port `ports/apr/pools`, preset `capstone-application` (new), virtual SDK r60
(`t3-freeze-nb-nomovc-cw-r60/sdk`). Upstream keeps every allocation decision in both arms: the arms differ only in
the adapter behind the existing seam (patches `apr-1.7.4-0002` and `apr-util-1.6.3-0002`, unchanged).

| arm | build | adapter |
|---|---|---|
| `virtual-malloc` | `APRP_SUBLET=OFF` | the stock CheriBSD adapter, unchanged: `src/cheribsd/node-malloc.c` (every node its own `malloc`, freed only at `apr_allocator_destroy`) and `src/native/bucket-pointers.c` (pieces are plain pointers into their block). APR as shipped, over virtual mallocng. |
| `virtual-nested-pools` | `APRP_SUBLET=ON` | new `src/virtual/node-derive.c` and `src/virtual/bucket-derive.c` (memcached patch 0006 as the template) |

The nested adapter. Node level:
- backing: `malloc` gives the node's memory. Its owner is a `CDERIVE` child of that memory covering the node, and
  carries MANAGE. The free list holds the owner.
- `aprp_node_issue`: a fresh `CDERIVE` child of the owner, covering the node, is what the pool gets.
- `aprp_node_release`: `CREVOKE(owner, issued child)`, so every handle into the node dies, and the owner goes back to
  upstream for the free-list link.
- `aprp_node_discard` frees the memory.

Bucket level:
- `aprp_node_lend` keeps the issued child as the block.
- A carve or a reissue is a `CDERIVE` child of the block.
- `apr_bucket_free`'s file is a `CREVOKE(block, piece)`.
- When the block chain goes back to APR, the node release revokes the block's subtree, so every piece goes with it.

This is the granularity of the domain `sublet` arm (`src/allocators/sublet/`).

Two seams: the driver's `aprp_set_mode(0)` becomes `aprp_set_mode(APRP_DRIVER_MODE)`, defaulting to 0, and the nested
build compiles the cases with `-DAPRP_DRIVER_MODE=1`. The nested adapter refuses any other mode (exit 75), so a
mis-paired build cannot run unprotected and read as a result.

## Runner, kit and controls

- **Runner and kit:** `bug-corpora/tools/run-virtual-cases.py --prebuilt` (`format=case-line`, `buggy|fixed N`), on
  qkit7-b2twin3-q8, the bitstream-2 twin all other bitstream-2 references used.
- **Configurations** (`tools/arms.json`): `virtual-apr-stock` and `virtual-apr-sublet`.
- **The heap's own controls**, `bounds-malloc` and `uaf-malloc`, fault on both arms.
- **apr-pool-repros port control:** `controls/virtual/90_ctl_node_released` reads a palloc'd byte after its pool is
  destroyed. Expected COMPLETE on `virtual-malloc` (the node goes on APR's free list and never reaches free()) and
  FAULT at `read_probe` on `virtual-nested-pools` (the node's release).
- **bucket-repros port control:** `controls/virtual/90_ctl_piece_filed` reads a bucket piece after `apr_bucket_free`.
  It exercises the per-piece revocation that no corpus case does. Expected COMPLETE / FAULT at `read_probe` in the same
  way.
- **If a control behaves otherwise**, the run reads NO-READING. Its verdicts are not reported.

## Predicted readings, per case

`virtual-malloc`: **all 9 MISSED** (VERDICT DEFECT-REPRODUCED, exit 0; the fixed run of the same image FIXED, exit 0).
- No case calls `apr_allocator_destroy`, and under `APR_ALLOCATOR_MAX_FREE_UNLIMITED` a destroyed pool's node, or a
  bucket allocator's returned block, goes on APR's own free list. The heap never sees a free.
- This is the same reason stock CheriBSD with libc revocation reads 9/9 MISSED (`results/20260921-cheribsd`,
  `results/20260922-cheribsd`).

`virtual-nested-pools`: **all 9 CAUGHT at `read_probe`**, with the fixed run completing.
- **What it should read:** cause 25, a revoked operand, which is what memcached's nested arm reads on this profile.
  That is the value read, not the verdict; the verdict is the probe attribution.
- **apr-pool case 0, and bucket cases 2, 3, 5 and 7:** a pool destroy releases the node, so the issued child dies.
- **Bucket cases 0, 1 and 6:** a bucket allocator destroy returns the block, and the node release revokes the block's
  subtree.
- **Bucket case 4:** the declared handback destroys and recreates the allocator, as in every other arm.
- This matches the domain `sublet` arm case for case (9/9 at the labelled probe).

**What would count against the design:**
- a fault outside `read_probe`, such as one inside APR's own bookkeeping through an owner or block that should still
  be live;
- a fixed run that fails;
- any nested-arm case that completes.

Each is reported as found, not explained away.

## Changes found at build time, before any run

- **The stock bucket adapter returned an integer cast back to a pointer** (`return (void *)p->address`, twice). It
  works natively, and on purecap CheriBSD, where `uintptr_t` is a capability. On Capstone the result is untagged, as
  the compiler's own `-Wcapstone-pointer-roundtrip` warning said, so the stock arm would have faulted inside its own
  bookkeeping. The fix: a piece's pointer is now the block header's pointer plus the piece's offset. It is the same
  address on every target, so the native and CheriBSD arms do not move.
- **The configurations are four, not two:** `virtual-apr{pool,bucket}-{stock,sublet}`. The runner runs every control a
  configuration lists, and each corpus carries only its own port control.

## Read, 2026-10-26: every pre-registered reading holds

**The run:** one boot per corpus and arm on qkit7-b2twin3-q8, QEMU domain A.
- Kit: QEMU 708bbff5663f0e86, module 3325473537072a3a, vexec fad465ce4dea7121.
- Runner: e464b106e1eb5c8e.
- Images: built from this lane's 543ee08b9cfa sources (the run's stamp names the pre-registration commit with the
  same files uncommitted).

| corpus | arm | controls | cases |
|---|---|---|---|
| apr-pool-repros | virtual-malloc | bounds-malloc, uaf-malloc fault; node-released-90 completes | 0: MISSED |
| apr-pool-repros | virtual-nested-pools | bounds-malloc, uaf-malloc, node-released-90 fault | 0: CAUGHT, cause 25 at read_probe (probe) |
| bucket-repros | virtual-malloc | bounds-malloc, uaf-malloc fault; piece-filed-90 completes | 0-7: MISSED |
| bucket-repros | virtual-nested-pools | bounds-malloc, uaf-malloc, piece-filed-90 fault | 0-7: CAUGHT, cause 25 at read_probe (probe) |

Every fixed run printed VERDICT FIXED and exited 0. Nothing faulted outside the probe. The bundles are under each
corpus's `results/2026-10-26-virtual/`, and the case.json cells are derived from them (`tools/derive-verdicts.py`).
