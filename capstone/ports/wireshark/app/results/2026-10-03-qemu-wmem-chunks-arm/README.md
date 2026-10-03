# The wmem chunks arm on fixtures 10-13: wireshark's nested arm, measured at last (2026-10-03)

**Question.** `chunks` is wireshark's **nested-allocator arm** — the sublet arm with one difference,
that wmem's BLOCK allocator is the wmem port's chunk port, so *"every chunk is a region of its own and
a chunk free, a scope reset and a block's end are revokes"* (`host/build-domain.sh`). Its predictions
for the wmem fixtures were registered on **2026-09-23** and, until this run, **no bundle in this port
contained a `chunks` section for fixtures 10-13**. Does the arm discriminate where the heap arms do not?

**Pre-registration.** The rows are in `host/safety-expect.txt` from 2026-09-23, untouched since. They
are the nested-discriminating shape: `level0`/`shrink`/`sublet` RETURN on 10-12, and `chunks` alone
faults on 11, 12 and 13.

## Verdict

**All four `chunks` cells exactly as pre-registered — three faults, each in a cell where every
heap arm that is registered for it returns.** One of the three needed a second boot with a longer
timeout.

Precisely: for fx11 and fx12 all three heap arms are registered and were measured returning
(2026-09-25). For fx13 only `sublet` is registered, and it returned here (`d00020`), measured today;
`level0` and `shrink` carry no fx13 row, so that cell's contrast rests on one heap arm, not three.

| arm | fx10 — BLOCK_FAST reset | fx11 — BLOCK reset | fx12 — BLOCK neighbour | fx13 — BLOCK chunk free |
|---|---|---|---|---|
| `level0` | RETURN `a0015b` | RETURN `b0015b` | RETURN `c000ee` | not registered |
| `shrink` | RETURN `a0015b` | RETURN `b0015b` | RETURN `c000ee` | not registered |
| `sublet` | RETURN `a0015b` | RETURN `b0015b` | RETURN `c000ee` | **RETURN `d00020`** |
| **`chunks`** | RETURN `a0015b` | **FAULT `temporal`** | **FAULT `oob`** | **FAULT `temporal`** |

The three heap-arm rows are from the committed bundles of 2026-09-25 and were not re-run here. The
`chunks` row and `sublet` fx13 are new.

## The nested adapter is visible in the capability — and it is selective

This is the part worth keeping. On `chunks`, a wmem **BLOCK** chunk is bounded to itself:

    chunks  fx12  p  cursor=a5100030  bounds=[a5100030,a5100070)  len-from-cursor=64
    chunks  fx12  q  cursor=a5100080  bounds=[a5100080,a51000c0)  len-from-cursor=64
    chunks  fx13  p  cursor=a5100030  bounds=[a5100030,a5100070)  len-from-cursor=64

while on `sublet` the same allocation carries the **whole 1 MiB block** (`len=1048528`) — and on
`chunks`, fixture 10 *also* carries the whole block:

    chunks  fx10  p  cursor=a5100030  bounds=[a5100000,a5200000)  len-from-cursor=1048528

because fixture 10 is **BLOCK_FAST**, which the chunk port deliberately leaves alone. So **within one
arm** the adapter narrows BLOCK chunks and not BLOCK_FAST. That is a two-sided check on what the port
claims to do, read off the capability rather than argued: if the adapter were doing nothing, fx12/13
would show the block bounds too; if it were doing too much, fx10 would not.

## Attribution

- **fx12 is a WRITE** through `p` at `q`'s address. The instruction in the diagnostic is `00c50023`
  (`sb`), and the OOB report names bounds ending at `a5100070` against a target of `a5100080`. A
  spatial fault — and **it is the per-chunk bound that makes it one**: the same write on `sublet`
  completes, because there the bound is the whole block and `q` is inside it.
- **fx13 is a READ** through a chunk released with `wmem_free` while its block lives on. The
  diagnostic's untagged value equals the fixture's printed target, `a5100030` — the temporal rule,
  which matters because without the value comparison a null dereference reads the same.
- **Neither section contains a `mark=` or `returned` line after its touch** (checked: count 0).

## Fixture 11 needed two boots, and the first one is worth keeping in the record

`fx11` is the BLOCK-reset case (`wmem_free_all`), where on this arm a scope reset must revoke **every
chunk in the block at once**. In the first boot it **began and ran** — 36 gp-fabrication lines after
`FIXTURE 11 BEGIN`, the counter advancing `#1561000` → `#1596000` with the pc moving `a021e000` →
`a04f9b4c`, i.e. **progressing, not spinning** — and the driver's expect timeout cut the boot
mid-stream. `out-11.txt` and `err-11.txt` were both 0 bytes, no fault record, and no capability
diagnostic anywhere in that boot. **Fixture 10 returned normally in the same boot**, so neither the
boot nor the arm is at fault.

**Re-run alone with `--timeout-multiplier 12.0`, it faulted as registered**: every line printed, and
the untagged value `a5100030` equals the printed target, at `pc=a02173ec` — the same
`tsapp_fix_touch+0x14` as fx13. So the first boot was **cut short, not wedged**.

Its capability evidence is the cleanest of the four:

    chunks  fx11  p  cursor=a5100030  bounds=[a5100030,a5100070)  len-from-cursor=64
    chunks  fx11  q  cursor=a5100030  bounds=[a5100030,a5100070)  len-from-cursor=64   same-address=1

`q` is the **same chunk reissued** after `wmem_free_all` — same cursor *and* same per-chunk bounds — so
the reset recycled that exact chunk and the stale pointer names it.

The first boot's TIMEOUT stays in `result-lines.txt` for two reasons: it is why the cell cost two
boots, and **"progressing but cut off" and "wedged" look identical in a single run** — what separated
them was that fixture 10 returned normally in the same boot and the pc kept advancing. The cause is
the node budget rather than anything about the defect: a BLOCK reset revokes every chunk at once,
against a 65,536-node per-boot pool, and fx11 is the only one of the four that needed the long
multiplier.

## What this does and does not say

- **These are nested-allocator cells, and `chunks` is the only arm that catches them.** That is the
  shape the paper's security table needs: the control completes and the arm that hooks the inner
  allocator faults.
- **fx12 is spatial, fx13 is temporal.** Only fx13 is a lifetime catch; fx12 is a bounds catch that
  exists *because* the adapter bounds each chunk. Counted separately, not merged.
- **These are synthetic fixtures, not upstream defects.** They exercise the wmem allocator unreduced
  but their consumers are probes. The 13 upstream defects in `bug-corpora/wireshark/wmem-repros` are a
  separate corpus with its own arms.
- **No native/ASan arm was run here.** For fx13 ASan would be blind by construction — the chunk's
  release never reaches `g_free` — but that is asserted from the allocator's structure, not measured in
  this bundle.
- **QEMU only. N = 1 per cell.**
- The run used the assembled pinned platform documented in
  [`../2026-10-03-qemu-upstream-live-defects/`](../2026-10-03-qemu-upstream-live-defects/README.md),
  `cma=1536M`.

Files: `result-lines.txt` (every line above).
