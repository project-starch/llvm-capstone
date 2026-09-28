# The wmem chunk port inside real tshark (step 2) — pre-registration

Written and pushed **before the first boot of the chunks arm**. Step 1 (`PREREGISTRATION-chunk-port.md`
and its v2 addendum) ran the chunk port in the replay harness: synthetic traces and directed
fixtures, no packet dissected. Its README names this step as the one that makes the port more than a
model: the same ported allocator, compiled into the tshark app port and dissecting real captures.

## The arm

`ports/wireshark/app/host/build-domain.sh TSAPP_HEAP=chunks`. It is the sublet arm with **one**
difference: `wmem_allocator_block.c` is the chunk port's. That is upstream plus this port's patches
0001 and 0002, exactly the source the harness builds (0001 touches only the four allocator `.c`
files, and tshark's copy of the file is upstream's), plus patch 0007's one macro edit re-anchored.
Beneath it:

- the port's own `src/allocators/sublet/chunks.c`, **unchanged**, which carves, takes, gives and
  revokes;
- `app/src/tsapp-wmem-chunks.c`, the level below, replacing the harness's `backing.c`. Blocks come
  LINEAR from the Sublet heap (`__capstone_sublet_malloc_linear`, as FFmpeg's pools take theirs),
  and records and the index are heap objects.

BLOCK_FAST, the packet scope, is the sublet arm's, unchanged, as step 1 found it needs nothing. The
sublet arm is therefore the matched control.

## Predictions

**T1 — build.** `build-domain.sh` passes its link gates and its negative control on the chunks arm.
The control names `__capstone_region`, as on sublet. The ported `wmem_allocator_block.c` it compiles
differs from the harness's `source-ported` copy only by the block-size macro (`diff` shows exactly
that edit).

**T2 — fixtures, N = 1 per cell** (`host/safety-expect.txt`, the `chunks` rows and `sublet 13`):

| fixture | sublet (control) | chunks |
|---|---|---|
| 1-9 (no wmem) | as registered | the same verdicts |
| 10 BLOCK_FAST reset | RETURN `a0015b` | RETURN `a0015b`: not ported |
| 11 BLOCK reset | RETURN `b0015b` | **FAULT temporal** |
| 12 BLOCK neighbour write | RETURN `c000ee` | **FAULT oob** at the printed target |
| 13 BLOCK chunk free, block live | RETURN (value not predicted) | **FAULT temporal** |

Fixture 13 is new. It is the one lifetime the chunk port adds over every other arm of this app.

**T3 — the dissection is unchanged.** M5 on the captures the sublet arm ran (dhcp, dns_port, http,
arp, dns-ooo, ntp, and the four flips) gives the sublet arm's verdicts: stdout MATCH except `ntp`
(the known timezone gap, DIFFERS), stderr MATCH throughout, and every flip fires. M1-M5 on dhcp are
REACHED and MATCH.

**T4 — the counters obey the chunk port's structure** on every full run. The TSAPP-HEAP line's
`wmem ...` fields are present. The chunk unit's `revokes` equals `reset_revokes + close_revokes +
retires`. `reset_revokes` equals `resets`, and `close_revokes` equals `closes`.

**T5 — the node budget, decided now so it cannot be tuned to the result.** The first chunks boot
runs one capture, dhcp, alone. Its spend S is the heap's `split + mrev` plus the chunk layer's
`nodes=`. Every later oracle boot holds at most `floor(60000 / (1.5 × S_max))` full runs, where
S_max is the largest spend measured so far, and the runner enforces it
(`TSAPP_CHUNKS_RUNS_PER_BOOT`). A boot that dies on the node pool's assertion is recorded as a
budget failure, never as a verdict.

**T6 — the cost of not coalescing: reported, not predicted.** `opens` (1 MiB BLOCK blocks) and the
heap's `peak_live` are reported per capture, beside the sublet arm's `peak_live`. The sublet arm
counts no wmem blocks, so an exact comparison with upstream's block count is not available here, and
this is said rather than approximated. What is predicted is only that no capture exhausts the 16 MiB
heap pool. A capture that does would fail T3, and that failure is the cost of not coalescing.

## What would refute it

- **T2:** 11, 12 or 13 returning on the chunks arm; or any of 1-10 changing.
- **T3:** any stdout or stderr verdict other than the sublet arm's.
- **T4:** revokes that the structure does not account for.

Any of these is a result, and is reported as one.
