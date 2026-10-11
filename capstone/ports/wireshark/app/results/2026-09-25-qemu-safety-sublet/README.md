# tshark safety on QEMU: the sublet heap arm (2026-09-25)

**Question.** The level0 and shrink arms (`../2026-09-25-qemu-safety/`) showed that neither catches
a use after free. What does tshark do on a heap that revokes on every free, and does it still
dissect correctly there?

The sublet arm and its predictions (appended to `host/safety-expect.txt`, earlier lines untouched)
were pushed in `cd06fd2` at 09:10, before any sublet image booted.

## Verdict

**tshark runs on the revoking heap, and every counted fixture run is as pre-registered: 36 of
36**, that is 12 fixtures × 3 repeats. Each cell's three runs are distinct boots of one image.
The bytes each boot ran hash to the arm's `SHA256SUMS`, and every boot used the one host binary.

| fixture | sublet |
|---|---|
| 1 g_malloc(64): the pointer's length | returns; exactly 64 |
| 2 write into the neighbouring object | bounds fault at the neighbour's first byte |
| 3 read one byte past the end | bounds fault at `p + 64` |
| 4 read after g_free | **temporal fault**: the freed pointer reloads untagged, and the fault names the freed object's own address |
| 5 read after g_free and reuse | **temporal fault** (the reuse happened: `same-address=1` in every run) |
| 6 stale g_free | **temporal fault**, inside free's own probe read (`lbu zero, 0(a0)` at `free`+0x48), before anything is revoked (`q-took-p's-address=1` in every run) |
| 7, 8 one byte past a global, a stack array | bounds fault |
| 9 a second global through the first | returns: the two share one capability (the compiler's merge) |
| 10 stale pointer after a BLOCK_FAST reset | **returns** the new scope's byte; the pointer carries its 1 MiB block (`bounds=[e1100000,e1200000)`) |
| 11 stale pointer after a BLOCK reset | **returns**, the same |
| 12 write into the next wmem allocation | **returns** `0xee`; the neighbour was overwritten |

What it says:
- **A g_malloc'd object is bounded and revoked.** Use after free, use after reuse and a stale free
  all fault. Both cheap arms returned in each case: the freed object's own byte (4), the new
  occupant's (5), and for the stale free a later allocation aliasing the live object (6). This is
  measured for 64-byte objects on capstone-qemu. `sublet_heap.c`'s `sh_narrow` rounds an object
  of 4 KiB or more up to its representable granule.
- **wmem is still out of reach.** A wmem allocation carries its whole 1 MiB block (jumbo
  allocations have their own g_malloc). A scope reset does free some blocks: BLOCK_FAST frees all
  but its first block and every jumbo, and BLOCK frees every jumbo. But it keeps, and rewinds or
  re-initialises, the block a fixture's allocation sits in, so that block is never revoked. So
  10–12 return here exactly as on level0 and shrink. That is the gap the wmem hooks
  (`ports/wireshark/wmem`) would close, and the remaining safety gap for tshark on this heap.
- **tshark on the sublet heap matches stock.**
  - M1–M5 all reach their stage and MATCH.
  - M5's `-V -n` stdout is byte-identical to stock on dhcp, dns_port, http, arp, their flipped
    copies (each flip fires) and dns-ooo. Its stderr is identical to the native minimal build's.
  - ntp's stdout differs from stock, as the negative control must.

## The heap and its revocation-node spend

One full M5 run, from its `TSAPP-HEAP` line:

| capture | allocations | frees (each revokes) | peak live objects | split | mrev | nodes (split + mrev) |
|---|---:|---:|---:|---:|---:|---:|
| dhcp | 4,692 | 4,552 | 2,598 | 3,953 | 8,643 | 12,596 |
| dns_port | 4,971 | 4,831 | 2,774 | 4,183 | 9,152 | 13,335 |
| dns-ooo | 4,637 | 4,497 | 2,588 | 3,813 | 8,448 | 12,261 |
| http | 3,900 | 3,760 | 2,608 | 3,249 | 7,147 | 10,396 |
| arp | 3,715 | 3,575 | 2,444 | 3,057 | 6,770 | 9,827 |
| ntp | 3,811 | 3,671 | 2,526 | 3,123 | 6,932 | 10,055 |

A flipped capture spends exactly what its original does. That so few allocations reach the heap
suggests wmem serves most of the dissection from its blocks, but that was not measured.

**capstone-qemu ran out of revocation nodes in one boot of six runs.**
- **capstone-qemu reclaims no revocation node.** A node's refcount is set to 1 and never changed.
  Nothing calls `cap_rev_tree_release`: its only caller, `cap_rev_tree_update_refcount`, is itself
  never called. So its 65,536-node pool is a cumulative per-boot budget. `cap_rev_tree.h` says
  so: "this emulator reuses no node". The free-list comment at `cap_rev_tree.c:8-10` describes
  dead code; a draft of this README repeated it until an audit caught it.
- **What happened.** Oracle A's first five runs completed. Their heaps spent 64,123 nodes (split +
  mrev): dhcp and dhcp.flip 12,596 each, dns_port and dns_port.flip 13,335 each, dns-ooo 12,261.
  Everything else up to the watermark spent 1,409: the genesis nodes, firmware, kernel and host,
  six domain setups, and ntp's first heap allocations. ntp then crossed #65,532, and QEMU
  asserted at the end of the pool: `cap_rev_tree.c:56: _cap_rev_tree_dup_node_before: Assertion
  new_node != CAP_REV_NODE_ID_NULL`.
- **What the runner does now.** Five of these runs fit in a boot, but five the size of dns_port
  (66,675) would not, so it allows at most four full runs per sublet boot. ntp was rerun alone and
  gave the expected result.
- **On a bitstream without the node reclaimer** (ISSUES R-12), the same budget is about five dhcp
  runs per boot, before the monitor's and the host's own node use.

## Method

- **The arm** (`host/build-domain.sh TSAPP_HEAP=sublet`):
  - `runtime/sublet_heap.c` over a 16 MiB pool, carved from a 32 MiB LINEAR region the host
    transfers (`libc_test_host.c`, `LT_HEAP_REGION_BYTES`);
  - `hostcall.c` rebuilt with `CAPSTONE_PROGRAM_REGIONS`;
  - wmem's block allocators rebuilt with 1 MiB blocks (patch 0007). The linked code carries the
    1 MiB constant, where level0's carries 2 MiB.
  The image is 28.9 MiB in a 32 MiB block.
- **Fixtures, verdict and boots** as in the cheap arms' folder. Per repeat, seven boots:
  `1 9 10 11 12 2`, then 3, 4, 5, 6, 7 and 8 alone, one predicted fault each, last.
- **The compiler** is the port's, `b7b31421e9fa`.

## Runner changes, each shown able to fire

- **The node-budget refusal.** A five-capture sublet oracle is refused with exit 2 before any
  boot.
- **A host-side wedge detector.** If the serial log has not grown for `TSAPP_WEDGE_SECONDS`
  (default 360), the runner stops this boot's QEMU, found by its own share path. Twice today a
  guest stopped outright in setup and held the lock for the whole budget.
  - **The first two controls never created the condition.** With 20 s and then 5 s thresholds on
    a normal boot, nothing fired: the detector polls every 15 s, and those two boots were not
    quiet that long at any of their polls. A clean result from a control that never created its
    condition says nothing.
  - **The third control created it.** A test-only `TSAPP_TEST_QUIET=120` holds the guest silent
    for 120 s. With a 40 s threshold the detector fired at 44 s, stopped QEMU, and the boot ended
    (`qemu-oracle-20260925-093329-p6Xk`).

## What this does not establish

1. **The board.** On silicon a stale data access retires (ISSUES Q-11). There free's probe read
   would not stop a stale free, which would then revoke the new owner's handle
   (`sublet_heap.c`). The temporal results here are capstone-qemu's.
2. **wmem.** 10–12 return: the arm leaves tshark's main allocator unprotected.

The verdict's `--self-test` gained three temporal cases (`instrument-checks.txt`): a temporal
fault at the target, one on a different value, and an untagged operand at the target.

## Files

- `result-lines.txt`: the gate's stage and oracle lines, then every fixture boot's verdict in
  order, with stalled attempts, and a per-cell count.
- `SHA256SUMS`: the sublet images and the host binary the fixture boots ran.
- `stall-classes.txt`: every boot of the campaign, by where it stopped. 25 of 28 reached a domain
  (2 after a slow setup); 1 wedged in setup before the wedge detector existed; 2 stopped before
  login and were ended by the login timeout.
- `instrument-checks.txt`: the verdict's self-test, with its temporal cases.
