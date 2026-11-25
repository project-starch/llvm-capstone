# Trap-route mapping reuse: qualification on QEMU, pre-registered (2026-11-25)

The vexec route's `CAPSTONE_HEAP_MAPPING_REUSE` (alloc-lean eb73a2bf475c, 3e70344d8c3f) is ported to the trap route's
`malloc-mapping.c`. A put renews the owner in place (owner' = CRENEW(owner)) instead of the vexec route's two-operand
CREVOKE of a holder's subtree. The port follows the semantics owner's ruling (2026-11-25):
- **(b) CRENEW ends the old lifetime:** by the reference model, CRENEW ends every id of the old lifetime, the
  derived objects and the old owner reference included.
- **(a) the renewed reference authorizes munmap:** by the kernel's design (sublet-blocks-internal.h at 6d4504f8b073:
  "Requests always use the handle's slot, including after CRENEW"). Untested until this run.
- **conditions:**
  1. a failed CRENEW (untagged result, old lifetime live) is not cached; the mapping is unmapped with the old owner;
  2. the out-of-place renewal (generation exhausted) must still find the block;
  3. zeroing is unchanged from the vexec route (memset before reissue);
  4. a reprotected mapping is never reissued.

Platform:
- QEMU: qemu-crv build at da49602d76.
- Firmware: sbi-keep-run fw-y (trap-route-keep-tags eb85e21c).
- Kernel: trapk linux-6.1 (native-kernel-new-qemu 6d4504f8b073, TRAP + TRAP_TEST, no SMP), booted with
  `sublet.dom_binfmt=1`, 2 GiB.
- Each image runs as an ordinary task. A fault is the kernel's `sublet: fault cause=` line.

Images: `renew-probe.c` and `mapping-reuse-probe.c`, built against eight SDK variants of this branch.

| variant | object lifetimes | mapping reuse | test switch |
|---|---|---|---|
| off-nolife | OFF | OFF | none |
| on-nolife | OFF | ON | none |
| norevoke-nolife | OFF | ON | TEST_NO_REVOKE |
| off | ON | OFF | none |
| on | ON | ON | none |
| nozero | ON | ON | TEST_NO_ZERO |
| keepprot | ON | ON | TEST_KEEP_PROTECTED |
| live64 | ON | ON | TEST_LIVE=64 |

Compiler: clang 29027126f39b.

## Predictions (written before the boot)

Kernel facts, renew-probe against `on`:
- **unmap-renewed:** `RENEW-RENEWED tag=1 cursor_same=1 byte=0x5a`, then `RENEW-UNMAP rc=0 ... mapped=0`; exit 0.
- **unmap-stale:** the old reference's munmap is refused (rc nonzero, or a fault line), with `mapped=1`. If it
  returns, the renewed reference's munmap must give rc=0 and mapped=0.
  - **Falsifier:** `RENEW-STALE-UNMAP rc=0`, which means a stale reference authorized munmap.
- **read-stale:** a fault line (cause 25, stale) after `RENEW-STALE-READ-BEGIN`; never `RETURNED`.
- **many 40000 and many 200000:** every renewal tagged (`renewals=N of N`), then `RENEW-UNMAP rc=0 mapped=0`.
  - A renewal that comes back untagged ends the loop with its index printed. That reading is condition 1's
    trigger, not a failure of the port.
  - An rc != 0 or mapped=1 after many renewals falsifies (a) or condition 2.

Safety, mapping-reuse-probe:
- **stale200 and stale16, against off-nolife:** a fault (the mapping was unmapped; cause 25 or a page fault).
- **stale200 and stale16, against on-nolife:**
  - puts > 0 and hits > 0, and same200 > 0 or same16 > 0, so the bytes really were reissued;
  - then a fault with cause 25. Only the put's CRENEW can end a lifetime here, because lifetimes are off.
- **stale200, against norevoke-nolife (the positive control):** `UAF-STALE-READ-RETURNED stale200 0x33`. This
  shows the stale read lands on reissued bytes, so the fault in the on-nolife row is the renewal's doing.
- **reuse:**
  - on: `UAF-REUSE-OK same_address=1 hit=1`;
  - nozero: `UAF-FAIL reuse data of the previous block is visible`, or the capability line;
  - off: `UAF-REUSE-OK` with hit=0.
- **alias, against on:** a fault (cause 25) after `UAF-ALIAS-READ-BEGIN same_address=1`; the sibling group intact.
- **protect:**
  - on: `UAF-PROTECT-OK`;
  - keepprot: a fault (cause 15) after `UAF-PROTECT-WRITE-BEGIN same_address=1`.
  - If the trap kernel refuses mprotect through a heap object's reference (`UAF-FAIL protect mprotect`), condition 4
    is untested on this route and is recorded as a residual.
- **saturate:** on: `UAF-SATURATE-OK` with untracked=0; live64: `UAF-SATURATE-OK` with untracked > 0.
- **realloc, against on:** `UAF-REALLOC-OK`.
- **pressure, against on and off:** `UAF-PRESSURE`. If it stops on enomem, on reads evicted > 0 and a block count no
  lower than off's. If it stops at the cap, the eviction path is not exercised (a residual).

Each probe line comes with its exit status and any fault line, and the run prints `renew_failed` with every set of
counters.
