# Two upstream FFmpeg defects on the capability machine: level0, shrink, sublet (2026-10-03)

**Question.** Fixtures 24 and 25 reproduce two defects that the triage found **live at the 9.0.1
pin**. Does the revoking heap catch them, and do the unprotected arms let them through?

**Pre-registration.** Both fixtures and all six predictions were pushed in `95aa3d340f80`, before
either image was built. Nothing in that file has been edited since, including the one cell that
differs.

## Verdict

**Five of six cells exactly as pre-registered; the sixth differs in a predicted value for a reason
entirely in the derivation, with the mechanism conclusion unchanged.**

| arm | fixture 24 — `vvc/thread` | fixture 25 — `ops_dispatch`, interior pointer |
|---|---|---|
| `level0` | RETURN `180015b` — **as predicted** | RETURN `190017b` — predicted `190015b` |
| `shrink` | RETURN `180015b` — **as predicted** | RETURN `190017b` — predicted `190015b` |
| `sublet` | **FAULT temporal, cause 24** — as predicted | **FAULT temporal, cause 24** — as predicted |

**The faults are attributed, not merely observed.** Each fixture published the address it was about
to touch, and QEMU's diagnostic names the same value:

    FFAPP-FIX 24 target=c4802000   <->  cincoffset with an UNTAGGED rs1 ... val=0xc4802000
    FFAPP-FIX 25 target=c4802020   <->  cincoffset with an UNTAGGED rs1 ... val=0xc4802020

Both faults follow the fixture's own `touch` line with no `returned` line after it. That is the
oracle `safety-verdict.py` applies: an untagged-operand fault counts as temporal **only** when the
untagged value is the fixture's printed target, because without that a null dereference would read
the same.

**The arms differ in the one thing they are supposed to.** The capability the fixture holds:

    level0   bounds=[c0253220,c03d3220)  len-from-cursor=1522896   the whole arena
    sublet   bounds=[c4802000,c4802040)  len-from-cursor=64        the object alone

and on `sublet` that alias is dead at the touch, while on `level0` and `shrink` the read completes
and returns the **new owner's** byte with `same-address=1` — the storage really was freed and
reissued.

## The differing cell, and why it is not a mechanism finding

Fixture 25 returned `0x7b` where the prediction said `0x5b`. The cause is in the prediction's
arithmetic, not in the machine: `fill(p, v0, n)` writes **`v0 + i`**, an incrementing pattern, and
fixture 25 reads at **offset 32** of the reissued 64-byte object, so the new owner's byte there is
`0x5b + 32 = 0x7b`. Fixture 24 reads at offset 0 and matched exactly, which is the control that
isolates the error to the offset rather than to the fill or the reuse.

The prediction file keeps the value it was given. What the cell establishes is unchanged and is what
the fixture was for: the stale **interior** pointer read the new owner's data at its own offset.

**And the registered hypothesis about interior pointers held.** The expect file predicted that
fixture 25's interior pointer "changes nothing, because revocation is per object and not per alias,
and a disagreement between 24 and 25 is itself the result". There is no disagreement: both fault on
`sublet`, both complete on the unprotected arms, and 25's capability is the same object's bounds
with the cursor 32 bytes in.

## What this does and does not say

- **It is the CONTROL half, not a detection claim.** Both defects are plain `malloc`/`free`
  use-after-free, which ASan also reports — measured the same day in
  `../2026-10-02-native-upstream-defects/`, where both buggy arms report `heap-use-after-free` and
  neither fixed arm does. The value here is that the 2×2 is now measured on *real upstream defects*
  rather than on shapes we invented, in both rows.
- **Nothing about nested allocators.** Neither defect is in FFmpeg's pools.
- **QEMU only.** The temporal faults are the emulator untagging a revoked capability on reload
  (Q-11); the deployed silicon lets such an access retire.
- **N = 1 per cell.** Six cells, one boot per arm, two fixtures per boot.

## The platform these ran on, which is not the default one

The images are application-SDK images and need the delegated runtime's **process ABI** — SBI
functions `0x21`–`0x2b`. The buildroot platform installed on this host provides none of them, which
is recorded in
[`docs/history/02-10-2026_20-30-00_application-images-blocked-on-monitor-process-abi.md`](../../../../docs/history/02-10-2026_20-30-00_application-images-blocked-on-monitor-process-abi.md).
That document said the implementation was nowhere on this machine. **That was wrong, and this run is
the correction:** it exists, built, in the delegation lane's tree, and the pieces are a matched pair
built within a minute of each other on 2026-10-01.

| piece | what was used | why not the default |
|---|---|---|
| monitor | `deleg-gate2/opensbi-T/.../fw_jump.elf`, 41 `context_step` symbols | the installed `fw_jump.elf` has **0**; it dispatches 15 capstone SBI functions and no `PROCESS_*` |
| QEMU | `deleg-gate2/qemu-12/build/qemu-system-riscv64`, carries `capstone_supervisor.c` | with the stock QEMU the monitor itself takes an illegal-instruction fault at `pc=0x80020b64`: the process path is behind `CAPSTONE_SUPERVISED_CALL` |
| kernel module | built out-of-tree from buildroot `8fd1ea1249a9` | the installed `capstone.ko` has **zero** module parameters; this one has `parm=process_cache_bytes` |
| launcher | `capstone-exec`, `capstone-job` built against the pinned `libcapstone.c` | absent from the rootfs entirely |
| compiler | `7d01722aab88`, Release + assertions | `dev`'s compiler has no intcap extensions, which the application SDK requires |
| kernel, rootfs | the installed ones, unchanged | — |

The run goes through `tests/runtime-qemu/run-domain-smoke.py` over the serial console rather than
`capstone-vm`, because the rootfs has no dropbear. The guest-side command is the one `capstone-vm`
issues: `capstone-job … -- capstone-exec -- <image>`.

**The weak point of this platform, stated plainly:** it is assembled from one lane's build trees,
not from this branch's submodule pins, and the pins disagree with each other — buildroot is checked
out at `d04bd83b13cd` against a pin of `8fd1ea1249a9`, QEMU at `deb7d757` against `674cdab03c3e`.
A result from an assembled platform is worth less than one from a pinned one. What makes these six
cells usable anyway is that the *images* are the pinned branch's, built by its own scripts, and the
oracle checks the fault against an address the image itself published.

Files: `result-lines.txt` (all six cells), and the staged platform at
`/tmp/capstone/pinned-platform/` with the per-arm run directories.
