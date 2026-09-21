# The httpd/APR case on stock CheriBSD, revocation on and off, QEMU, 2026-09-21

**The case completes under libc revocation, and the control beside it faults.**
Stock CheriBSD — the platform's own `malloc` under every node, its own
kernel-defaulted revocation on and verified in the guest — does not see a
consumer allocating through a pool it destroyed. The same run's control frees
a block, sweeps, reads through the old pointer at the same labelled load, and
faults exactly there. The mechanism is active; it is never asked.

    matrix.tsv    two result lines, one per revocation setting, each with its
                  control's outcome beside it
    inputs.json   binaries, platform fingerprint, ABI-control line, fixture
                  hash per run; the negative control's counts

`matrix.tsv`, not the serial logs.

## What was run

One `defect-00` built through the port's `cheribsd` preset against
`src/cheribsd/node-malloc.c`: nodes come from `malloc`, go back through `free`,
exactly as upstream APR does. No adapter authority, mode 0 only. What varies is
the per-process libc revocation policy, which the common runner sets
explicitly and the guest's ABI control verifies through
`malloc_revoke_enabled()`. The kernel default stayed on for the whole guest
(`guest_default_revocation=preserved`).

| `--runtime-revocation` | ABI control saw | `revocation-control` | case 0 |
|---|---|---|---|
| **on** | `runtime_revocation=1` | `tag_after_sweep=0`, then `SIGPROT` `si_code=2` at `pc=0x101dd6` = `apr_defect_read` | **completed**, `nodes=3 node_reuses=1` |
| off | `runtime_revocation=0` | `tag_after_sweep=1`, completed | completed, `nodes=3 node_reuses=1` |

Every fault line is the supervisor's — signal, `si_code` and PC from the
kernel, the expected address from the child's memory map and ELF — and the
program under test printed nothing about itself.

## Why the case completes

`apr_pool_destroy` ends with `allocator_free(allocator, active)`
(`apr_pools.c:414`), which files the pool's nodes on `allocator->free[index]`,
APR's own size-bucketed LIFO list. The next `apr_pool_create` pops the same
node back. Under the default `APR_ALLOCATOR_MAX_FREE_UNLIMITED` no node on
this path ever reaches `free()`, so libc's quarantine never holds it and its
revoker never sweeps it. `node_reuses=1` in the report is the adapter counting
exactly that pop. This is the same argument the pymalloc corpus made for
CPython's pools and the FFmpeg corpus for `AVBufferPool`: a sub-allocator that
recycles without freeing is invisible to a mechanism that lives on `free()`.

## The controls, and that they can fail

`revocation-control` is the positive control: with revocation on it must fault
at the very shape the case probes, and it does, in the same boot, before the
case runs. A completing case beside a faulting control says "the mechanism is
active and did not fire", which is a different sentence from "there is no
mechanism". With revocation off it must complete, and does.

`--negative-control` (revocation on) corrupts the fixture so the case's
`CHECK(700)` refuses it before any pool exists: the oracle fired, 1/1, and the
revocation control — which takes no fixture — still faulted as required.

## What this does not show

Nothing about httpd's tree (`live_in_pin` stays `null`). No PoisonCap build of
APR exists, so no protected arm on this target. One case, one shape. The guest
image is the PoisonCap platform's with its libc patched
(`../../../../cpython/pymalloc-repros/platform/`); without that patch a
process with revocation on dies in libc's own start-up, before any of this
runs — so "stock" here means the platform's kernel default and its `malloc`,
not the published binary byte for byte.
