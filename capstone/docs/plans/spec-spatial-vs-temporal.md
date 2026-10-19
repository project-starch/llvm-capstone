# SPEC CPU2017 on the virtual profile: spatial-only against spatial-plus-temporal

Status 2026-10-19: the heap switches, the contract and the harness are on the lane branch
`runtime/heap-lifetime-switches`; the SPEC kit and the runs follow once the project lead settles the
sizes and the host window. Reviewed once before any code was written (two findings about the controls,
folded in below) and again as code.

## The question

What does the temporal half of the virtual profile's heap protection cost on SPEC CPU2017's C
benchmarks, with the spatial half held fixed? Two arms that differ in exactly one object file, measured
under the same compiler, libc, platform and inputs.

## What "temporal" is in the virtual profile

Every per-object temporal mechanism is in `runtime/virtual/heap.c`: `malloc`, `realloc` and
`aligned_alloc` give the object a lifetime of its own (CDERIVE, after a CSR read of the node reserve
and, when it is low, a node-service round trip), `free` zeroes the payload and CREVOKEs it, an in-place
`realloc` revokes and re-derives, and a nested mallocng group is a child of the slot that holds it. The
compiler emits no CDERIVE or CREVOKE, so there is no stack or global temporal protection to switch.
Shared by every arm and not per object: mapping retirement on `munmap` and `mremap` (module),
thread-stack retirement at join, node-table collections (only under node pressure), and QEMU's
per-access liveness check. The physical level0 heap has the same switch as `CAPSTONE_LEVEL0_LIFETIMES`;
the bug corpora's "spatial" arm and the paper's "Capstone spatial" column are that bounds-only mode.

## The switches

`runtime/application` options `CAPSTONE_HEAP_LIFETIMES` and `CAPSTONE_HEAP_SCRUB_FREE`, both ON by
default, compiled into `heap.c`, recorded in `sdk.json` as `heap_lifetimes` and `heap_scrub_free`,
forwarded by `build-sdk.sh`; `mallocng.md` describes them. With lifetimes off a public object and a
nested group are SHRINK copies of the slot's authority (same bounds, same rights, the slot's
lifetime): nothing below a VM mapping is ever revoked. With the scrub off, `free` leaves bytes and tags.

Arms, in the bug corpora's vocabulary:

| arm | lifetimes | scrub on free | measures, against the next |
|---|---|---|---|
| `virtual-mallocng` (temporal) | on | on | the mechanism as shipped |
| `virtual-mallocng-spatial-scrub` | off | on | temporal minus this = the lifetime machinery alone |
| `virtual-mallocng-spatial` | off | off | spatial-scrub minus this = the free-time zeroing alone |

The fourth combination (lifetimes on, scrub off) builds too and is used only by the contract gate.

## Controls: the arm must lose exactly the temporal catches

`run-malloc.py --sdk DIR` reads the switches and selects its expectation table. With only the
object path switched, mallocng's release of an empty nested group would still have revoked the
group and a stale probe would have faulted for the group's reason (review finding 1); the switch
therefore covers nested groups too, and the denial cases anchor their size class so that a freed
object never empties its group and nothing above it (outer slot, mapping) is released either. The
anchor and the probe are the first two allocations of their class, which mallocng places in the
class's first group; the contract checks their distance as a witness. Readings pre-registered for
the bounds-only SDK:

| case | qualified heap | bounds only |
|---|---|---|
| `bounds` | 28 at `cap_malloc_fault_store` | 28, unchanged |
| `remap-stale` | 25 | 25: the module retired the old mapping, a VM-level retirement every arm keeps |
| `stale`, `inplace-stale`, `reuse-stale` | 25 | survive: `MALLOC_SURVIVED`, exit 0, no fault record |
| `reuse-free` | 25 at `cap_malloc_validate` | survives, freeing the new owner's object |
| `stale-free` | 25 at `cap_malloc_validate` | 2 at `cap_malloc_invalid_free`: the first free unindexed the slot |
| `interior-free` | 2 at `cap_malloc_invalid_free` | 2, unchanged |

The harness cannot express "must complete" by exit code alone (review finding 2): the contract takes
`may-survive` and prints the marker, and the survive check requires marker, exit 0 and the absence of a
fault record. The qualified SDK must still read the left column in the same harness. The node-collection
check is dropped only for the bounds-only arm, where only the lifetimes per mapping and the reserve
requests remain, far below any collection threshold.

## Kit (private SPEC kit, separate commit)

- Three SDKs from one tree against the PINNED clang build and libc archive: `build-sdk.sh` takes both,
  so no compiler or musl rebuild. Prove it: the SDK members differ only in the runtime archive, and the
  runtime archives only in `heap.c.o`.
- One SPEC tree, one label per arm; `build.sh` installs src.alt once and builds every label;
  `verdict.py` takes the label. Integrity gate over the complete runcpu build inventories (objects,
  archives, command lines): identical across labels except the SDK runtime archive.
- `capstone-run.sh` passes `-e CAPSTONE_VM_STATS=1` to `capstone_vm run`; the launcher's stats line
  (nodes, collections, reclaimed, faults, node bytes, launch and elapsed time) lands in the command's
  `.err` file, which the kit parses per invocation. The benchmark's own stderr is untouched.
- `runcpu --action=validate` on every arm: unscrubbed reuse can change what code that reads
  uninitialized memory does.

## Measurement

- Primary: runcpu's reported seconds for the benchmark executable's invocations only (validators and
  the host ldecod excluded, pre-registered). Secondary: the QEMU process's CPU time (utime+stime,
  sampled from /proc) and the launcher's `elapsed_ns - launch_ns`. Report arm ratios per benchmark,
  and both against the plain-RV64 binaries that already exist on the build host.
- Breakdown: capprof instruction arrivals in the whole-program window, summed over the publish and free
  paths, the node service and the module's collection counters. Arrivals include faults, retries and
  the wall-time quantum's exits, so they are reported as arrivals, with that share, not as a
  deterministic count. Caller attribution of `memset` is not available; the scrub arm supplies it.
  Per-PC counts are only meaningful with QEMU's `x-pcrel=false` (the plugin's own warning: with
  pc-relative translation an address can belong to another translation alias), the launch arguments
  recorded, and every PC resolved against the ELF that actually ran.
- Conditions: the build host's socket 1 only, one guest per physical core, the SMT sibling idle, fixed
  concurrency, core assignment counterbalanced across repetitions, host load and per-QEMU scheduling
  recorded, three repetitions, rejection criteria pre-registered independently of the ratio.
- Sizes: train (the qualified size) first; ref only on the project lead's go.
- What the numbers cannot say: QEMU performs the liveness check in both arms, but its lookup cache is
  keyed by node identity and invalidated by table writes, so part of a temporal-arm slowdown can be the
  emulator's implementation of a check both arms perform; the hardware cost of lifetime checking comes
  from the RTL, not from here. Exact-bounds shadow metadata follows capability-bearing pages in both
  arms. Host QEMU RSS is reported beside guest pinned pages and node bytes.
