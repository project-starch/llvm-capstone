# Platform qualification, 2026-10-26

The virtual Capstone platform these PHP arms run on, qualified before any PHP verdict from
this tree is believed. `runtime/virtual/run.py` is the application-adapter gate,
`run-nodes.py` the revocation-node table, `run-malloc.py` the Capstone mallocng the PHP seam
sits directly on.

| gate | record | result |
|---|---|---|
| `run.py --skip-recycling` | `run.json` | **PASS 42/42** |
| `run-nodes.py` | `nodes.json` | 15/16 |
| `run-malloc.py --exact-bounds` | `malloc.json` | 20/23 |

This is the first PASS of `run.py` recorded for this port. The previous best, on the
pre-merge v5 platform (2026-10-12), was 34 of 41 distinct checks across two attempts, and
that platform's own record said FAIL while PHP arms were measured on it anyway.

`vm-mappings` passes, which is the check that holds 1200 mappings and fails on the old
module: it is the positive control for the arena-ceiling fix (`CV_MAX_ARENAS` 512 -> 8192)
that the whole upstream merge was for. `mruby` passes, which it never had: it is the only
input in the battery shaped like PHP -- an interpreted runtime with its own allocator doing
file I/O through delegated syscalls.

## The configuration, and why it is not the default

    CAPSTONE_QEMU_CPU_EXTRA='x-capstone-test-csrs=true,x-capstone-supervisor-slots=32'
    --module-args 'supervisor_slots=32'

Upstream made CSRs 0x5c0 (scapctl) and 0xcc0 (urevavail) illegal by DEFAULT on 2026-10-18
(Q-21/Q-22), because vp2 silicon has neither. The PAGED build of `capstone_vm.ko` still reads
0xcc0 once per context creation (`module/capstone_vm.c:2006`); only the flat build
(`-DCAPSTONE_VM_FLAT_TABLE`) does without it. So the default module and the default emulator
are mutually incompatible and nothing in either one says so.

The PAGED table is a deliberate choice, not the default taken by accident: flat caps the
lifetime table at `CV_FLAT_RECORDS` (65,536), and this seam's measured node high-water is
64,587-90,218. Flat would exhaust it mid-run, and a cause-30 resource fault lands at an
arbitrary pc -- indistinguishable from a catch. `nodes.json` records `nodes=140028` with
`collections=2` and reclaim working, so the paged table grows and sweeps at the scale PHP
needs.

`supervisor_slots` defaults to 1, which is what silicon holds. Every thread and shared-heap
check fails there; the 7 that remained after the CSR fix were exactly those.

Diagnosing the absence of either is expensive, which is why the properties are now set and
recorded in `run-virtual.py` rather than passed by hand: the missing-CSR trap is undelegated,
reaches M-mode, and `return_from_domain` stores through a null `caller_buf` -- a CSRUNV domain
has no CALLING domain -- wedging the guest with `cause = 24` inside the monitor, naming
neither the CSR nor the illegal instruction. `qemu -d int` shows the real chain.

## The four residual failures, each accounted for

None is a defect in this build, and none weakens an enforcement this experiment relies on.

- **`node_count_reported`** (`run-nodes.py:90`) greps for `NODE_COUNT_USED <n>`, a string
  `node-contract.c` never prints; it prints `NODE_LIVE:0:70000`. A runner/contract naming
  mismatch, with the quantity present under the other name.
- **`bounds`** (`run-malloc.py:134`) stores at +200003 of a 200003-byte allocation and expects
  cause 28. The same merge introduced Option (iii) granule bounds in mallocng -- exact below
  4 KiB, else `2^(msb(len)-9)` -- which for this size is a 256-byte granule, so the bound is
  legitimately 200192 and the store lands in 189 bytes of slack. The test asserts a denial the
  new policy forbids.
- **`local_allocation_fast_path`** requires no allocator service in the hot region and saw 32
  `CV_SERVICE_NODES` crossings. Q-21 deliberately removed the CSR pre-check from `heap.c`,
  which documents the change as "behaviour-preserving ... the cost is a service call on the
  path that previously returned without one". The check predates it.
- **`real_linux_mremap_growth_and_shrink`** is open upstream and recorded there as "a check to
  re-scope, not a defect".

## Carried into the PHP results as a validity limit

Option (iii) means bounds are exact only below 4 KiB; at or above, they are enforced to a
`2^(msb(len)-9)` granule. Nearly every `emalloc` object in this corpus is small, so the
`exact` arm is genuinely exact for them -- but it cannot catch an overflow that lands within
the granule of a large allocation, and no cell should be read as evidence that it can.

## Reading these records

`run.json`'s `virtual_vm_abi` field says 5 because `run.py:322` hardcodes it. The images
measured here carry the `CPONVVM6` marker. Trust the marker, not that field.
