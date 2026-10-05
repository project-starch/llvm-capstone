# memcached class-A spatial fixtures 20 and 21, three heap arms, QEMU (2026-10-05)

**What this measured.** Two new not-nested spatial probes — fixture 20 (a one-past **read** of a
`calloc`'d buffer with no terminator) and fixture 21 (a NUL **write** one byte past a `malloc`'d
buffer whose reservation was one short) — on `level0`, `shrink` and `sublet`, together with the
standing controls fx2/fx3 and the temporal control fx17. **15 of 15 cells as predicted**, against
rows registered before any image existed.

`result-lines.txt` has every row, each image cited by hash.

## The reading

| | fx2 | fx3 | fx17 (temporal) | **fx20** | **fx21** |
|---|---|---|---|---|---|
| `level0` | RETURN | RETURN | RETURN | **RETURN** | **RETURN** |
| `shrink` | FAULT cause 7 | FAULT cause 5 | RETURN | **FAULT cause 5** | **FAULT cause 7** |
| `sublet` | FAULT cause 7 | FAULT cause 5 | FAULT **cause 24** | **FAULT cause 5** | **FAULT cause 7** |

**`level0` RETURN with `shrink` FAULT is what makes these class A** — a crossing of the `malloc`
bound itself, which per-object bounds already cover. The falsifier was written into
`host/safety-expect.txt` before the run: had `level0` also faulted, the fixture would not have been
class A and the row would have been wrong rather than the port.

Three independent consistency checks the cells pass, none of which the pass/fail verdict needed:

- **the cause matches the kind of access.** fixture 20 is a read and reads cause **5** with
  `insn = 00054503` (a load); fixture 21 is a write and reads cause **7** with `insn = 00c50023`
  (a store). The read/write distinction this project established on the tshark rows holds here.
- **the faulting address is the fixture's own printed target**, not merely some out-of-bounds
  address: `MCAPP-FIX 20 target=a048c380` against `addr = a048c380` on `shrink`.
- **fx17 separates the axes.** The temporal control returns on `level0` and `shrink` and faults on
  `sublet` with cause **24**, revoked authority — a different cause class entirely. An arm set
  where every cell faulted for the same reason would not have shown that.

**Sublet adds nothing spatially, as expected.** `sublet` reproduces `shrink`'s readings on all four
spatial fixtures because the catch is `shrink`'s per-object bound, which `sublet` inherits.
Revocation has nothing to fire on while the object is alive. The arms differ only on fx17.

## What these fixtures are, and are not

They are **probes**, not reductions of live upstream defects. Each is shaped on a historical
memcached defect — fixture 20 on `ddee3e2`, fixture 21 on the cachedump reservation fixed in
`d5d9ff0` (cited by hash) — and **both of those are already fixed in the source this port pins**:
`authfile.c:50` is `calloc(1, sb.st_size + 2)` and `items.c:678` reads `bufcurr + len + 6`. Calling
them live defects would repeat the error retracted in `543d485bd44e`.

A probe is the right instrument for this row regardless: its job is to show the arms discriminate at
`malloc` granularity, which no upstream class-A defect in these three programs can do, because class
A is the class upstream fixes first. See
`docs/ref/spatial-and-temporal-bug-inventory.md` for the dispositions behind that.

## How it was run, and the instrument changes it required

`ports/common/application/run-fixtures-9p.py`, which judges with the ports' own oracle
(`check-safety.py`'s `classify()`/`matches()`, imported rather than reimplemented) instead of the
hand check that earlier memcached bundles had to use and flagged as *"a hand check is not the gate"*.
Three fixes were needed before a single cell could be believed, and **each was caught by a control
rather than by the new fixtures**:

1. **the runner had no memcached shape.** ffmpeg and wireshark build one image per fixture;
   memcached builds one image per *arm* and selects the fixture with a hidden server command, so its
   launcher is `host/mc-harness`. Added as a strict extension — the other two ports' guest command
   is byte-identical.
2. **the oracle's "stdout" is the server's output**, which `mc-harness` collects as `server.out`,
   not the harness's own stdout. Feeding it the latter made all five cells report *"fixture mark and
   actual exit status disagree"*: an 8-bit exit status cannot carry the full mark, so only the
   printed line has it.
3. **`check-safety.py` translated `TSAPP-` but not `MCAPP-`.** With the prefix untranslated the mark
   regex matched nothing, which raises the same error for the same reason. One word in the oracle,
   where the `TSAPP-` translation already lived.

Defects 2 and 3 each produced **0 of 5 as predicted, on all three known-good controls at once** —
a uniform failure, which is what an upstream instrument fault looks like and not what a finding looks
like. Had fixtures 20 and 21 been run alone, the same output would have read as "the probes do not
work".

**The gate is negative-tested.** With fx20's `level0` row forced to `FAULT oob`, the judge reports
`fx20: DIFFERS: RETURN 1400040` while fx2 and fx21 still pass, and the runner exits 1. A gate that
has never blocked anything is unproven; this one blocks.

## Platform and caveats

- The pinned platform: `deleg-gate2/qemu-12` and `pinned-platform/images-root`'s `fw_jump.elf`,
  kernel command line `cma=1536M`, `process_cache_bytes=402653184`,
  `CAPSTONE_GP_NONLIN=1`, `CAPSTONE_REV_NODES=65536`. Application-SDK images need the monitor's
  process ABI, which the installed buildroot monitor does not have.
- **QEMU only.** `sublet`'s fx17 fault is the emulator untagging a revoked capability on reload
  (Q-11); deployed silicon lets such an access retire.
- **N = 1 per cell**, three boots plus one for the negative control.
- Built with the qualified toolchain (`llvm-capstone-cc/build-release`). The shared tree's
  `llvm/cmake-build-debug` fails the SDK's ABI gate on its own account (C-46, a linear direct-call
  target), and the SDK reads `CAPSTONE_LLVM_BUILD_DIR` rather than `CAPSTONE_LLVM_BIN` — inside a
  worktree that defaults to a directory with no LLVM in it, which surfaces as
  `capstone-domain.cmake:29` failing and then as a spurious "missing Ninja".
