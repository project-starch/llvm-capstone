# Cases 19-21: CPython's system-allocator defects, 2026-10-11

Cases 19, 20 and 21 are this corpus's three temporal defects in memory that comes straight from
the system allocator, never from a pymalloc pool. On Capstone that allocator is musl 1.2.5's
mallocng (the upstream archive, sha256 a9a118bb..., which `check-mallocng-policy.py` enforces),
compiled for Capstone with patch 0009 (slot metadata in capability records) and the runtime's
`runtime/virtual/heap.c` (bound each object to its request, retire its lifetime on free). On
CheriBSD it is the platform's jemalloc with its revocation quarantine. What was predicted, and
every change made after the first run, is in [PREDICTIONS.md](PREDICTIONS.md), each part
committed before the run it governs.

## Result

| case | virtual-malloc | virtual-nested-pools | CheriBSD |
|---|---|---|---|
| 19 | CAUGHT, cause 25 in `fnv`, by control | CAUGHT, cause 25 in `fnv`, by control | no fault; all 22,336 frees quarantined, none reused |
| 20 | CAUGHT, cause 25 in `memcpy`, by function | CAUGHT, cause 25 in `memcpy`, by function | no fault; all 6,191 frees quarantined, none reused |
| 21 | CAUGHT, **cause 24** in `little2_toUtf8`, by function | CAUGHT, cause 25 in `little2_toUtf8`, by function | SIGPROT, `PROT_CHERI_TAG` |

For the paper's table, CPython's system-allocator cell is 3 cases. On CheriBSD all three earn
use-after-reallocation protection by the registered rule: no stale pointer reached a reallocated
object. The quarantine held 19's and 20's freed memory, and a sweep revoked 21's stale
capability before its read. Only 21 is a caught use-after-free there. On Sublet (virtual-malloc,
the system allocator alone) all three are caught use-after-frees. virtual-nested-pools agrees row
for row, as predicted, since pymalloc plays no part in these three.

Every row was predicted, with one deviation: case 21 on virtual-malloc faults with cause 24, not
the predicted 25.

## Case 21's cause 24

Both images carry the same code for `little2_toUtf8`. Its first instruction loads `*fromP`. That
slot is `fromPtr`, a local of `doContent` (xmlparse.c:3368 in host ASan's stack), which holds the
stale pointer into the freed 512 KiB buffer.

* On virtual-nested-pools the load returns a tagged capability, and the byte read at +0x48
  faults with cause 25 at a heap address: an access through a revoked capability.
* On virtual-malloc the same load returns an untagged one, and `cincoffset` at +0x14 faults with
  cause 24. The reported address is 0x0 because `helper_cscincoffset` raises the
  untagged-operand exception without an access address.

capstone-qemu demotes a capability to untagged at `ldc` when it is revoked and its type is not
NONLIN (`target/riscv/op_helper.c`, `helper_reg_set_cap_compressed`, at QEMU af37cc32). A
copyable reference keeps its tag and faults only when used. So the cause-24 fault is most
plausibly the same retired lifetime, observed at the reload instead of at the access. That is an
inference, not an observation. It rests on the code between the store of `fromPtr` and this load
being identical in both images, with the slot holding the capability on the other image; nothing
in that code stores data over the slot.

Not resolved: why the buffer's capability is NONLIN on one image and not on the other. Patch
0014 does not touch requests over 512 bytes, so the difference lies in which mallocng path served
the buffer. No fault value was recorded that would settle it.

## The runs

| directory | what | runner revision |
|---|---|---|
| `first-run/` | both arms, 19-21. 19 unattributed: its whole-file control faults at another site (see PREDICTIONS, first amendment); 21 timed out at 300 s | 9239e39689e9 |
| `virtual-malloc/`, `virtual-nested-pools/` | both arms, 19-21, with 19's control paired with `probe-142664.py`. 21 timed out at 300 s | 07fe2f38f349 |
| `case21-timeout-3600/` | both arms, 21 alone, `--timeout 3600` (each arm's whole run, controls included, took 400 s and 449 s) | 07fe2f38f349 |
| `cheribsd-first-run/` | 19 not reached (no `test` package), probe lines from `timeout`; recorded, not used | 0bac348cd567 |
| `cheribsd/` | 19-21 | 6c985a59425b |

Virtual images, both built by `build-virtual.sh cpython` from 9239e39689e9 with the 8 MiB main
stack: stock a58346545eaa, pools 77191403e26a. Their manifests say `runtime_dirty`; the only
untracked paths in that tree were two `__pycache__` directories the build writes. Platform: the
Sublet QEMU af37cc32, module a1c6cb6b, launcher 3975314a, as `inputs.json` records in full.

CheriBSD: the purecap image e7470361 booted from snapshot on p13, CheriBSD 15.0-CURRENT,
`runtime_revocation_default=1`, `runtime_revocation_every_free_default=0`,
`runtime_revocation_async=1`, the `security.cheri` subtree unchanged across the run. The
interpreter is `ports/cpython/app/cheribsd/build.sh` as committed in 0bac348cd567 (the build ran
just before that commit from the same file, sha256 compared), binary c9bec41630d4. The
helpers are `tools/cheribsd/build-helpers.sh`: `sicode.so` 511a3074, `quarantine-probe.so`
4f68353c. Every guest control passed before the cases. The workload printed `EXP-OK cpython 552`
with the probe reporting 4,878 frees, all quarantined. The self-test reported `PROT_CHERI_BOUNDS`
and `PROT_CHERI_TAG` through the preload. `run.meta`'s times come from p13's clock, which runs
about 26 minutes behind.

What the CheriBSD rows rest on, case by case:

* 19 reached its defect. The six `test_hash_use_after_free` tests ran and failed with
  "BufferError not raised by hash", which is what the defect does: `hash` read the released
  buffer and returned. The file ended `ABR EXIT 1`, its unittest status.
* 20 ran to `ABR RETURNED`.
* 21 died on SIGPROT before any end line, so it has no probe line; the fault is its record.

A tag fault on CheriBSD here is not attributed to a site: the arm's record is the signal and
si_code, as in `results/20261006`.
