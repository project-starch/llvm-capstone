# Sublet and PoisonCap on the SQLite engine corpus, in PoisonCap's own memsys5 configuration (2026-10-26)

The rerun of `../2026-10-24-poisoncap/` that the first author asked for on 2026-10-26: every
comparison as close to each system's defaults as possible, and both systems in the identical
configuration. The 2026-10-24 record gave memsys5 the corpus harness's 1 MiB static heap with a
minimum request of 64. That is our harness's choice, not PoisonCap's. PoisonCap's own speedtest1
`main()` configures memsys5 with a 256 MiB heap from `malloc` and a minimum request of 16
(`../../../../ports/sqlite/poisoncap/results/2026-10-25-speedtest1/README.md`). This record runs
both systems that way.

Same population as 2026-10-24: the 17 temporal cases 00 03 05 06 07 10 12 14 15 16 17 19 20 21 22
23 24. Same verdict rule: a case counts when the process faults in one of the case's fault sites,
the functions host ASan reported (`case.json`). Every configuration ran twice, and the two runs
agree in every case (`tabulate.out`).

## Configurations

| run | system | heap | minimum request | port |
|---|---|---|---|---|
| `driver-la`, `-rep2` | PoisonCap, published artifact | 256 MiB from malloc | 16 | their memsys5 |
| `v6-sublet-block`, `-rep2` | Sublet, dev-v6 platform | 256 MiB from malloc | 16 | block port |
| `v6-sublet-block-keeps`, `-rep2` | Sublet | 256 MiB from malloc | 16 | block port, ablation |
| `v6-sublet-driver`, `-rep2` | Sublet | 256 MiB from malloc | 16 | request-bounds port |
| `heap256-la`, `-rep2` | PoisonCap | 256 MiB static | 64 | their memsys5 |
| `diag-driver16-la`, `diag-driver64-la` | PoisonCap, diagnostic amalgamation, cases 16 and 17 only | 256 MiB from malloc | 16, 64 | their memsys5 |

Lookaside on everywhere (SQLite's default). PoisonCap boots as on 2026-10-24: single-user with
`security.cheri.runtime_revocation_default=0`, because their published kernel panics in a normal
boot, so the harness removes `CHERI_PERM_POISON` from the heap itself, the one change their malloc
makes. Sublet runs on the platform whose six hashes equal the 2026-10-24 v6 runs
(`sublet-platform.sha256`), CPU as recorded there (`runs/*/cpu.txt`). Both guests have 2 GiB.

### Harness changes this record needed

- `--heap-bytes`, `--heap-min` and `--heap-malloc` on both builders
  (`ports/sqlite/repro322/build-virtual.py`, `ports/sqlite/poisoncap/corpus/build-cases.py`), with
  `SQLITE_HEAP_MIN_REQ` and `REPRO322_HEAP_MALLOC` in `ports/sqlite/repro322/repro322_common.h`.
  The heap comes from `aligned_alloc` before `main()`, aligned to its bounds granule.
- Why from malloc: with a 256 MiB static array the Sublet launcher refuses the image grant
  (`CAPSTONE_VM_ADD_REFUSED base=0x3f80000000 bytes=536870912 errno=22`), so no case starts. Their
  driver takes its heap from malloc as well.
- `ports/common/host/cheribsd/guest.py` decodes the guest's output with `errors="replace"`: case 16
  printed the bytes its stale read returned, and strict UTF-8 ended a whole run.
- The controls are built without the case flags, so they keep the 1 MiB static heap and minimum 64.

## The block port

`ports/sqlite/sublet/sublet-3220000-memsys5-block.patch` (generated from the request-bounds port by
`make-sublet-block-port.py`), selected by `build-virtual.py --block-port`:

- every memsys5 block is a child lifetime bounded to its whole power-of-two block, with stock
  rounding, sizes and heap geometry, as PoisonCap's memsys5 hands blocks out;
- every successful realloc ends the old lifetime and returns a new one, also when the block stays:
  memsys5Realloc's in-place branch (`memsys5Renew`), SQLite's same-size shortcut in `sqlite3Realloc`
  (removed, so it reaches memsys5Renew), and the lookaside slot in `sqlite3DbRealloc`
  (`lookasideRenew`). The new child is a sibling of the old one, derived after the old one is
  revoked;
- `--realloc-keeps-lifetime` builds the ablation (`-DSUBLET_REALLOC_KEEPS_LIFETIME`): a realloc that
  keeps its block returns the old lifetime, and SQLite's shortcut is back.

Its four controls (`controls/control_realloc_*_mem5.c`, `controls/control_capacity_mem5.c`), read
by `run-virtual-staged.py` with expectations that follow the build:

| control | block port | ablation |
|---|---|---|
| realloc to the same block size, then read the old pointer | cause 25 in control_read | returns |
| realloc that shrinks in place, then read the old pointer | cause 25 in control_read | returns |
| realloc larger than the heap fails, then read the old pointer | returns, contents kept | returns |
| write the last byte `sqlite3_msize` reports, without a realloc | returns | returns |

Before each stale read the control prints the new pointer: same address, contents kept
(`same_address=1 kept=1` in every run). The lookaside path of `sqlite3DbRealloc` has no control:
no public API reaches it directly.

## Results, 17 cases

| configuration | fault at site | not at site |
|---|---|---|
| Sublet, block port | **17** | none |
| Sublet, block port, ablation | 16 | 12 |
| Sublet, request-bounds port | 17 | none |
| PoisonCap, driver configuration (256 MiB, minimum 16) | **13** | 12, 15, 16, 17 |
| PoisonCap, 256 MiB static, minimum 64 | 15 | 12, 15 |
| PoisonCap, 1 MiB static, minimum 64 (2026-10-24) | 12 | 07, 12, 15, 21, 23 |

**Heap size.** At 256 MiB the three quarantine cases of 2026-10-24 (07, 21, 23) fault at their
sites on PoisonCap. Their 1 MiB result was the harness's heap, not PoisonCap's.

**Case 12.** `fts5StructureExtendLevel` grows `aSeg` by realloc while an fts5vocab cursor still
reads through a pointer into it. With stock rounding the array grows inside its block, so no
memory is freed. On the block port the realloc still ends the old lifetime, and the cursor's read
faults (cause 25 in `fts5SegIterNextPage`) at the same placement as stock memsys5. The ablation
completes it, so the catch comes from the realloc lifetime and not from the rounding. PoisonCap
completes it in every configuration: nothing is freed, so nothing is poisoned.

**Case 15.** As on 2026-10-24: their QEMU's poison lookup keeps address bit 1 (EVIDENCE.md there).

**Cases 16 and 17 at minimum 16.** PoisonCap completes both, and 16 prints the bytes its stale read
returned. At minimum 64 both fault with the poison report: in `memcmp` on the scored static run
(`heap256-la`), and at the same libc offset (0x1b095c) with poison console lines from malloc
(`diag-driver64-la`, not scored). The diagnostic runs rule out a quarantine flush: 16 and 17 free 176 and 259 blocks
at minimum 16 (77 and 152 at 64), far below the 4,096 entries that flush the quarantine without a
revocation, and no flush is printed. Why the poisoned block is not reported at minimum 16 is not
established. Their QEMU's bit-1 lookup (case 15) is a candidate, since the minimum changes every
block's address, but the address of the stale read was not recorded.

## Limits

QEMU only, on both sides. Function-level sites, as on 2026-10-24. The controls keep the corpus's
default heap. The request-bounds rows are kept for comparison with 2026-10-24 and are not the port
the paper presents.

## Files

`runs/<run>/`: `results.jsonl`, `build.json` of the build it ran, and `run.json` (PoisonCap) or
`cpu.txt`, the guest's `-cpu` string (Sublet). The staged runner's own output (`run-staged.out`) is not kept:
it repeated that string beside lock messages and a path to a serial log that is not committed. `verdicts/<run>/`: PoisonCap's scoring
(`corpus/score-cases.py`). `scripts/`: the drivers as run on p13 (`heap256-driver.sh`: both systems
in the driver configuration; `block-port.sh`: the block port and its ablation; `diag-1617.sh`;
`heap256-run.sh`: the static 256 MiB PoisonCap runs; `tabulate.py`). `kit-log.txt`: the kit
commits the runs were built from, which are this branch's harness files at this commit.
