# Reuse distance at memsys5, Sublet's block port: `--size 100` (2026-10-26/27)

The Sublet arms of `../2026-10-25-reuse-size100/`, rebuilt on the block port that the corpus record
`bug-corpora/sqlite/engine-repros/results/2026-10-26-poisoncap-heap256/` introduced. Same instrument
(`reuse/reuse-wrap.c`), same workload (speedtest1 with no arguments, PoisonCap's driver: `--size 100`, the
20 tests of its main set, memsys5 with a 256 MiB heap from malloc and a minimum request of 16), same platform.

| arm | port |
|---|---|
| on (`block-port`) | `ports/sqlite/sublet/sublet-3220000-memsys5-block.patch`: every memsys5 block a child lifetime bounded to its whole power-of-two block, stock rounding, and every successful realloc a new lifetime, also when the block stays (SQLite's same-size shortcut in `sqlite3Realloc` removed, so it reaches `memsys5Renew`) |
| off (`block-port-spatial`) | the same with every per-object revocation removed (`sublet/sublet-spatial-only.py`, including `BLOCK_PORT_EDITS`, which removes `memsys5Renew`'s) |

## Build, qualification, runs

- Built by kit `repo-heap256` 0cfe025 ("reuse arms block-port and block-port-spatial", llvm-capstone
  98e7ae9a62a) with `sdk-v6`, through `reuse/build-sublet.sh`. The same kit and SDK rebuild the
  2026-10-25 `block-bounds` image `6bfe4d1b` bit for bit. Hashes: `SHA256SUMS`.
- Qualification: run `qual-block` (`scripts/launch-qual-block.sh`), checked by `reuse/qualify.py capstone`:
  ALL PASS for both arms (`qualification.txt`).
- Result runs: set 1 on p13, `scripts/launch-block.sh` (paged platform, `supervisor_slots=32`), the two arms
  in separate QEMU domains at the same time, started 2026-10-26 about 20:50 +08.
- Collected by `reuse/collect.py MIRROR OUT --block-port` (`checks.txt`, `runs.json`, `reports.jsonl`).

## Results

Both runs are admissible (exit 0, the 20 tests, installed=1, error=0, 32 bins, the four identities) and
agree in every counter and bin:

| | issues | reuses | distinct | failed | moved | inplace | total |
|---|---|---|---|---|---|---|---|
| block-port-spatial (off) | 52,146 | 51,499 | 647 | 0 | 20,541 | 7 | 38,934 s |
| block-port (on) | 52,146 | 51,499 | 647 | 0 | 20,541 | 7 | 38,855 s |

bins (both): 11444, 32334, 5149, 1660, 424, 232, 91, 14, 19, 12, 0, 43, 36, 5, 9, 27, then zeros.

- **Temporal protection changes no placement.** On and off are identical, as for the 2026-10-25 arms.
- **The block port places objects as PoisonCap's memsys5 does.** The bins equal those of PoisonCap's own
  memsys5 with its switch off (2026-10-25, set 2, `aad6a979`) in every bin. The 2026-10-25 Sublet arms had the
  same issues and reuses with slightly different bins (11439, 32319, ...).
- **Realloc in place renews a lifetime without a false alarm.** `inplace` counts reallocs that return the
  old address (`reuse-wrap.c`); they issue no allocation, so the bins do not see them. It is 7 here and 0 in
  every 2026-10-25 arm, because the block port removes SQLite's same-size shortcut. On the protected arm
  each of the 7 ends the old lifetime and returns a new one at the same address, and the 20 tests
  complete with exit 0 and no fault.
- One set per arm. The 2026-10-25 arms ran three sets each, and their repetitions were identical.
