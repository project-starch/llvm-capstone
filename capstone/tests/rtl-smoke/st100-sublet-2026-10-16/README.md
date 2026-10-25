# speedtest1 `main --size 100`, domain against native, on the Sublet bitstream: pre-registration (2026-10-16, before the boots)

The board campaign's wave 3, item 10 (`docs/plans/2026-10-16-board-campaign-all.md`). It repeats measurements doc §7p
on `caplifive_sublet_d93a8d66b.bit`. §7p was boot sw73 on `caplifive_r30r31_1bfff7776`: ratio 1.1819, domain
398,572,346,349 cycles against native 337,235,381,252.

## The images, rebuilt to §7p's recipe

§7p's set is on no disk of this machine: domain `f795151f3ed4883b`, native `ed54878519b96816`, host
`2af56927aaf907e9`.
- The search hashed every `.dom`, every `*baseline*` and `*host*` file, and every file of the domain's size
  (1,861,088 B), on the root disk and /mnt/hdd.
- Its control was P1's native `9a80c1cd2a576ed2`, stored as `native-baseline-O2`, which it found in ten places.

So the set is rebuilt with §7p's recipe, on lane b0-silicon-runtime with
`clang version 22.0.0git (llvm-capstone 612b3ec514c0)`:
- `SQLITE_FULL=on SQLITE_FLOAT=on`, SQLite at -O0 and the support code at -O1, the lookaside at its default;
- the domain's memsys5 arena a 128 MiB shared region (`SPEEDTEST1_REGION_ARENA=1 SPEEDTEST1_ARENA_SIZE=134217728`);
- the native's heap 128 MiB, static.

The recipe lives in a private wrapper around `ports/sqlite/run-speedtest1-measure.sh` and
`build-speedtest1-baseline.sh`. Each arm runs under QEMU `-icount` at size 1 and at size 100 (the main clone's
capstone-qemu deb7d75756).

| arm | image | QEMU `-icount`, size 1 | QEMU `-icount`, size 100 |
|---|---|---|---|
| domain | `2e839991ffd34e79`, host `2af56927aaf907e9` | `111130 1e792c9d`, 697,087,510 instructions, `arena_bytes=134217728`, HEAP 134217728 DROPPED 0 | not run (below) |
| native | `50bd00d74f63fd4c` | `111130 1e792c9d`, 548,764,450 instructions | `23674002 573a4409`, 90,025,116,459 instructions |

- **The host is byte-identical to §7p's measurement host**, `2af56927aaf907e9`.
- **The domain image rebuilt byte-identical** in a second build (21:33, then 22:20). The second build's log replaced the first's.
- **The size-1 instruction ratio is 1.2703.** §7k's `main` pair, the images sw63 measured, had 1.270.
- **The native's size-100 count is within 5 ppm of §7p's** (90,025,541,852), at §7p's oracle.
- **The domain did not run at size 100 under QEMU.**
  - At size 1 it ran 697 M instructions in about 100 s with its build.
  - A size-100 attempt had not finished after 28 minutes. I stopped it so that select4's SLT validation could have the
    lock.
  - It would hold the host's one QEMU lock, which every lane shares, for hours.
  - Its licence is the size-1 pass, for the same image file: the size is a run-time argument. Its size-100
    instruction count is predicted below.
  - The board's pre-boot runs it at size 1 first.
- All of it is in `build.result-lines.txt`.

**What this comparison is.** The tree and the toolchain have moved since §7p's images were built, as well as the
silicon. So the ratio here compares verdicts and ratios across tree, toolchain and bitstream together, not across
instruction streams. The QEMU instruction ratio separates the program's change from the silicon's.

## The boots

Driver: a private copy shaped as `../slt-sublet-2026-10-16`'s.
- Line: the classic line, d04bd83 / df0e9a8 / 93440d5, with the #3 module and `REFUSAL_RECORD=1`.
- Control: the k800 relinked to 0x20000, because the domain enters at 0x10000.
- Watchdog: the entry watchdog runs.
- Each boot keeps its `fw_payload.bin`.

| boot | arms, in order | budget per arm |
|---|---|---|
| st100s16-pre | k800, native `warm --size 1`, domain `--size 1` LAST | 1,800 s |
| st100s16-pair | k800, native `warm --size 100`, k800, domain `--size 100` LAST (§7p's composition) | 43,200 s |

**Why a separate pre-boot.** Nothing on this line has carved a 128 MiB arena.
- Before tonight the largest carves on it were H1's 4 MiB arenas. R1's 8 MiB heap arena runs earlier in the same
  chain.
- The board's CMA area is 256 MiB (`configs/caplifive.dts`, `linux,cma-default` at 0xac000000).
- The pre-boot answers whether the arena carves, and whether the domain runs, in minutes rather than four hours
  into the pair.
- It also gives a size-1 ratio on the same images.
- The pair keeps §7p's composition untouched.

## Pre-registered readings

**Pre-boot (size 1):**
- k800 4;
- both arms at the size-1 oracle `111130 1e792c9d`, the native with its `BASELINE-WARM` line;
- the domain with `arena_bytes=134217728` and `HEAP 134217728 DROPPED 0 RC 0`, and no share-trap or NOMEM.
- `gate-pre.py`, private, checks these arm by arm on the run-scoped transcript, and the pair starts only if all hold.
  It was tested on a transcript built from the two QEMU runs: it passes with the arms in place and fails with them
  swapped.
- **Cycle ratio about 1.22.** sw63 read 1.2195 on images whose instruction ratio, 1.270, equals today's 1.2703; that is
  a CPI ratio of 0.960. The size-1 absolutes are recorded, not predicted: the arena is now a 128 MiB region.

**Pair (size 100):**
- k800 4 twice;
- both arms at `23674002 573a4409`;
- `HEAP 134217728 DROPPED 0 RC 0` on the domain, and no NOMEM. NOMEM at hours in is §7p's named failure.
- **Predicted from sw73's decomposition** on today's native count:
  - domain instructions 90,025,116,459 × 1.2383 = about 111.48 G;
  - native cycles at sw73's 3.746 per instruction = about 337.2 G, 3.75 h;
  - domain cycles at 3.575 = about 398.6 G, 4.43 h;
  - **ratio about 1.18, band 1.14-1.24** (§7p's).
- The size of any move off 1.18 is the reading. H1 reads this bitstream's load latency as R-43 v2's, and the
  REVOKE path it found cheaper is not on speedtest1's path.

**Not established:** N = 1 per arm, as §7p; one bitstream.
