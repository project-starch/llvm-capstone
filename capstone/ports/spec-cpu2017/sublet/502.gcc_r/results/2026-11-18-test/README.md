# The lifetime port of gcc's allocators on SPEC's test input (QEMU, 2026-11-18)

**Question.** Does 502.gcc_r still give SPEC's output with its pools, GC heap and obstacks
ported, and what does the port cost on the test input?

**Answer.** Yes, with one fix in gcc. The ported compiler gives SPEC's reference output in
30 of 30 runs. The job runs **0.58 % more instructions** and takes about 8 % more QEMU wall
time. The test input is a 69-byte file, so this is the cost on a compiler start, not on a
compilation workload.

## Configuration

- **Kit:** revision `71ce199d8` of this branch, native task route: its SDK, kernel image,
  adapter and QEMU. The guest is Linux 6.1.0 with one hart and 4 GiB. Hashes are in
  `SHA256SUMS`.
- **SDK:** application ABI 2, virtual profile, mallocng with heap lifetimes on, object bounds
  on, scrub on free off. clang 22.0.0git.
- **Benchmark:** SPEC CPU2017 1.0.2, 502.gcc_r with the kit's `capstone` source alternative,
  built by `tools/build.sh` at `-O2` (`BENCHMARKS=502.gcc_r`, one SDK, label of its own).
- **Arms:** `stock` is that build. `sublet` is a copy of its build directory after `apply.py`,
  rebuilt by `replay-build.sh`. Both arms run on the same heap, so they differ in the port
  and the one gcc fix.
- **Command:** the benchmark's own for this input, `t1.c -O3 -finline-limit=50000`, started
  by `capstone-vexec`. Each boot alternates the arms. The first round is cold and is left out.

## Results

| | stock | sublet | difference |
|---|---:|---:|---:|
| instructions of the whole job, mean of 5 warm rounds | 49,501,416 | 49,786,311 | **+0.58 %** (+284,895) |
| spread of those rounds within an arm | 0.10 % | 0.08 % | |
| QEMU wall seconds, median of 8 warm rounds | 4.74 | 5.14 | +8 % |
| range of those rounds | 3.50 to 5.07 | 4.42 to 5.49 | |
| output against SPEC's reference | 15 of 15 match | 15 of 15 match | |

`result-lines.txt` has every run.

The instructions are those of the guest during the job: the compiler, the kernel and
whatever else ran on the one hart. Under `-icount` the count repeats to a tenth of a percent,
so the difference is resolved. The wall time is not: the stock arm alone spreads over 1.6
seconds.

A counting build of the same port, with a tally at each of the six sites, executes 14,798
lifetime instructions on this input. That build is not part of this directory.

| | `CDERIVE` | `CRENEW` | `CREVOKE` |
|---|---:|---:|---:|
| pools | 1,031 | 521 | |
| GC heap | 12,102 | | 157 |
| obstacks | 769 | | 218 |

## Controls

- **The rebuild is faithful.** `replay-build.sh` on an unchanged copy of the build directory
  gives the kit build's executable byte for byte.
- **The port is in the image.** `lifetime-instructions.py` passes `--expect present` on the
  three ported allocator objects and fails it on the stock ones, and the reverse for
  `--expect absent`.
- **The gcc fix is needed.** The port without it stops with cause 25 in
  `forward_propagate_into`, on a reference to a 96-byte pool element.
- **`apply.py` refuses a tree it has already ported.**

## What these numbers do not say

- QEMU wall time is how long the emulator takes. It does not predict processor time.
- The instruction count includes the kit's kernel, and this kit's lifetime table is the
  older paged one. Another kit can move the kernel's share.
- One boot gave these counts. Three earlier boots of the same kit, with an earlier build of
  the port, read +0.6 % twice and +1.1 % once, the last with the stock arm moving by 1.9 %
  between rounds.
- One tiny input ran. Larger inputs, the trap route of this branch's head and the board did
  not.
