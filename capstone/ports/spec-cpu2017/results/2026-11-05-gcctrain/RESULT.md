# gcc train (train01), Capstone Temporal against Plain, with every counter: result

Plain: board 1, 17:06-18:13. Temporal: board 2, 17:42-19:26. Both on C.5a, kit bkit26-gt. The Temporal arm used the
18-bit module 65ede944a87ea742 (Addendum 2). Grade in `grade.txt`, instruction mix (QEMU) in `mix.txt`. N = 1 per arm.

## Verdicts

| # | Prediction | Reading | Verdict |
|---|---|---|---|
| G0 | rc, specdiff, control, fresh, params, QNT0 | both arms | holds |
| G1 | counters programmed | in-context hpm3-7 in (0, cycles); machine-wide > 0 in both | holds |
| G2 | Temporal 3,000-6,500 s; Plain 700-2,500 s | Temporal 5,562 s; Plain 3,516 s | Temporal holds; **Plain FAILS** |

**G2 failed on Plain.** The window assumed about 3 cycles per instruction. Plain read 6.2 (IPC 0.162), with 52.5 L1I
misses per 1,000 instructions. gcc misses in the instruction cache even without Capstone.

## Capstone ÷ Plain: 1.58× (test: 3.22×)

| | Temporal | Plain | Ratio |
|---|---|---|---|
| run time | 5,562 s | 3,516 s | **1.58×** |
| program instructions (QEMU twin) | 14.97 G | 10.89 G | 1.38× |
| kernel instructions (QEMU twin) | 2.45 G | 0.82 G | 3.0× |
| machine-wide instructions (board) | 20.73 G | 14.17 G | 1.46× |
| machine-wide cycles | 5,555 s | 3,508 s | 1.58× |
| inside the program / Plain user | 3,810 s | 2,865 s | 1.33× |
| outside / Plain real − user | 1,752 s | 650 s (sys 266) | 2.7× |

Machine-wide counters per 1,000 instructions, Temporal / Plain:
- L1I miss 70.1 / 52.5; L1D miss 25.5 / 25.0; DTLB miss 1.12 / 1.53;
- mispredict 40.8 / 36.7; issue stall 1,924 / 1,794; scoreboard full 167 / 366.
- In absolute terms, L1I misses are 1.45 G against 0.74 G (1.95×).
- Inside the context alone: L1I miss 57.2, L1D 27.3, DTLB 1.40, mispredict 41.3, issue stall 2,186.

Pages and faults: Temporal peak 48,743 pages and 95,400 faults; Plain minflt 89,080 and maxrss 98.9 MB.

## Where the outside time goes: 155,699 exits

- **Exits:** 95,597 faults, 55,854 system calls, 4,248 quantum exits, over 77,246 CV_STEPs.
- **Outside time per exit: 11.3 ms**, split as follows:
  - **6.96 ms between CV_STEPs**: the launcher, the calls it makes for the context, interrupts and other tasks.
    That is about 1,083 s, half of the 2,047 s gap.
  - **4.02 ms in the CV_STEP kernel path**, of which 1.61 ms is fault resolution.
  - **1.14 ms in other ioctls.**
- The parts measure separately and overlap slightly: they sum to 12.1 ms per exit against 11.3 ms.
- **Not separated:** how much of "between" is work Plain also does. Plain's whole system time is 266 s for the same
  program's faults and calls.
- teardown: 42,986 pages, 22.3 s (sweep 3). Launch: 2.22 s.

## What follows

- **The launcher round trip is the largest single cost** on gcc train, about 14 ms per return to the launcher.
  Candidates, to pre-register:
  - serve the frequent system calls in the module, without returning to the launcher;
  - find what the launcher does per call.
- **Instruction cache:** 1.95× the L1I misses for 1.46× the instructions. Code size (no compressed capability
  instructions) is the lever.
