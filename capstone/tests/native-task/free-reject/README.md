# free() rejects pointers it did not issue: directed test, and today's outcomes

The gate for making the Capstone mallocng implementation cheaper without changing its policy (lane branch
alloc-lean). Any change to the malloc/free paths must leave every case below with the same outcome and the same fault
cause. The pc may move with the code.

`frej.c`: a parent spawns one child per case, because a rejection kills its process. It prints each child's exit
status or signal, plus `FREJ_SETUP` before the free under test and `FREJ_RETURNED` after it. The native route logs
one `capstone_task: fault ... cause=N` line per killed child, in spawn order. The judge (local to the build host,
perf/alean/judge-frej.py) pairs the two and compares them against a reference run.
- The judge was checked against a copy of the reference with one cause altered: it reports MISMATCH.
- A run without `FREJ_DONE`, or one whose fault-line count is not the killed-case count, is VOID.

Built with arm FRCDKMS's SDK and flags plus `-fno-builtin`, so that an unused malloc/free pair is not elided. It is
staged as the binary of a copy of the native mcf stage, which copies it to /tmp/nt/sr/.frej-native, inside
dom_grant_root.

## Today's outcomes

Run 2026-11-14 on the native twin, kernel images-a0ed with the board's dom_ flags, 262144 records. Binary
237fa3e354e5fe1e; SDK perf/memset (runtime-memset-cap-zero-arms beb1bbabe26a).

| case | what is freed | outcome | cause | the check that fires (heap_sublet.h heap_access) |
|---|---|---|---|---|
| ok | a live issued pointer | returns | - | - |
| int | an integer address (untagged) | SIGSEGV | 24 | the probe load `lbu zero, 0(p)` |
| interior | cursor 16 bytes past the base | SIGSEGV | 2 | cursor != base: `__builtin_trap` |
| pastend | cursor at the object's end | SIGSEGV | 28 | the probe load, out of bounds |
| global | a global object | SIGSEGV | 2 | trap (cursor != base, or no mapping entry) |
| stack | a stack object | SIGSEGV | 2 | trap (as global) |
| double | the same pointer twice | SIGSEGV | 25 | the probe load on the revoked capability |
| reused | freed, the slot reallocated (same address) | SIGSEGV | 25 | revoked |
| remapped | a 1 MiB object freed, reallocated at the same address, the old pointer freed | SIGSEGV | 25 | revoked |
| remapped-new | as remapped, the new pointer freed | returns | - | - |

The reuse and remap cases did land at the same address (FREJ_INFO same-address 1), so they test what they claim.
The malloc and nodes contract suites do not run on this SDK lineage: no virtual kit at hand runs its images (cause
24 at their first capability access). This test, the hosted contract on the native route, and the perl and gcc
twins are the gate.
