# free() rejects pointers it did not issue: directed test, and today's outcomes

The gate for making the Capstone mallocng implementation cheaper without changing its policy (lane branch
alloc-lean). Any change to the malloc/free paths must leave every case below with the same outcome, the same fault cause and the
same faulting instruction, decoded from the binary. The pc itself may move with the code.

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
| double-live | the same pointer twice, a sibling kept live in its group | SIGSEGV | 25 | as double (`lbu`) |
| reused | freed, the slot reallocated (same address) | SIGSEGV | 25 | revoked |
| remapped | a 1 MiB object freed, reallocated at the same address, the old pointer freed | SIGSEGV | 25 | revoked |
| remapped-new | as remapped, the new pointer freed | returns | - | - |

The reuse and remap cases did land at the same address (FREJ_INFO same-address 1), so they test what they claim.

**get_meta's checks.** No killed case above reaches get_meta: each dies in heap_access first (a claim-auditor found
this on 2026-11-14). `double-live` was added for that reason. It keeps a sibling object live, so the group survives
the first free. On the SPATIAL configuration (lifetimes off, no revoke), its second free gets past heap_access, and
get_meta's freed-mask assert rejects it: **cause 24 at `sb`**, the a_crash store through x0. Every other spatial case
reads as in the temporal table.

Results of the extended test, all 11 cases, `judge-frej.py` comparing outcome, cause and mnemonic:

| SDK pair | result |
|---|---|
| temporal: FRCDKMS (perf/memset) against 0012 + 0013 (alloc-lean) | MATCH |
| spatial: perf/memset-spatial against spatial 0012 + 0013 | MATCH, double-live at get_meta's `sb` on both |
The malloc and nodes contract suites do not run on this SDK lineage: no virtual kit at hand runs its images (cause
24 at their first capability access). This test, the hosted contract on the native route, and the perl and gcc
twins are the gate.
