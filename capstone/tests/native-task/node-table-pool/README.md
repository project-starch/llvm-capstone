# The flat-table pool (capstone_vm node_table_pool): directed test

`node_table_pool=N` (default 0, off) keeps up to N released flat lifetime tables. Each is zeroed back over
[0, next) on release, where next is the hardware's bump pointer table[1], and the next context takes it instead of a
fresh cma_alloc + 4 MiB memset. The module comment above `cv_table_pool_get` gives the argument that [0, next) is
the whole of what a context can have written: the header table[0..3] included, nothing at or above next. It also
explains why that has to hold for every record, not only those below next. `node_table_pool_verify=1` checks the
argument on every reuse with a whole-table scan, counts a violation, and zeroes the table whole before use.

Why: on perl test.pl (native twin, node_flat_records=262144, 318 execs), each exec's table peaked at a mean of
7,791 records (p50 5,492, max 132,493) of 262,144. Census: perf/census, kernel with a per-context node line.

**What it checks** (`ntp.c`). A parent spawns, four times, a `dirty` child and then a `fresh` child, each its own
process and so its own context.
- `dirty` allocates 6000 32-byte objects (a lifetime record each) and frees every other one, leaving 3000 on the free
  list (table[2]). It then exits with the rest live.
- `fresh` allocates, pattern-checks and frees 6000 objects three times.
- A stale free list makes the fresh child's context_create refuse (node_available reads table[2]), or its first
  allocation fault.
- A stale record that no context names is invisible to the program. The verify scan, read back from sysfs after the
  command (`@@POOL`), is the check for those.

**Controls** (test build `CV_TEST_HOOKS`, `test_pool_mutant`):
- 1 leaves table[2..3] stale;
- 2 leaves record next-1 stale.

**How it runs.**
- The native route, as the SPEC images do. Built with arm FRCDKMS's SDK (llvm-c9) and flags, and staged as the binary
  of a copy of the native mcf stage. The stage copies it to /tmp/nt/sr/.ntp-native: a native spawn cannot name a path
  outside dom_grant_root, as the perl stages' .perl-native copy shows.
- Run on the native twin (images-a0ed's tree with this module), with the board's dom_ flags and
  node_flat_records=262144. The scripts are local to the build host (perf/ntp: kbuild.sh, mkpool.py, chain.sh,
  judge.py, PREDICTIONS.md).

## Results (2026-11-14, twin)

| arm | kernel, parameters | NTP_DONE | hits / puts | violations |
|---|---|---|---|---|
| D0 | pool off | PASS | 0 / 0 | 0 |
| D1 | pool=4, verify | PASS, 8 children exit 0 | 7 / 9 | 0 |
| H0 | test build, pool=4, verify | PASS | 7 / 9 | 0 |
| M1 | test build, mutant 1, verify | PASS (the scan repaired each table) | 7 / 9 | 7 |
| M2 | test build, mutant 2, verify | PASS (no context names the record) | 7 / 9 | 7 |
| M1u | test build, mutant 1, verify off | FAIL: every fresh spawn 1095 = 1000 + EOPNOTSUPP | 7 / 9 | 0 (off) |

Notes on the counts:
- 8 children give 7 hits, because the first child finds the pool empty: the parent's table is live. The prediction
  said 8, which was miscounted and corrected after D1 was read.
- Puts are 9: the children and the parent.
- In M1u, dirty children after the first also fail. A failed fresh spawn leaves its stale table in the pool for the
  next dirty child.

Withdrawn on the way: the first build spawned argv[0] (/tmp/nt/b), outside the grant root. posix_spawn returned
ENOENT, so no child ran; those arms are void.
