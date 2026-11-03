# Mapping registry as an AVL tree: measurements and gates against dev

The launcher (`exec.c`) kept its mappings in an array sorted by address, highest first, and moved the
tail on every insertion and removal. This change keeps them in an AVL tree of nodes (`struct
map_node`), so `map_floor`, `map_find`, `map_insert` and `map_remove` each walk one root-to-leaf path.
Only the launcher differs between the arms below; module, QEMU, firmware, kernel, rootfs and the
application images are the same files.

- control: dev 976df58996f3's launcher, `capstone-vexec` 82822c5d176aa4d3 (the cross build of dev
  reproduces the profiling platform's launcher byte for byte).
- this change: `capstone-vexec` d0d06d2bb0a2055a, built by `build-adapter.sh` with the same
  buildroot `riscv64-buildroot-linux-gnu-gcc`.

## Why: the array moved its tail on every unmap of a large heap

A diagnostic launcher (dev plus counters, not committed) on PHP 5.0.0's own bench.php tests, the
per-test sizes of the PHP benchmark lane (`capstone/ports/php/bench/`), Capstone arm:

| test | heap maps | unmaps | peak live mappings | entries moved by removals |
|---|---:|---:|---:|---:|
| ary(100000) | 10,470 | 10,296 | 10,462 | 48,969,423 |
| ary2(150000) | 15,620 | 15,385 | 15,610 | 112,956,848 |
| hash1(100000) | 5,406 | 5,290 | 5,402 | 13,275,542 |

15,072 of ary2's grants are one page: mallocng groups, each its own mapping. The test builds its
arrays (no removals) and then frees them in allocation order. New mappings are placed below the
previous ones, so the oldest sat at the far end of the array and each removal moved almost all of
it: 20.3 instruction attempts per moved 48-byte entry, 2,288,454,930 attempts in
`_wordcopy_fwd_aligned` (memmove) for ary2.

## Results (capprof attempts, QEMU `-icount shift=0`, same-time control, all PASS)

PHP per-test windows (one VM per arm, image sha256 cd2bd47e):

| test | control | this change | B - A | Capstone / plain, control -> this change |
|---|---:|---:|---:|---|
| ackermann(6) | 2,754,265,052 | 2,753,959,749 | -305,303 (-0.01 %) | 2.410 -> 2.410 |
| ary(100000) | 3,013,498,819 | 1,989,469,004 | -1,024,029,815 (-33.98 %) | 4.559 -> 3.010 |
| ary2(150000) | 5,240,124,771 | 2,908,183,848 | -2,331,940,923 (-44.50 %) | 6.522 -> 3.620 |
| hash1(100000) | 1,999,131,901 | 1,727,233,369 | -271,898,532 (-13.60 %) | 2.492 -> 2.153 |

The launcher class of ary2 goes from 2,369,984,249 to 32,678,012 attempts. The plain-arm
denominators are the PHP benchmark lane's per-test plain windows (661,000,051 ary,
803,441,146 ary2, 802,179,112 hash1, 1,142,829,519 ackermann).

bench.php reduced to about a fifth (ary/ary2/hash1 at 10000): 22,576,743,266 -> 22,529,305,506
(-47,437,760, -0.21 %), launcher class 83,355,755 -> 35,022,220. At these small sizes the
registry is a small term; printed per-test guest seconds change only for ary 0.173 -> 0.163,
ary2 0.148 -> 0.138, hash1 0.144 -> 0.142 and sieve 0.742 -> 0.717.

SPEC CPU2017 test inputs (kit3 `profile-batch.sh`, same-time control on platform-dev3), launcher
class B - A: 502.gcc_r +7,583, 538.imagick_r -13,541, 519.lbm_r -9,956, 544.nab_r +3,075 attempts.
The program class differs by at most 37 attempts; the totals move within the kernel and idle-loop
noise of the platform (control against control: imagick 209,799,045 and 206,118,177).

## Gates

p13 `~/retire-gate/run-arm.sh`, control then this change, sequentially, on two real-file copies of
the gate platform (`find -type l` empty) that differ only in `capstone-vexec`; the module is dev's
with `supervisor_slots = 32` (bceb7055), the hooks module dev's with `CV_TEST_HOOKS` (804819fa);
the ten gate images are sha-identical between the arms.

| gate | control | this change |
|---|---|---|
| hosted | rc=0, PASS | identical, 0 JSON fields differ |
| runtime | rc=1; false: perl, perl_output, mruby, mruby_results, sqlite_read/write/results, linux_demand_faults, linux_pages_released | identical set, 0 JSON fields differ |
| nodes | rc=1, FAIL node_count_reported (15 PASS) | identical |
| malloc | rc=1, false: real_linux_mremap_growth_and_shrink | identical, 0 JSON fields differ |
| fp, pthread | rc=0, PASS | identical, 0 JSON fields differ |
| memcached | rc=1, FAIL 76 / PASS 29 | identical 76 |
| pin-ahead | rc=1, FAIL 4 | rc=1, FAIL 7 (different names) |
| step-hooks | rc=1, FAIL 15 | rc=1, FAIL 6 (different names) |

What did not run: the runtime gate's sqlite_write, sqlite_read, mruby and perl rows exit 126
(`VM_EXIT:*:126`, "capstone-vexec: image: Exec format error", stale kit images) in both arms, and
28 of memcached's 34 guest server runs (its pthread case among them) end in the same error in both
arms; only its 6 native host runs and harness checks ran. `vm-mappings` ran in both
(`VIRTUAL_VM_MAPPINGS_OK 9000`, `VIRTUAL_VM_MAPPINGS_MOVED 8`).

pin-ahead and step-hooks are not deterministic on dev's runtime with guest ASLR on: three more
runs of each arm gave pin-ahead FAIL 6, 7, 6 (control) and 6, 4, 10 (this change), step-hooks
FAIL 9, 7, 8 and 6, 6, 6, each with a different set of window checks (`w*_seq/stop/vma/vmax/retire`,
`*_walk_accounting`, `*_syscalls_accounting`). With guest ASLR off (`randomize_va_space=0`, the
runners patched as lane K's recipe), three runs per arm:

| gate | control | this change |
|---|---|---|
| pin-ahead | FAIL 4 (all w64), 0, 0 | 0, 0, FAIL 5 (all w16) |
| step-hooks | 0, 0, FAIL 3 (resched4/signal4/deadline4_walk_accounting) | 0, 0, 0 |

So both arms pass both gates in most runs, and each arm's one failing run is one window's set of
checks, a class the control shows as well. No difference attributable to the launcher.

## Host test

`test-registry.sh` (host, no module or hardware) runs the real registry functions from `exec.c`:
200,000 random insertions, removals, moves (remove then insert without a reserve, as
`remap_mapping` does) and lookups against a model, after each step the floor of a random address,
the count, an in-order walk (strict order, every node's height and balance), and pointer
stability; then top-down, oldest-first, near-the-end and bottom-up sweeps and 20,000 top-down
registrations whose height must stay within 1.44 log2(n + 2). It passes on this change (also under
ASan and UBSan) and fails on eleven broken copies: a flipped or 32-bit floor comparison, no
rebalance, no double rotation, an inverted double-rotation test, a swapped height update in the
rotation, the wrong child kept for a one-child node, swapped subtrees in the two-children case, a
count that is not decremented, and a free list that is not popped or not pushed.
