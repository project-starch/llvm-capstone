# Node floor (lane runtime/vp2-node-floor 770daec2dbd6): QEMU gate, pre-registered 2026-10-20 before it runs

The change: in the flat build, a context's table starts allocating at CV_NODE_FIRST = min(1024, records/4) instead of
2, so no virtual index a context hands out can equal a low physical index (S-23's fetch face: the CSRETIRE of
a->ancestor broadcasts virtual index 2 and clears caller CPMP entries tracking physical node 2). The creation check
and nodes_high_water count from the floor. The paged build keeps 2.

## Builds (control: dev's module, source 528bc48bba1f6315, the srevroot-zero landing)

| | control (srzdev) | node floor |
|---|---|---|
| source | 528bc48bba1f6315 | ed13010c78f12db9 |
| QEMU kit | 057846f74a993748 | 43ddff418388992e |
| board | 90bf036f59da1231 | a4aa5668a69b94ee |
| test hooks | - | bcdd760fa1b605fc |

Board census, node floor:
- 4,659 instructions, 0 RVC, vermagic 6.4.14, 0 QEMU-only CSR;
- silicon gate 0 refused / 0 candidates / 0 attributed / 4 conditional / 0 unimp, as the control's;
- srz-check: CSRUNV 3, each followed by `csrw 0x5c1, zero` (srevroot-zero kept);
- banner `node-floor-1` in the image.

## The arithmetic

- node_available = capacity - table[1] + recycled, and table[1] starts at CV_NODE_FIRST. So a table of R records has
  R - F available instead of R - 2, where F = min(1024, R/4).
  - R = 65,536 (the default): F = 1024, and 1,022 fewer records.
  - R = 4,096 (dense-churn): F = 1024, also 1,022 fewer.
- nodes_high_water = table[1] - CV_NODE_FIRST, and table[1] starts at CV_NODE_FIRST. So it still counts the records
  allocated, and is predicted UNCHANGED, not shifted.
- The image-grant check (2 + 256 records) still passes at R = 4,096 (3,072 available).

## Predictions

All on QEMU 7a4f3a64776f0c88 with svadu=false and the tm-kit images (built from the corpus SDK).

1. **Tier 1, two-sided in one session** (control, then node floor): both 30/45, the same 15 FALSE and the same 24 exits
   in order; banner 2 (`srevroot-zero-2` / `node-floor-1`), 'wrote no event record' 0, 'collect skipped' 2.
2. **The 15-arm chain** on qkit7-nf against gate6-t3-srzdev: every arm's images equal; 0 differ; refused lines 0;
   every console carries the node-floor-1 banner. **mpt** equals gate6-t3-srzdev's.
3. **Counters, both modules** (contract `ok`):
   - baseline and no-context (0,0,0); `ok` resume_sfences = faults, satp_refreshes = steps - 1, satp_changed 0;
   - the CAPSTONE_VM_STATS line equal between the modules except launch_ns and elapsed_ns. In particular nodes and
     node_capacity (65536) are equal;
   - `capstone-vexec --stats`: nodes_high_water, nodes_live, nodes_retired, nodes_allocated_total and node_capacity
     equal between the modules.
4. **Step hooks** (test-hooks module): 210 PASS, 0 FAIL, including malloc16_ok_exit, malloc64_ok_exit and
   no_contract_failure.
5. **Frame tests** (two slots, qkit7-nfs2): multipage -22; twice first > 0, second -16, child_ran=1; copy cause 25;
   overlap PASS; 0 halts.
6. **dense-churn, three arms**, each starting with vm-contract `ok` exit 0, with node_flat_records read back:
   - (a) control at 4096: collect skipped once; cause 30 at the CDERIVE; last progress line derived=3000 (as on the v6
     gate's dense-churn).
   - (b) control at 3072: 3,070 available.
   - (c) node floor at 4096: 4,096 - 1,024 = 3,072 available.

   (b) and (c) differ by 2 available records, so both collect skipped once and end in cause 30 at the CDERIVE. With
   N as (a)'s exact derive count (unprinted, 3,000 to 3,999 if (a) reads 3000):
   - (c) derives N - 1,022 and reads 2000 when N >= 3,022, else 1000;
   - (b) derives N - 1,024 and reads 2000 when N >= 3,024, else 1000.

   They read the same last line unless N is 3,022 or 3,023. Prediction: (b) = (c), each 1000 or 2000, never 3000.

What QEMU cannot show: whether a physical id above 1024 exists on the board, which would defeat the floor. The board's
Tier 1 boot and sql0 on tree B against sql0-srz-treeB (FRTY predicted 0) are the test of the mechanism.

## Addendum 1, before its run: node-floor-2 (lane runtime/vp2-node-floor c2b843ad88ad)

770daec2 plus a corrected comment and one observation-only KERN_EMERG line per context at teardown (flat build only):
`capstone_vm: release retires attempted=N succeeded=M first_id=F`. attempted counts the arenas CSRETIREd at
teardown (those not already ancestor_retired), and succeeded is the change in retired_flat, which rises by exactly 1
per CSRETIRE returning status 0. Banner node-floor-2.

Builds: source 7242996f59be29d3; QEMU a1fa252f527e5a43; board 54d8a3af1158c3f4. Board census: 4,670 instructions,
0 RVC, vermagic 6.4.14, srz-check 3 of 3, silicon gate 0 refused / 4 conditional, 0 QEMU-only CSR. The format string
is present in both builds.

**Prediction, one QEMU Tier 1 pass** (same images and QEMU as above):
- the status lines equal gate6-t1-nf's (30/45, the same 15 FALSE and 24 exits), with banner node-floor-2;
- at least one release line on the guest console;
- every release line has succeeded = attempted (QEMU accepts every CSRETIRE) and first_id=1024.

A rejected-retire teardown cannot be produced on QEMU, so this pass shows only that the line prints and counts. The
board's sql0 on tree B is where succeeded < attempted would show.
