# Node floor: the QEMU gate, read against PREREG-node-floor-gate.md

QEMU 7a4f3a64776f0c88, svadu=false, tm-kit images. Control: dev's module 057846f74a993748 (banner srevroot-zero-2);
node floor: 43ddff418388992e (banner node-floor-1).

| step | reading | prediction |
|---|---|---|
| 1 Tier 1, two-sided in one session | both 30/45; status lines equal (the same 15 FALSE, the same 24 exits); banner 2, 'wrote no event record' 0, 'collect skipped' 2 | holds |
| 2 chain, 15 arms (qkit7-nf) | against gate6-t3-srzdev: 15 arms, 0 differ, 0 no-data; images equal (verify); refused lines 0; all 16 consoles carry node-floor-1 (pb13 VOID rc 75, its 9 NO-READING as in every earlier gate) | holds |
| mpt | equal to gate6-t3-srzdev's | holds |
| 3 counters, both modules | (426,536,0) on both; the CAPSTONE_VM_STATS lines equal but for timings (nodes=76, node_capacity=65536, steps=537, faults=426); `--stats` equal: nodes_high_water 76, nodes_allocated_total 76, nodes_live 0, nodes_retired 0 | holds: nodes_high_water is unchanged, not shifted |
| 4 step hooks (bcdd760fa1b605fc) | 210 PASS, 0 FAIL | holds |
| 5 frame tests (qkit7-nfs2) | multipage -22; twice first=1, second -16, child_ran=1; copy cause 25; overlap PASS; 0 halts | holds |
| 6 dense-churn | (a) control at 4096: last derived=3000; (b) control at 3072: 2000; (c) node floor at 4096: 2000. Each: vm-contract 0 first, collect skipped once, cause 30 at the CDERIVE (offset 0x15bd4, image 9c41c82a), exit 139; node_flat_records read back | holds: (b) = (c), neither 3000 |

QEMU cannot show whether a physical id above 1024 exists on the board, which would defeat the floor. The board's
Tier 1 boot (board module a4aa5668a69b94ee, control 39d8492d5c18216b) and sql0 on tree B against sql0-srz-treeB, with
FRTY predicted 0, are the test.

## Addendum 1: node-floor-2 (c2b843ad88ad), its QEMU Tier 1 pass

- Module a1fa252f527e5a43, banner node-floor-2.
- Status lines equal node-floor-1's: 30/45, the same 15 FALSE and 24 exits; banner 2, 'wrote no event record' 0,
  'collect skipped' 2.
- 26 release lines on the guest console. Every one reads succeeded = attempted (9 to 17 retires per context) and
  first_id=1024.

Every prediction holds. Board module: 54d8a3af1158c3f4.
