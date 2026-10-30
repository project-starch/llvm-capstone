# The flat ids55 module on the QEMU twin: results

Read against `PREREG.md`: section 1 at 8c05865190f2, addendum 1 at bed4d41f67ff, addendum 2 at e08dfaa95f22. The grid
ran 2026-10-30 01:14-09:41 and the perlbench arm 09:41-14:58. Records are under `/mnt/hdd/<user>/capstone/f55/` (GRID,
PERL, t1-*, live-*, perl-F0). Below are result lines only.

## Twin grid

| arm | Tier 1 | sparse_recycling | live 1000 | live 400000 | live 1100000 | allocation line |
|---|---|---|---|---|---|---|
| W | 34/45 | false | OK | OK, nodes 414,217 | OK, bad=0, nodes 1,138,917 | none |
| F | 34/45 | false | OK | OK, nodes 414,217 | NULL at 1,011,158, rc 3, nodes 1,046,942, 0 faults | CMA, pa b0100000, NON-LINEAR |
| F0 | 34/45 | false | OK | OK, nodes 414,217 | NULL at 1,011,158, rc 3, nodes 1,046,942, 0 faults | CMA, pa b0100000, NON-LINEAR |
| F0s | 35/45 | **true** | not run | not run | not run | buddy, pa 82c00000, linear |

- The same 11 Tier 1 cells fail on W, F and F0: fault_causes, heap-threads, heap_threads, shared_heap_pages_released,
  sparse_recycling, threads, threads_output, vm, vm-large, vm_contract, vm_large_contract. F0s fails the same set less
  sparse_recycling. No cell differs between W and F, or between W and F0.
- sparse-churn on F and F0: nodes 600,041, collections 0, reclaimed 0, node_bytes 16,777,216, capacity 1,048,576. That
  is false by the check's threshold, as predicted. On F0s: the same 600,041 nodes through a 65,536-record table,
  node_bytes 1,048,576, true as predicted.
- Parameters as predicted: F and F0 read ids=55, index_bits=25, node_flat_records=1048576. W has neither width
  parameter.

Against the predictions:

- **1 holds for F.** W's sparse_recycling cell was not predicted. It is false by run.py's metadata bound: node_bytes
  28,672 > 16 × capacity 1,024 (addendum 2).
- **2 to 7 hold.**
- **8 missed on one word.** The CMA table on the twin reads NON-LINEAR, where linear was predicted. The 6.1 guest's DMA
  default is non-coherent, so the buffer is remapped. The board's 6.4.14 is coherent, so the board prediction stands
  (addendum 2, from both kernels' sources). F0s's `linear` on the buddy path shows the check can print either word.

## Perlbench train T3 on F0 (addendum 1)

| command | nodes (F0 = W) | instret F0 | instret W |
|---|---|---|---|
| 0 | 7,893,537 | 137,302,910,926 | 137,302,827,511 |
| 1 | 1,047,762 | 25,248,121,790 | 25,247,961,170 |
| 2 (scrabbl) | 14,753,501 | 67,055,444,900 | 67,055,444,842 |
| 3 | 2,230,218 | 67,941,786,669 | 67,941,518,770 |
| 4 | 998,956 | 3,561,035,152 | 3,561,021,314 |

- 5 of 5 commands return rc 0, and specdiff passes 6 of 6.
- Every command reads node_capacity 1,048,576, collections 0, reclaimed 0 and node_bytes 16,777,216.
- The serial log has no ENOMEM, ALLOCATION FAILED or domain-fault line. As controls, it carries one `flat table` line
  and one `loaded` line.
- **All five predictions hold.** Node counts are equal and instret is within 0.001 %. scrabbl creates 14.1 times the
  table's records on a fixed 2^20 table and completes.
- The stage image was b05dcf06469c2024, hashed directly. The record's BEGIN line has an empty image field because its
  script hashed a directory.

The board boot (section 2) is run by the runtime lane under spec-silicon c9ee688c8629, Addendum 14.
