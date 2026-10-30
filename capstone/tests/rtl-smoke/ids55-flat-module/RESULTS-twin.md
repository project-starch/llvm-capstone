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

## F0r: F0's Tier 1 with resolve_dirty=1 (addendum 3)

- **The prediction holds.** F0r equals F0 in all 45 cells: 34/45, with the same 11 failures, `vm` exit 1, and an
  identical final stats line.
- The twin never takes resolve_dirty's branch, so this shows only that the parameter changes nothing there. The silicon
  evidence for the branch is S-24's r13 reading.
- The record's `END rc=0` is the status of `date`, not of run.py; the reading is result.json, status FAIL.

## Board: bitstream 325966ee4, board 2 (section 2 and addendum 3)

The runtime lane ran the boots: spec-silicon Addendum 14 (c9ee688c8629) and Addendum 17 (cd44551c2ae4). Their result lines
are under `ports/spec-cpu2017/results/2026-10-28-board-test/b4/`. The readings below were re-read from the raw transcripts.

- **Boot 1: module and table.** The module hash was 6422805f, and the launcher was 62753dd6. The allocation line read
  `CMA, pa ac100000, linear mapping`, inside the board's CMA window 0xac000000..0xbbffffff. That is prediction 8's board
  half. The live counts equal F0's.
- **Boot 2: Tier 1, module loaded bare.** 28/45. `vm` read 139 because of S-24: its fix, resolve_dirty, defaults off.
  Section 2 had wrongly taken F0 as the board reference. Addendum 3 repairs that.
- **Boot 3 (b4-t1c): Tier 1 with resolve_dirty=1 resolve_pte=1, kit bkit14-b4r-b2.** The kit's .ko is 6422805f, and
  both parameters read back Y three times.
  - Every cell equals F0r's except the two the addendum pre-registered: cleanup, false on the board by construction, and
    spin_entered, excluded. That is 33/45 against 34/45. The twin column of the runtime lane's judge matches
    `t1-F0r/result.json` in all 45 cells.
  - `VM_EXIT:vm:1`, the twin's status. The S-24 bound line (`... with no change`) is absent. As a control, the same grep
    finds it once in boot 2's transcript.
  - The allocation line again reads CMA, pa ac100000, linear.
  - Transcript: spec.txt sha256 d3dc5c8b0b10be20.

**So the flat ids55 module matches its twin reference on silicon, provided it is loaded with resolve_dirty=1.** B's
kits load it that way. S-24's D=0 PTE value is still unread: with the fix on, the loop never forms.
