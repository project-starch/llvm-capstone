# record-cache-model: replay record-access traces through candidate caches (programme item 4 (i), 2026-11-20)

`rcm.c` replays a record-access trace of the tracing twin QEMU (the nested lane's tracer; format in its
`rectrace-read.py`: one little-endian u64 per run, `code[63:56] = kind << 2 | op`, `count[55:32]`, `slot[31:0]`) through
five organisations in one pass, each with its own 16-entry LRU record TLB over 256-record pages:

| name | organisation | source |
|---|---|---|
| P1 | 64 sets x 4 ways of ids, set = slot[5:0], round robin; a check miss reads the record; every record write invalidates | capstone-ariane d83d3a633 `load_store_unit.sv:277-278, :1251, :1311, :1375-1377` |
| B5 | P1's cache in front of a 64 x 2 LRU record cache of 4-record lines, per-record valid, write-through, no write-allocate, full fills, slots 0-1 uncached | B5a 6e82d7bd2 / 7180a83b6 `ex_stage.sv:2203-2232` |
| RC | the paper's 5.2 cache: 8 KiB, 2 ways, 64-byte lines of 4 records, LRU, write-back, shared by checks and operations | paper notes `record-cache-design-2026-11-13.txt` |
| GC | 512 generations, 4 ways, set = slot[6:0], LRU; checks only; operations go to memory | paper notes `generation-cache-design-2026-11-12.txt` |
| NC | no cache | |

**Hardware pattern (the default; `--raw` turns it off).** The twin writes each allocated record and each released record
once more than hardware needs, and re-reads records within one operation (the nested lane's reading of the twin's
`cap_rev_table.c` / `cap_rev_tree.c`; the RTL unit reads each record an operation touches once,
`capstone_rev_node.anvil` derive_req / kill_req / REVOKE_NODE). A record write to the slot of the previous record write
by the same operation kind, and a repeated read of a slot within one operation, reach no organisation. On gcc train01
that drops exactly 2,186,704 writes (= allocations + releases, the nested lane's count) and 7,579,488 reads.
Flushes: T lines and fresh tables (`ahw 8` then `aw 1024`) flush everything; context edges (kind 10) flush P1 and B5,
as B5a's namespace flush does. PCC checks are excluded (all slot 1024 on these traces).

**Controls.** `synth-control.py` writes a synthetic trace whose every counter was computed by hand (LRU eviction,
write-back, fills, invalidation on write, a run, a fresh-table flush); the reader reproduces the nested lane's `.stats`
event and run counts exactly on every trace used here.

## Calibration against silicon (B5 vs board 2, B5a + R-76 7180a83b6, the board's own FRCDKMS images)

| | mcf test (3d7e62118636) model / silicon | gcc train01 (3ed5c0be9791) model / silicon |
|---|---|---|
| ev34 read hits | 135.34 M / 134.98 M | 11.06 M / 9.75 M |
| ev35 read misses | 45.91 M / 46.14 M | 5.39 M / 5.91 M |
| ev37 fill reads | 137.72 M / 137.49 M | 16.17 M / 14.11 M |
| ev31 record accesses | 183.98 M / 184.04 M | 27.78 M / 27.20 M |
| hit rate | 74.67 % / 74.53 % | 67.2 % / 62.3 % |

mcf calibrates within 0.6 % on every event. gcc's residual has the sign of B5a's fill YIELD (a fill is dropped at the
unit's next request), which the model does not have (it fills whole lines: an upper bound), and the twin counts 6,956
context runs where the board counted 15,040. Writes: model 6.22 M, silicon (ev31 - ev35 - ev37) 7.18 M, unexplained
(header writes sent as slot-0/1 records are a candidate).

## Cycle model (two parameters, one design)

In-context run_cycles on board 2, stages board-frcdkms: P1 = the cq-b2 boots (2026-11-17), B5a + R-76 = the 7180a83b6
boots (2026-11-20), HPM events from the same 7180a83b6 boots. dC = -s * ev34 + f * ev37, fitted on mcf and gcc:

| | dC (B5a+R76 - P1) | fit / prediction |
|---|---|---|
| mcf test | -734,016,061 (-0.507 %) | fitted |
| gcc train01 | +149,735,821 (+0.357 %) | fitted |
| perl makerand (held out) | +3,594,981 (+0.051 %) | predicted +3,232,737 (+0.046 %) |

s = 54.9 cycles saved per record-cache read hit, f = 48.6 cycles per fill read. The held-out point agrees in sign and
size, but its effect is 0.05 % and the run-to-run noise of in-context cycles is not measured here, so it is CONSISTENT,
not a validation. The parameters describe B5a's organisation (a record cache behind a verdict cache); they do NOT
transfer to RC or GC, which change the pipeline. RC and GC wait for item 4 (ii)'s own calibration.

## Verdict on B5a's geometry (the programme's rule: a gain on at least one program and no loss on any)

Silicon already shows the loss on gcc (+0.357 %), and the model reproduces it. No RTL for this geometry. The paper's perl
row (the full test input) has no silicon point: board 2's perl row is makerand only.

`table.txt`: every organisation per trace, record-path reads and writes per 1k protected instructions, with each trace's
image hash. Traces: the nested lane's `rectrace/traces/` (INDEX.md there names each by its image's sha256).
