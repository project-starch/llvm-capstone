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

## Out-of-sample test: REFUTED (2026-11-23)

The fit above has two parameters on two programs, so it is exactly determined, and its one held-out point sits at
noise level. Its first real out-of-sample prediction was the demand-only record cache: capstone-ariane b5a-demand-only
2ca17ad2a, which removes the background fill from 7180a83b6. That prediction was pre-registered in that commit's
message before lint, sweep, synthesis or any board run.
- **gcc train01:** predicted -1.68 % (-705 M in-context cycles) against 7180a83b6. The decisive refutation criterion
  was "not below -0.5 %". Read: **41,874,434,751, -0.391 %**; against P1 it is -0.035 %.
- **mcf test:** predicted +0.40 %. Read: **144,919,108,858, +0.672 %**, which is slower than P1.
- **HPM:** ev37 is 0 on both, so the change is present.
  - ev34: gcc 8,520,989 against the model's 9.61 M (-11.3 %); mcf 2,756,098 against 2.98 M (-7.5 %).
  - ev35: gcc 7,143,248 (+4.4 %); mcf 178,364,955 (+0.04 %).
  - The count criterion (more than 5 % off) also fires, though the commit said it could fire on calibration alone.

The run lines were re-read by the rtl lane from the board transcripts: the last `@@NATIVE-CNT 0 end` line, with
counters hpm3..hpm8 = ev34, 35, 31, 37, 36, 2, in
`/mnt/hdd/<user>/capstone/spec/board/out-nat-b5adem-hpmrc-frcdkms-b2/{505.mcf_r-native,502.gcc_r-train01-native}/board.txt`.
The outcome is in llvm-capstone spec-silicon 5c5b00610a4d (results/2026-11-08-train/PREREG.md, Addendum 226; the
pre-registration is Addendum 214).

**What is retracted:**
- the model as a tool for choosing levers;
- its reading that B5a's mcf gain (-0.49 %) is read hits minus fill cost.
The fill term (48.6 cycles per fill read) overstates what removing the fill gains on gcc. mcf moved the predicted way,
but further.

**What stands:** the reader's event calibration (mcf within 0.6 % on every event) is a property of the trace replay,
not of the cost fit. B5a's mcf mechanism is open.

**Do not use the cost model for a lever choice until it is refitted on more than two programs, with a held-out point
outside run-to-run noise (which is not measured).**

## Line B (the Sublet format): `rcm-lb.c` (2026-11-26, the rtl lane)

The record-cache decision for B5b-2 is made on EVENTS from this model and a MEASURED per-miss cost. The two-parameter cycle model above was refuted out of sample and is not used. The semantics owner's threshold: build only if misses x the measured cost is at least 1.0 % of cycles on one of gcc, mcf and Perl, with no organisation adding events beyond NC.

**Organisations:**
- LB: b42c8c644's own verdict cache. 64 x 4, set = slot[5:0], full-id tags. Installs on live record writes and live unit reads; same-index reuse, then the lowest free way, then round robin. A release clears liveness in place, so the entry is kept and later denies without a probe. An E event with word 1 bit 0 set empties it (see the format below).
- RC, GC, NC: as in `rcm.c`.
- `rcm-lb.c`'s header has the exact rules and their source lines.

**Trace format v2:** two u64 per event. Word 0 is `rcm.c`'s; word 1 is `{live[32], generation[31:0]}`.
- Line B tags carry the generation, so a reissued slot is a different id.
- Installs depend on whether the record read or written is live.
- **Produced by the nested lane's Sublet tracer** (capstone-qemu nested/record-trace-lb, on the twin's own QEMU 3a9fc39167; off unless CAPSTONE_RECTRACE_V2=<prefix>): checks (c), allocations (a: CDERIVE, CRENEW including its old record's release, CSMINT's create; their operand reads are op reads of kind a), releases (r/v: CREVOKE, including the 4(b) walk's reads), other operations (o), table switches (T) and edges (E).
- **E:** op 0 = a trap out of protected U, 1 = an srevroot write, 2 / 3 = an SRET / MRET into protected U. Word 1 bit 0 = 1 exactly when the key {srevroot, mode != M && srevroot != 0} changed, which is when b42c8c644 flushes (revnode_ns_flush_o; under CAPSTONE_SUBLET_FMT vctx_table is 1 in S and U whenever srevroot != 0, so a U <-> S switch under one srevroot does not flush). The csupctl bit-2 strobe also flushes on the RTL; the twin has no csupctl, and Line B's runtime never writes 0x7C4 (the runtime lane's search, firmware and kernels, by source and by encoding).
- **Raw, not deduplicated, and the twin's walk is not the RTL's.** `rcm.c`'s hardware pattern is NOT ported: it was derived from the rev_node unit, while b42c8c644's Line B unit is `capstone_sublet_life.sv`, which re-reads records on purpose (two passes; re-gets before each splice write) and writes a CRENEW-in-place operand twice (S_K_NODE0 live 0, S_K_NODE1 live 1). The tracer drops the two shapes the unit never makes: an allocation's second write of its record (S_D_WCH, S_C_W, S_K_FRESH write it once) and a re-read of the operand (S_RS_R reads it once). The twin's CREVOKE is the C reference's get/put per link and makes MORE accesses than the unit: the nested lane's host count of one cap_rev_kill_step with prev and stop non-zero reads 5n + 12 records (17, 22, 27 for n = 1, 2, 3) where my reading of the unit's states gives 3n + 6 (not yet measured in RTL simulation), and writes stop twice and the operand live 1 before its release.
- **What that leaves usable.** A re-access of a slot LB already holds reuses its way, so LB's state after an operation depends only on each slot's first-install order and its final (generation, live), unless the operation displaces a slot it installed itself; `LB op_stretches ... self_evict ...` counts those. With self_evict 0, the LB check rows (the decision's input) are the RTL's IF each operation's first-install order and final states match the unit's. COMPARED (2026-11-26): the nested lane's host sequences for CDERIVE (free list empty / non-empty), CSMINT's create, CRENEW in place / out of place, CREVOKE empty and a 2-chunk CREVOKE with its re-execution match the unit on both, operation body by body. The one divergence was the OPEN: every lifetime operation (CDERIVE is two, a check then a commit) reads the first two free-list entries, and the read tap fills them live for their stored next generation (ex_stage.sv:2628-2629), while the twin traced only its own pop read, at live 0. The tracer now emits the open's reads and drops the pop read (capstone-qemu nested/record-trace-lb 702f24732b). The RTL side was MEASURED, not only read: capstone-ariane slu-walk 4bbdbc6ed, b5b2-crevint-walk3 (a one-chunk CREVOKE of 3 makes 15 record reads = 3n + 6; each open reads free[0], free[1]; the allocation does not re-read the popped record). The LSU's install on those reads is by source. LB numbers are usable once a trace from 702f24732b or later reads self_evict 0. The op rows (accesses, memory reads and writes) of every organisation are the TWIN's pattern, not the RTL's.
- No trace has been read yet. Until one has, this tool has no reading.

**Control:** `synth-control-lb.py rcm-lb` runs a synthetic v2 trace whose LB, RC, GC and NC counts were computed by hand, and asserts each one.
- Its cases: eviction past 4 ways, the round-robin victim, a release in place, a hit-dead deny, a same-index reissue at generation 2, a namespace edge, and a dead id's one probe.
- `--perturb` changes one expectation, and the control must then FAIL. Both were checked: PASS rc 0; perturbed FAIL rc 1.
- The open phase: a free entry read live at its next generation turns a later check of the stale id from a deny into a probe (probes 1, deny_hit_dead 1); at live 0, the twin's former shape, it stays a deny (0, 2). A build that does not fill on op reads FAILS.
- self_evict: five installs of one set in one stretch count 1; four, or five split by a check, count 0. Builds that never count it, or never end a stretch, FAIL.
- An edge with the key unchanged must not flush: a build that flushes on every E FAILS the control on six counts.
- A malformed trace exits 2: six malformed traces (E word 1 = 2; T, I, PI with a non-zero word 1; a record with bit 33; kind 11) must each exit 2, and E word 1 = 1 must exit 0. f82523965196 checked word 1 after the marker branches, so the four marker cases were accepted; c156970f12cd fixed it and its build passes.

**Measured per-miss cost** (RTL simulation, b42c8c644's core, memory delay 12; capstone-ariane b5b2-prefix bd3932800, b5b2-srvcost and -DSC_COLD):
- warm record (in the D$): (P3 - P2) / 33 = 14.4 cycles per verdict miss;
- cold record (D$ evicted): (P3 - P4) / 33 = 70.0 cycles, and independently (P7 - P2) / 33 = 71.5 cycles (P7's probes read records evicted before P4).
- These are simulation costs at a fixed 12-cycle memory. The silicon cost comes from the same images run as a bare kit on the board, as the matched pair.
