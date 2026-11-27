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
- **Raw, not deduplicated, and the twin's walk is not the RTL's.** `rcm.c`'s hardware pattern is NOT ported: it was derived from the rev_node unit, while b42c8c644's Line B unit is `capstone_sublet_life.sv`, which re-reads records on purpose (two passes; re-gets before each splice write) and writes a CRENEW-in-place operand twice (S_K_NODE0 live 0, S_K_NODE1 live 1). The tracer drops the two shapes the unit never makes: an allocation's second write of its record (S_D_WCH, S_C_W, S_K_FRESH write it once) and a re-read of the operand (S_RS_R reads it once). The twin's CREVOKE is the C reference's get/put per link and makes MORE accesses than the unit: the nested lane's host count of one cap_rev_kill_step with prev and stop non-zero reads 5n + 12 records (17, 22, 27 for n = 1, 2, 3) where the unit makes 3n + 6 (measured for n = 3, below), and writes stop twice and the operand live 1 before its release.
- **What that leaves usable.** A re-access of a slot LB already holds reuses its way, so LB's state after an operation depends only on each slot's first-install order and its final (generation, live), unless the operation displaces a slot it installed itself; `LB op_stretches ... self_evict ...` counts those. With self_evict 0, the LB check rows (the decision's input) are the RTL's IF each operation's first-install order and final states match the unit's. COMPARED (2026-11-26): the nested lane's host sequences for CDERIVE (free list empty / non-empty), CSMINT's create, CRENEW in place / out of place, CREVOKE empty and a 2-chunk CREVOKE with its re-execution match the unit on both, operation body by body. The one divergence was the OPEN: every lifetime operation (CDERIVE is two, a check then a commit) reads the first two free-list entries, and the read tap fills them live for their stored next generation (ex_stage.sv:2628-2629), while the twin traced only its own pop read, at live 0. The tracer now emits the open's reads and drops the pop read (capstone-qemu nested/record-trace-lb 702f24732b). The RTL side was MEASURED, not only read: capstone-ariane slu-walk 4bbdbc6ed, b5b2-crevint-walk3 (a one-chunk CREVOKE of 3 makes 15 record reads = 3n + 6; each open reads free[0], free[1]; the allocation does not re-read the popped record). The LSU's install on those reads is by source. The self_evict-0 condition I set for using LB cannot be met by any trace: self_evict counts per STRETCH (operations with no check between them), and a stretch that installs more than 4 slots of one set displaces its own installs on the RTL too. The first trace read 203,897. So LB is read as a BRACKET instead: `--walk dedup` replays each stretch with every slot once, at its first live access, carrying its final state (an install at its last live generation, then an invalidate if a dead write follows). That walk keeps the first-install order and drops every re-read, so a slot displaced mid-stretch is never re-installed. `--walk asis` (the default) replays the twin's re-reads. The unit's 3n + 6 walk lies between the two in re-reads; the two LB check rows are a sensitivity bracket over that, NOT a bound over every order. Shuffled or reversed orders were not used, because they change the first-install order, which the comparison above fixed. A decision whose threshold falls inside the bracket is UNDECIDED. The op rows (accesses, memory reads and writes) of every organisation are the TWIN's pattern, not the RTL's.
- **First trace** (the nested lane's `small` arm on 702f24732b, trace sha256 c961ee6a79946085…, 23,957,851 pairs, checked here): LB check misses 20,309 asis and 19,615 dedup (3.4 % apart), over 32,861,650 checks; instructions_protected 114,582,873, instructions_all 4,763,859,720. self_evict is 203,897 asis and 79,181 dedup; by installing kind, asis a 1,716 / r 123,215 / v 78,966, and dedup a 316 / r 16 / v 78,849. So dedup's remainder is the CREVOKE walk's verify reads, which touch more than 4 slots of a set once each. deny_hit_dead and dead_record are 0: no check in this arm reads a dead id. asis reproduces 664d6a6466a2's output line for line, plus the new kind line. The share of cycles needs the program's cycle count, which is not this tool's, and the silicon per-miss cost, now read (below).

**Control:** `synth-control-lb.py rcm-lb` runs a synthetic v2 trace whose LB, RC, GC and NC counts were computed by hand, and asserts each one.
- Its cases: eviction past 4 ways, the round-robin victim, a release in place, a hit-dead deny, a same-index reissue at generation 2, a namespace edge, and a dead id's one probe.
- `--perturb` changes one expectation, and the control must then FAIL. Both were checked: PASS rc 0; perturbed FAIL rc 1.
- The open phase: a free entry read live at its next generation turns a later check of the stale id from a deny into a probe (probes 1, deny_hit_dead 1); at live 0, the twin's former shape, it stays a deny (0, 2). A build that does not fill on op reads FAILS.
- self_evict: five installs of one set in one stretch count 1; four, or five split by a check, count 0. Builds that never count it, or never end a stretch, FAIL.
- `--walk`: a stretch installs s1, displaces it with four more installs of its set, then re-reads it; a check of s1 follows. asis hits (probes 1, self_evict 2) and dedup misses (probes 2, self_evict 1). A build whose `--walk dedup` replays as asis FAILS on three counts.
- `--lb SETSxWAYS` (the verdict-cache size; default 64x4, the RTL's): installs separated by checks, then a check of
  slot 1, under each geometry.
  - Expected probes: 64x4 2, 64x8 1, 128x4 1 for one slot list; 64x4 1, 32x4 2 for a second.
  - Six malformed geometries must exit 2.
  - A build that parses `--lb`, reports the geometry asked for, and models 64x4 FAILS on three counts.
  - On `small`, the default and an explicit `--lb 64x4` reproduce the previous build's output line for line, plus
    the new `LB geometry` line.
- An edge with the key unchanged must not flush: a build that flushes on every E FAILS the control on six counts.
- A malformed trace exits 2: six malformed traces (E word 1 = 2; T, I, PI with a non-zero word 1; a record with bit 33; kind 11) must each exit 2, and E word 1 = 1 must exit 0. f82523965196 checked word 1 after the marker branches, so the four marker cases were accepted; c156970f12cd fixed it and its build passes.

**The numerator, ruled by the semantics owner (the runtime lane, 2026-11-26, second ruling):**
- NO BUILD may use the CEILING: all LB check misses x the cold cost (the higher). If removing every miss saves under
  1 %, no organisation can pass.
- BUILD uses only the misses REMOVED by a named candidate x the warm cost (the lower), MINUS the candidate's own added
  cost.
- For RC, that added cost is RC's own misses x RC's miss cost: one 64-byte line fill, unmeasured. So an RC BUILD stays
  UNDECIDED until that cost is bounded, first in simulation and then by one silicon reading, as for the LB pair. LB's
  per-miss cost is NOT a proxy for it.
- An LB-size candidate adds no new miss type, so its BUILD needs only removed misses x warm.
- A program without its board row is UNDECIDED, and no estimate is written for it.
- **A candidate-specific NO BUILD** (third ruling, 2026-11-26): the candidate's removed misses x the COLD cost, minus
  the candidate's own misses x c at c's PROVEN lower bound. Under 1 % is NO BUILD for that candidate. This is the
  candidate's best case, so every reading agrees.
  - Until c is bounded, its lower bound is 0.
  - A lower bound from simulation counts once it is checked against the unit's read waits. It is written, with its
    source, beside the verdict.
- **Residency:** the full-stream D$-residency pass waits for the program's board run time. It is asked of the nested
  lane only if the run lands in the UNDECIDED band.

**Verdicts, per program** (rule as ruled above; each row carries its cost pair and its denominator's source):

| program | trace (nested lane, twin) | LB check misses asis / dedup | cycle denominator (board row) | share at cold 40.4 | share at warm 15.6 | verdict |
|---|---|---|---|---|---|---|
| 502.gcc_r test (t1.c) | spec-gcc.2262833, sha256 0cfc265bfa31dbb2..., recomputed here | 8,921 / 8,921 (self_evict 0) | 144.0 M = 5.76 s x 25 MHz | 0.250 % | 0.097 % | **NO BUILD** |
| 505.mcf_r test (inp.in) | spec-mcf.2201746, sha256 633ca24aabba3062..., recomputed here | 181,127,530 / 181,127,654 (self_evict 124) | 165.03 G = 6,601.15 s x 25 MHz (board 3, B5TL1-g1b 539b2ef41ee4d444) | 4.434 % | 1.712 % | **UNDECIDED**: RC. LB 1024x8 candidate NO BUILD (2026-11-27, below; tick placement open); every smaller LB size NO BUILD |
| 500.perlbench_r test (makerand.pl, the campaign's g4mk stage) | spec-perl3.2883687, sha256 1af7b0c90ab7cb56..., recomputed here | 277,290 / 277,290 (self_evict 0) | 6.8925 G = 275.70 s x 25 MHz (board 2, CF-B5TL1-g4mk e82d42ebe558cb18) | 0.163 % | 0.063 % | **NO BUILD** |

**505.mcf_r test (inp.in): UNDECIDED on its board row** (2026-11-26, below): two candidates remain, LB 1024x8 and RC.
- The trace is the nested lane's spec-mcf.2201746: 65,420,275,952 B, sha256 633ca24aabba3062..., recomputed here. It
  has 4,088,767,247 events and 61 tables.
- The image: mcf_r_base.native a1daec395a9e7260..., which equals board-stages B5TL1/505.mcf_r (recomputed here) and
  B5TL1-g1b's overlay (the board lane; payload 539b2ef41ee4d444...).
- LB check misses at 64x4: 181,127,530 asis (dev 6d87efe8d50a's build, reproducing the nested lane's
  b35fb24d8e97-built number exactly) and 181,127,654 dedup (the nested lane), over 10,044,639,808 checks. self_evict
  is 124, so the bracket is closed.
- The other organisations' check misses: RC 44,836,912; GC 177,942,095.
- By size (asis): 128x4 178,924,715; 256x4 173,320,429; 64x8 179,088,405; 128x8 173,978,679; 1024x8 = 4096x32
  126,537,860.
- The floor is set by LB's 524,176 namespace flushes, about 241 misses per flush. A verdict cache that survives an
  edge is a lever this model cannot measure: the trace says only that the key changed, not to which key.
- The rule's break-even run times at 25 MHz. These are thresholds, not estimates; the run time below is B5TL1-g1b's.
  - NO BUILD (ceiling, cold 40.4) at 29,270 s or more. g1b's mcf stage bound is 30,000 s, so even a run cut at the
    bound decides.
  - BUILD (removed x warm 15.6) by LB size:
    - 1024x8 removes 54,589,670 and needs a run of 3,406 s or less (v2's population, superseded 2026-11-27 below);
    - 256x4 removes 7,807,101 and needs 487 s or less;
    - 128x8 removes 7,148,851 and needs 446 s or less;
    - 128x4 and 64x8 need 137 and 127 s or less.
  - RC (removed 136,290,618; its own check misses 44,836,912; c = RC's miss cost):
    - **c's lower bound is 2 cycles, the minimum observed** (not shown to be the unit's structural floor; no RTL path
      was traced to make it one), from simulation:
      - Source: capstone-ariane b5b2-hangmon-c cf72629ad, whose SUP_TRACE-only R81-RD line prints every lifetime-unit
        read's latency from request to response. s2pre-wt `.sup/hmlat`, dirty=no.
      - Over 69 Line B tests and 46,358 reads: minimum 2 cycles (261 reads), 4 cycles for 92 %, maximum 1,225 (the
        cold srvcost reads).
      - An RC miss makes at least one unit read. A D$-hit latency is on-chip, so the cycles carry over to silicon.
      - One read, not four: a line-fill design could fetch the line in one access, so four reads are not proven.
    - **RETRACTED 2026-11-26: "RC BUILD needs a run of 8,146 s or less at c = 2"** (dev 6be718f28fb9). RC's own misses
      are SUBTRACTED from its saving, so a BUILD must hold at c's UPPER bound. c = 2 is RC's most favourable value and
      gives no BUILD threshold (the semantics owner's correction).
    - **RC BUILD is UNDECIDED at any run time until c has a proven upper bound.**
      - The warm saving, 136,290,618 x 15.6 - 44,836,912 x c, is positive only for c < 47.4 cycles. So a BUILD needs a
        64-byte line fill costing under about 1.2 cold record fetches of 16 bytes.
      - No ceiling is proven: 40.4 is the AVERAGE cold verdict miss, not a maximum, and a 64-byte fill spans four of
        the D$'s 16-byte lines.
      - Under either candidate ceiling, four cold fetches (161.6) or the worst observed read (1,225, a simulation
        figure), RC BUILD is impossible on mcf at any run time.
    - RC candidate NO BUILD at c = 2 needs a run of 21,666 s or more (22,024 s at c = 0).
    - Between those, the result is UNDECIDED, and only the residency pass resolves it.
- **The denominator: board 3, B5TL1-g1b on b42c8c644, 505.mcf_r PASS** (N = 1; graders v2 b976d288 and v1 PASS, no
  cause line; the resident name read back before and after).
  - The run's time is its own SPEC-END line, `up0=6854.08 up1=13455.23`: 6,601.15 s, 165.03 G cycles at 25 MHz.
    Read here from the raw capture (uart sha256 0184faff239659d3), in stage B5TL1-505.mcf_r, rc=0.
  - The image is the traced binary. The board lane read mcf_r_base.native a1daec395a9e7260... from the payload's
    embedded initramfs, which is the image above. The runtime lane checked the payload against SHA256SUMS.g1b.
  - Disclosed: this is the qualification kernel (fpga-v1, periodic tick, no fast rdtime), not the campaign relink. A
    CF-B5TL1-g1b row does not exist yet.
- **Read against the thresholds above:** 6,601 s lies between BUILD (3,406 s for 1024x8) and NO BUILD (29,270 s), so
  the rule decides only part of it.
  - Ceiling: 4.434 % at cold, so not NO BUILD.
  - LB 256x4, 128x8, 128x4, 64x8: candidate NO BUILD, at 0.191 / 0.175 / 0.054 / 0.050 % even at the cold cost.
  - LB 1024x8 (v2's population, superseded 2026-11-27 by the tick-rate reading below): UNDECIDED, 0.516 % at warm and 1.336 % at cold.
    - It is a BUILD exactly when its 54,589,670 removed misses cost more than 30.2 cycles on average, that is when
      more than 59.0 % of them are cold.
    - What decides it is the D$-residency of those misses.
  - RC: candidate NO BUILD fails (3.282 % at cold with c = 2).
    - A BUILD needs its saving at warm to exceed 1 % with c at its upper bound: c < 10.6 cycles. At the cold cost,
      c < 86.0 cycles.
    - So RC is BUILD or UNDECIDED, never NO BUILD, at this run time.
    - What decides it is a proven upper bound on c, together with the residency of RC's 136,290,618 removed misses.
  - The residency pass is the nested lane's full-stream D$-residency model, checked on silicon. It was asked for on
    2026-11-26 because the run landed in this band, as the ruling prescribes.
  - The silicon check the ruling requires does not exist yet: no b42c8c644 mcf row counts D$ misses (the runtime
    lane's grep of all three captures). The order:
    1. The nested lane's v3 trace with the D$ model gives the prediction, board-free.
    2. Then a board row on b42c8c644 that counts around mcf's stage. It is a new image, pre-registered against that
       prediction, with a keep-tags firmware variant.
  - **That row's counters** (b42c8c644, perf_counters.sv / csr_regfile.sv; six counters, mhpmcounter3..8, MHPMCounterNum = 6):
    - Access:
      - mhpmevent3..8 are M-only (csr_regfile.sv:819-847), so firmware programs them;
      - S or U reads hpmcounter3..8 only with mcounteren bits 3..8 (:758), and U also needs scounteren (:664);
      - mcountinhibit is at :792;
      - read at the stage's start and end.
    - Prior art for the path: the hpm-bare-2026-11-03 kit (lane land/rtl-hpm-kit 289ad4e544d8).

    | event | what it counts (perf_counters.sv) | why |
    |---|---|---|
    | 0x02 | L1D miss (:115): wt_dcache_missunit.sv:226 `mshr_allocate && !miss_nc`, every allocating cacheable miss on ANY D$ port, the lifetime unit's port 2 and the PTW included | the count the model predicts; the model must include the same populations |
    | 0x17 | R-43 replays (:138) | approximately the silicon's verdict-cache misses, a cross-check of the 181,127,530 (whether LDC/STC misses replay the same way is checked before calling it equal) |
    | 0x11 | L1D accesses (:133-134), ports ex_cache[0..2] only (cva6.sv:2493), so EXCLUDING the lifetime unit's port | with 0x02 and the model, separates program from record misses |
    | 0x1d | record-TLB walk started (:147) | the model's record TLB (round-robin in the RTL, ex_stage.sv:2395-2398) |
    | 0x1f | paged-table record access (:149) | the same, its denominator |
    | 0x04 | DTLB misses | PTW traffic, the model's blind spot |

**505.mcf_r, LB 1024x8: candidate NO BUILD on the tick rate; the tick placement is open** (2026-11-27). RC stays
UNDECIDED.
- **v2's 1024x8 population is a property of the twin's speed, not of mcf.** It is the 54,589,670 above and its
  3,406 s threshold.
  - Each timer tick ends the verdict cache's contents for protected U.
  - Without `-icount` the twin's guest clock follows the host. v2 placed about 131 k ticks (262,027 MRETs / 2, inferred by the same identity on an older tracer build) in mcf's run, and v3
    (tracer c3f93ba340) placed 203,935.
  - Silicon's run is 660,115 ticks, and fewer ticks remove more misses:
    - g1b's kernel (trapk/fpga, payload 539b2ef41ee4d444...; its .config equals trapk/fpga-v1's except the initramfs
      path) is CONFIG_HZ=100 HZ_PERIODIC, over 6,601.15 s;
    - the trace kernel, Image.twin 749fed26 = trapk/linux-6.1, is CONFIG_HZ=250 NO_HZ_IDLE;
    - neither side has Sstc, so every tick is an M-mode timer interrupt.
- **The traces.** The nested lane's v3 trace carries the D$ model. The two icount traces run under QEMU
  `-icount shift=6` and `shift=7`: the guest clock advances 64 or 128 ns per instruction, so ticks land at fixed
  instruction counts. Sources:
  - pre-registration nested-workloads 7650843941fe, Amendment 1 bb10424b43e8, RESULT 4fa28baf40c1;
  - instruments in capstone/tests/record-trace/dc-v3;
  - each graded B5TL1-505.mcf_r PASS, WINDOW-OK, mismatch_vs_LB0 0;
  - identical protected instruction counts (26,644,263,390 / 394 / 390).
  The rtl lane read every count below from the rcm-lb-dc 1024x8 outputs under
  `<hdd>/capstone/nested/rectrace-dc/runs/spec-mcf{,-ic6,-ic7}`.
- **A tick is one flush window.** This is measured by `mcf-tick-thin/` (eana.py and eseq.py on escan.py's extraction
  of each trace; result lines in `mcf-tick-thin/results.txt`).
  - The tracer emits an edge on every namespace-key change. The key is `root<<1 | (!m_mode && root != 0)`, so a tick
    is four flushing edges: MTI to M, MRET to U, the kernel's sbi_set_timer ecall to M, and MRET to S. The
    sequence is MTI → MRET → STI → ecall 9 → MRET → SRET in 453,726 of ic6's 467,219 ticks.
  - No record event lies between the MTI's MRET and the next edge: 467,207 of 467,219 on ic6, 203,932 of 203,935 on
    v3, 1,001,347 of 1,001,357 on ic7.
  - So E op 3 counts two MRETs per tick, and LB flushes = 2 x E op 3 + 2 x tables exactly (816,018; 1,870,662;
    4,013,054). Ticks, not MRETs, are the unit to compare.
- **The uniform-placement curve.** The tick flush bits are cleared in the ic6 and ic7 traces, every K-th tick kept
  (`thin.py`), and the unchanged rcm-lb-dc binary rerun on the copy. Only n is valid in a thinned run.

| ticks in mcf's run | trace | removed_lb n (LB0 misses - 1024x8 misses) | share at all-cold 40.4 |
|---:|---|---:|---:|
| 116,804 | ic6, 1 tick in 4 | 129,066,128 | 3.160 % |
| 233,609 | ic6, 1 in 2 | 85,199,575 | 2.086 % |
| 350,414 | ic6, 3 in 4 | 55,952,193 | 1.370 % |
| 408,816 | ic6, 7 in 8 | 41,337,822 | 1.012 % |
| 467,219 | ic6 | 26,617,270 | 0.652 % |
| 500,678 | ic7, 1 in 2 | 23,308,489 | 0.571 % |
| **667,571** | **ic7, 2 in 3 (silicon: 660,115)** | **17,917,437** | **0.439 %** |
| 1,001,357 | ic7 | 7,130,356 | 0.175 % |

- Not on the curve: v3, host clock, 203,935 ticks, n = 28,507,321 (0.698 %). That is the placement effect, below.
- **Reproduced independently** by the nested lane's own tool (rectrace-dc/rcm-lb-dc-thin, `--thin K:PH`; outputs in
  `<hdd>/capstone/nested/rectrace-dc/thin/`):
  - control 1:0 on ic6 = 26,617,270 exactly;
  - ic6 2:0 = 85,197,216 against 85,199,575 here (0.003 %; its tick windows close by its own rule, 3 reopened);
  - the other phase, 2:1, = 85,180,582;
  - ic7 2:0 = 23,308,489, exactly as here.

- **Reading the curve.**
  - It is monotone in the ticks placed at fixed instruction spacing. The ic7 half lands next to ic6, so the two runs
    differ only in tick density.
  - It crosses the BUILD line (n = 1.6503 G / 40.4 = 40,849,010) between 408,816 ticks (41.34 M) and 467,219
    (26.62 M), near 411 k by linear interpolation between those two measured points.
  - At silicon's count it reads 17.9 M, 0.439 % even if every removed miss is cold. NO BUILD on this row needs a run
    of 2,895 s or more; the run is 6,601 s.
  - **Placement is first-order.** v3's host-clock placement at 204 k ticks removes about what uniform placement does
    at about 460 k (28.5 M lies between the 408,816 and 467,219 points). So neither the trace order v2 > v3 > ic6 > ic7 nor any count of ticks alone is evidence for the
    verdict.
- **Verdict: candidate NO BUILD for LB 1024x8, with the tick placement open.**
  - Silicon places ticks in cycle time. A BUILD needs that placement to leave the regions where 1024x8 helps with no
    more than about 0.62x the mean tick density (411 k / 660 k), that is, those regions running at 0.62x mean CPI
    or less.
  - Nothing measured excludes that. The only non-uniform placement measured, the twin's host clock, went about 2.2x
    toward NO BUILD, but host time is not cycle time.
  - The settling reading: silicon's tick positions in mcf's stage, as the minstret delta of every tick. The runtime
    lane owns that row: a kernel-side tick log on the same kernel config, with its own instrument controls. Its binding test re-places the
    ic7 trace's ticks at silicon's positions (`mcf-tick-thin/thin_keep.py`: silicon tick k keeps ic7 tick floor(F_k x N),
    F_k its minstret fraction of the window; ic7's ticks sit at constant instruction spacing under -icount, so rank
    is instruction fraction; its control, 667,564 evenly spaced F_k on ic7, reproduces the 2-in-3 thinning's
    n = 17,917,437 exactly, with 0 collisions; a clustered negative, 1,000 F_k in a 0.1 % band, reports 355) and reruns the model: NO BUILD if n < 40,849,010.
  - **Where cycle-time placement leans: against BUILD, on the twin** (the nested lane's mcf-tk6, a repeat of ic6
    with an exact per-tick side file; pre-registration nested-workloads 678d571b2cb6, RESULT 57d52b7c0236; result lines
    in capstone/tests/record-trace/dc-v3/results-tk6/, read here).
    - The repeat: 467,258 ticks, removed 26,621,160 (ic6 26,617,270). Every replay row joins the side file.
    - The slice is the top decile of instructions by LB0 misses per 1000 instructions. It holds 97.4 % of LB0
      misses and 99.85 % of the removed misses.
    - The slice makes 60.568 LB0 misses per 1000 instructions against the rest's 0.179. Its program D$ MPKI is
      178.068 against 55.996 (all sources: 419.106 against 93.657).
    - **The test.** Take ticks per instruction in proportion to CPI. A BUILD then needs the slice's density at
      <= 411/660 of the mean, that is, the rest's CPI Y at >= 1.6732 x the slice's X, whatever the mean CPI.
    - **The proxy.** CPI = 1 + (LB0 x lb_cost + D$ misses x dc_cost) / instructions, with no overlap.
    - **The reading.** Y/X reads 0.23-0.39 at every listed cost (lb 40.4 or 15.6, dc 20-80, program or all
      sources). As dc_cost grows, Y/X tends to the rest/slice MPKI ratio, 0.3145 (program) or 0.2235 (all), so no
      D$ cost reaches 1.6732.
    - **The no-overlap proxy, from source.** CVA6's write-through D$ has one read MSHR (wt_dcache_missunit.sv:223-224 at
      b42c8c644: "currently we only have one outstanding read TX"), so read misses serialize. That supports the
      proxy for loads; it does not cover other stall classes.
    - **A bound from silicon's mean CPI alone.** 165.03 G cycles over about 29 G instructions is about 5.7; the
      instruction count is v3's instructions_all, not a silicon reading.
      - A BUILD needs X <= 0.6227 x 5.7 = 3.55.
      - The LB term alone gives X >= 3.447 at 40.4 and X >= 1.945 at 15.6.
      - So the slice's program D$ misses would have to cost <= 0.58 or <= 9.0 cycles each. On silicon a record
        line's D$ miss costs 40.4 - 15.6 = 24.8 cycles (the measured cold-minus-warm pair below).
    - **Status.** This is the twin's placement proxy, not silicon's placement. The verdict stays candidate until the
      runtime lane's tick log, which records mcycle and hpm 0x02 per tick and so reads X and Y directly.
  - The smaller LB sizes were NO BUILD already.
  - The modelled LB is invalidated on every namespace-key change. One that survived a tick would remove v2-like
    populations or more: that is the lever above that this model cannot measure.
  - The verdict is scoped to this row (the qualification kernel, 165.03 G). A campaign kernel with another HZ or fast
    rdtime changes both the tick count and the denominator.
- **RC stays UNDECIDED.**
  - Its removed misses hardly move with the tick rate: 136,588,885 / 137,469,116 / 139,194,783 on v3 / ic6 / ic7.
  - The model's cold fraction for them is 0.988 / 0.986 / 0.980. At that residency RC is a BUILD iff c < 85.4 / 86.0
    / 87.0 cycles, against c < 10.6 at the warm cost.
  - Using the residency needs the silicon check above (0x02 against the model's window misses), and c still needs a
    proven upper bound.
- **Withdrawn before landing** (sent to the nested lane only, never committed):
  - "E op 3 = MRET into protected U": half are MRETs into S, as above.
  - "every trace at or above v3's MRET rate gives n <= 28.5 M": refuted by the ic6 half, 233,609 ticks and n = 85.2 M.
  - Found by the claim-auditor before this row landed.

**500.perlbench_r test (makerand.pl): NO BUILD** (decided 2026-11-26 on its board row, below).
- The trace is the nested lane's spec-perl3.2883687: 10,130,629,600 B, sha256 1af7b0c90ab7cb56..., recomputed here. It
  has 633,164,350 events and 17 tables. It comes from the campaign's own stage, SP-B5TL1-g4mk.
- The image: perlbench_r_base.native f3c8c80d71844564..., which equals g4mkroot's B5TL1/500.perlbench_r native
  (recomputed here).
- LB check misses: 277,290 under both walks, over 1,216,156,215 checks, with self_evict 0. RC reads 228, and the size
  floor (4096x32) is 277,204.
- Thresholds at 25 MHz (not estimates):
  - NO BUILD on the ceiling (cold 40.4) at a run of 44.8 s or more;
  - a larger LB removes at most 86 misses;
  - RC's candidate NO BUILD at c = 2 (removed 277,062 x 40.4 - 228 x 2) is at 44.8 s or more as well.
- **The denominator: board 2, CF-B5TL1-g4mk on b42c8c644, 500.perlbench_r PASS** (N = 1; the resident name read
  back before and after; graders v2 and v1 PASS, no cause line).
  - The run's time is its own SPEC-END line, `up0=51.16 up1=326.86`: 275.70 s, 6.8925 G cycles at 25 MHz.
    Read here from the raw capture (uart sha256 2522348b57230d99), the stage named B5TL1-500.perlbench_r.
  - Shares: 277,290 x 40.4 is 0.1625 % of it (warm 15.6: 0.063 %); RC's candidate saving 277,062 x 40.4 -
    228 x 2 is 0.1624 %. Both are about 6x under the 1 % line, so neither run-to-run noise nor the kernel
    choice below can move the verdict.
- **The image is the traced binary, checked from the booted bytes here.** Payload e82d42ebe558cb18 carries
  the build's Image at 0x200000 (full compare). Its gzip member at Image + 0x6109c0 decompresses to the build's
  initramfs.cpio. That holds overlay.cpio 5a07486727a883fe... at 0x767000, which packs
  stages/B5TL1-500.perlbench_r/stage/bin/native/perlbench_r_base.native as f3c8c80d71844564..., the traced image.
  The board lane and the runtime lane reached the same binary independently.
- **Disclosed:** this payload is the campaign relink: the final Full kernel with the firmware's fast rdtime
  (FW_FAST_RDTIME=y), built from /mnt/hdd/<user>/capstone/trapk/fpga/spec/CF-B5TL1-g4mk.
  - It is not the qualification payload 192a5a755b9fb8b8.... The board lane diffed the two initramfs entry by
    entry: 1,107 entries, 0 differing in content. Only the kernel differs.
  - Its hashes are not yet in a committed kit; they go into the campaign pre-registration.
- **Not the denominator:** CFP-B5TL1-g4mk (Full + ports, 432e6ae27d5aa062..., 279.75 s) runs the ports build
  b4eb3e6f267f5f34... (the board lane's reading of the payload bytes, not re-read here), not the traced binary.

**The denominator: board 3, B5TL1-g3 on b42c8c644, 502.gcc_r PASS** (board/meas-pw f47b2a686ce2, b42c8c644-board3 result lines).
- The run's time is the SPEC-END line's up1 50.54 minus up0 44.78 = 5.76 s. The row prints no cycle count, so cycles are
  that time x the core clock.
- The core clock is 25 MHz, by three independent readings:
  - the payload's own DTB: timebase 12,500,000 and cpu clock-frequency 25,000,000 (the board lane's parse);
  - silicon: 2.0 cycles per mtime tick (crevint-lat 30062/15031; boot 13's mtrip 1.99);
  - kernel time against host wall time within about 0.1 % over 14,499 s.
  The verdict holds for any clock above 6.26 MHz.
- The image, as loaded: fw_payload.bin sha256 ce0ca633c490c4713cbde33f88ba4c2d474b21907273b273c989067d00c94ef8, pinned
  by the plan and checked before the load.
- Its overlay holds cpugcc_r_base.native f800e015d161c911... (10,310,640 B), and the same in/t1.c (de7b86d79a68965a),
  plan.json and run.sh. Recomputed here, the traced stage directory (c4/b5b2spec/board-stages/B5TL1/502.gcc_r, the
  path in the nested lane's spec-chain.sh) holds the same native, input, plan.json and run.sh. The native's mtime is
  2026-11-25, before the trace.
- The twin's kernel (Image 749fed26) is not the board's (7e47228e...). The record traffic is the user-space
  allocator's, and the trace's S-mode checks are 299 of 6,525,939, so the kernel does not move the row.

**Reading the gcc row:**
- At the cold cost it is 360,408 cycles (0.250 %); at the warm, 139,168 (0.097 %). Both are under 1 %, so it is NO
  BUILD for a record cache on gcc.
- The verdict-cache size is NO BUILD too. The floor at 4096x32 is 8,897 misses, so no size removes more than 24.
- The smoke background (15 misses in spec-smoke.2198060) is left in. That is the conservative side for a NO BUILD.
- Still owed: mcf and perl, each on its own B5TL1 trace and board row.

**Verdict-cache size** (`--lb`, the `small` arm only; LB check misses, asis / dedup):

| geometry | misses |
|---|---|
| 32x4 | 44,611 / 45,117 |
| 64x4 (b42c8c644) | 20,309 / 19,615 |
| 128x4 | 11,441 / 11,344 |
| 256x4 | 10,601 / 10,601 |
| 64x8 | 10,728 / 10,700 |
| 128x8 | 10,582 / 10,582 |
| 512x8 | 10,582 / 10,582 |
| 4096x32 | 10,579 / 10,579 |

- The floor, about 10,580, is first-touch misses plus the refills after LB's 8,280 namespace flushes. No size removes
  it.
- So on this arm a larger verdict cache can remove at most 9,730 misses (asis) or 9,036 (dedup), and 128x4 or 64x8
  already removes most of that.
- The size decision uses the same rule as the record cache: the misses REMOVED x the cost pair, against the
  program's measured cycles. It is made per program on gcc, mcf and Perl, not on this arm.

**Measured per-miss cost** (RTL simulation, b42c8c644's core, memory delay 12; capstone-ariane b5b2-prefix bd3932800, b5b2-srvcost and -DSC_COLD):
- warm record (in the D$): (P3 - P2) / 33 = 14.4 cycles per verdict miss;
- cold record (D$ evicted): (P3 - P4) / 33 = 70.0 cycles, and independently (P7 - P2) / 33 = 71.5 cycles (P7's probes read records evicted before P4).
- These are simulation costs at a fixed 12-cycle memory. They are not the decision's input.

**Silicon per-miss cost** (the decision's input): `capstone/tests/rtl-smoke/b5b2-misscost-bare-2026-11-26`, run on
b42c8c644's seal on board 2 on 2026-11-25. The seal was read back by name. The kit's RUN lines and image hashes were
checked by the board lane against the arming file. Result lines are at board/meas-pw f47b2a686ce2,
`b5b2-seals-board-2026-11-25/b42c8c644-board2.result-lines.txt`.

| image | SV (positions 1-13) |
|---|---|
| b5b2-srvcost | 0x1bd 0x674 0x688 0x674, 0x5f2 0x83 0x290 0x83 0x83 0x83 0x286 0x83, 0x8 |
| b5b2-srvcost-cold | 0x1c2 0x674 0x682 0x674, 0x5ef 0x83 0xb2c 0x666 0x83 0x83 0x5b8 0x9b, 0x8 |

- **Graded here by `mc-grade.py`** (the kit) against `expected-vectors.json`:
  - sublet-life (SR 35): all 25 E positions exact;
  - both images: SR 13 with position 13 = 0x8, and rule 3 (AS PREDICTED) holds.
  - Negative test: a copy with one sublet-life E value and image 2's P3 perturbed FAILS both checks.
- **The costs, in cycles per verdict miss:**
  - warm: 15.9 (P3 - P2) and 15.6 (P7 - P2);
  - cold: 37.0 (P3 - P4) and 40.4 (P7 - P2).
  The cold cost on silicon is about half the simulation's (78.7 / 71.5).
- **Break-even, under the rule** (both bracket ends and both readings must fall on one side of 1.0 %). On a
  trace, misses x cost spans:
  - cold: asis misses x 40.4 at the top, dedup misses x 37.0 at the bottom;
  - warm: dedup misses x 15.6 at the bottom.
  An LB miss's record may be either warm or cold in the D$, and this model does not track the D$. So:
  - if the cold span is below 1 %, the result is NO BUILD whatever the residency;
  - if the warm span is above 1 %, the result is BUILD;
  - otherwise it is UNDECIDED until the D$ residency of the missed records is modelled.
  **Ruled by the semantics owner (the runtime lane, 2026-11-26), as written above:**
  - NO BUILD only when the saving is under 1 % at the COLD cost, taking the higher of the pair (40.4);
  - BUILD only when it is over 1 % at the WARM cost, taking the lower (15.6);
  - anything between is UNDECIDED.
  An UNDECIDED is resolved by exactly one of two things:
  - (a) a model of the missed records' D$ residency, checked against one silicon reading before it is used;
  - (b) a variant kit on silicon whose missed records sit in a known residency.
  No interpolation between warm and cold is allowed. Every verdict records:
  - the cost pair;
  - the program-cycle denominator and its source, which must be a board row cited by image hash and never a twin
    figure.
- **On the first trace** (the `small` arm, which is NOT one of the decision's three programs):
  - cold: 20,309 x 40.4 = 820,484 cycles and 19,615 x 37.0 = 725,755;
  - warm: 19,615 x 15.6 = 305,994.
  It is NO BUILD at 82.05 M cycles or more, and BUILD at 30.6 M cycles or fewer. Its cycle count is not measured.
