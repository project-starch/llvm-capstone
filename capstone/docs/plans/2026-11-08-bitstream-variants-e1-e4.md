# Bitstream variants E1-E4 (spec-silicon backlog e5ed6ba38803, section E): sizing, predictions, gates

RTL lane, 8 Nov 2026. Every variant is model-gated: built only when its prediction says it moves a run. Each
prediction below was written and pushed BEFORE the run that tests it.

## Common base

Every variant is built on the local **stage-B batch** (capstone-ariane fe4810dd1 = C.5a 1fc2cd5aa + R-32 fix +
R-68 NI rule + DYN handshake trim + LDC rs1; lint = stage A exactly; R-32 and R-68 two-sided in sim). The batch
itself is the CONTROL bitstream and goes to synthesis first, so each variant differs from its control by one change
and a Plain pair runs on the same bitstream. The lineage stays local (not pushed to origin).

## E2: L1D and DTLB geometry

C.5a: L1D 32 KiB, 8 ways, 128-bit lines (16 B), write-through, loads allocate, stores never allocate, random
replacement (8-bit LFSR, `wt_dcache_missunit.sv:202-212`); the perf event "L1D miss" is the read MSHR allocation
(`miss_o`). DTLB 16 entries, fully associative, tree-PLRU (`cva6_tlb.sv`).

**Line width is NOT a free parameter here:** the capability tag is one bit per (line, way) (`cap_tag_q`,
`wt_dcache_mem.sv:146`) and the write buffer's granule logic assumes a line is one 16-byte granule
(`wt_dcache_wbuffer.sv:121`). A 32-byte line is a rewrite of the tag path, not a config change: out of scope before
Wednesday. E2 therefore = more ways (VIPT caps a way at 4 KiB: 16 ways = 64 KiB) and/or more DTLB entries.

**Instrument:** QEMU plugin `dcm.c` (this folder), one pass, L1D at 8/16/32 ways random + 8 ways true LRU, DTLB
16/32/64 entries LRU (4 KiB user pages, 2 MiB kernel), accesses classed by instruction address (K kernel, F firmware,
U other). Kits `$CAPSTONE_WORK/spec/kit-dcm-egq` (Plain) and `kit-dcm-etq` (Capstone), the bodies of kit-egq/kit-etq
plus the plugin; stages as perf/rss (Plain) and perf/et qgate (Capstone devmk), mcf test.

**Control (must hold before any geometry is believed), against the silicon mcf test counters (results
2026-11-05-mcfx, out-mx-b2 Plain / out-mx-b1 bundle; machine-wide Plain, in-context Capstone):**
- C0: Capstone run: LDC/STC executed > 0 and their memory callbacks > 0 (helper accesses visible), else VOID.
- C1: model 8-way random, Plain all classes: within ±30 % of silicon's 2.898 G L1D misses (84.79 / 1k x 34.176 G).
- C2: model 8-way random, Capstone U: within ±30 % of silicon's 3.796 G (140.35 / 1k x 27.043 G).
- C3: the Capstone-U / Plain ratio within ±0.10 of silicon's 1.31.
- C4 (reported, looser): DTLB-16 Plain vs 0.341 G, Capstone U vs 0.463 G; ratio vs 1.36.
If C1-C3 fail the model is not used for E2's decision, and that is reported.

**Build rule:** E2 goes to synthesis if the model's 16-way (and/or DTLB 32/64) reduction of Capstone-U misses,
times the per-miss penalty range 12-34 cycles (sim floor .. the mcf account's upper), is at least 1 % of mcf's
in-context cycles (27.043 G instructions at IPC 0.116 = 233 G cycles, 9,303 s) at the LOWER penalty, or 2 % at the upper. Below that, E2 is
reported as not a lever with the numbers. The Plain pair is mandatory either way (a larger cache helps Plain too).

## E3: the revocation verdict cache (written before the run)

C.5a's RVC (`load_store_unit.sv:263-296`, `:1185-1275`, `:1332-1334`): 64 sets x 4 ways = 256 entries, set =
revnode_id[5:0], 30-bit hit test, victim = same index else a free way else a per-set round-robin pointer advanced on a
displacing install; a miss is replayed through a rev-node probe whose read-back installs the id; the whole cache empties
on `revnode_ns_flush_i` (the virtual-context flag or the table base changed), i.e. at every context entry and escape.

**Instrument:** the twin QEMU (capstone-qemu 533163eb25 = virtual/flat-index-18 46b69c03a4 + a doc commit; its build is
byte-identical to the kits' 708bbff5) plus `qemu-rvcm.patch` (this folder): every tagged capability access in
`capstone_check_access` looks the id up in six geometries at once (64x4, 128x4, 256x4, 1024x4, 64x8, 128x8), a miss
installs, a change of `capstone_virtual_c` or `srevroot` flushes. Kit `$CAPSTONE_WORK/spec/kit-rvcm-etq`; mcf test,
Capstone devmk stage (as qgate).

**Controls:** R0 vctx lookups > 0 and within ±25 % of the dcm plugin's Capstone U-class loads+stores (every in-context
access is a capability access); R1 flushes > 0 and of the order of 2x the run's context entries (32,513 on the
recorded QEMU run).

**Prediction (desk):** mcf's accesses go through few long-lived array capabilities (nodes, arcs, basket), so the hot
set within a quantum is far below 256; misses are dominated by the cold start after each flush: of order 10^5 flushes x
tens of ids = 10^6-10^7 misses at 64x4, and larger geometries remove little.

**Build rule:** E3 goes to synthesis if (misses at 64x4 - misses at the best larger geometry) x 12 cycles >= 1 % of
mcf's 233 G in-context cycles (>= 194 M misses saved), or x 34 cycles >= 2 % (>= 137 M). Otherwise E3 is not a lever,
reported with the numbers.

## E4: the escape/resume hardware share — sized, NOT a lever

**Switcher walks, measured in sim** (c5a-wt `.sup/irqsweep`, irq-p200-q1500-d0, fe4810dd1 + sim-only traces, zero
memory delay; script `e4walk.py` over the `Save idx` / `Restore idx` dprints): 206 escape walks of ~101 ids in ~400
cycles, 205 resume walks of ~69 ids in ~270 cycles, plus walks of 14 ids (~60) and 11 ids (~30) per round. About
760 cycles of switcher work per exit+resume, ~3-4 cycles per id (the stores go through the store buffer, the loads
hit the frame just written).

**What else the hardware adds per exit:** every satp write by the switcher flushes all TLBs (S-29 fix), so a resume
refills at most 16 DTLB + 16 ITLB entries; the verdict cache empties on the namespace change (E3 measures its refill).
On silicon, at an assumed 1-3x the sim walk and ~100 cycles per page walk, that is at most ~5 k cycles per exit.

**Against the per-exit cost on silicon:** perlbench ~500 k cycles per exit (backlog, test size); mcf 40.5 ms per exit
outside the context at ~25 MHz = ~1 M cycles (results 2026-11-05-mcfx, bundle). The hardware share is at most 1 % of
an exit (perl) and 0.5 % (mcf). Exits are ~18 % of perl's run (1.38x vs 1.13x inside) and 19 % of mcf's (2,232 of
11,535 s), so even a switcher that cost nothing would move perl's run by at most ~0.2 % and mcf's by ~0.1 %.
Halving the walk saves a quarter of that. **E4 is not built.** The exit cost is the Linux work around the exit
(backlog C2).

## E0: the stage-B control bitstream on silicon — first-boot gates (DRAFT; vectors from image sims, filled below)

The runtime lane's conditions (8 Nov): E0 clears the same board gates C.5a did before any variant on it is read; C.5a
stays the registered known-good fallback on every board; each seal records its routing margin (placed utilization,
congestion level, WNS).

**Boot 1, in this order (every image returns; one image per kit run as C.5a's kit session):**
1. slot kit (`slot-bare-2026-11-04`, 9 images; reproduced bit for bit from kit-98ff08e6's build.sh): the
   expected CHANGE. The DYN trim + LDC rs1 move the stc/ldc pair arms; slot-9 (bare loop), slot-5/6 (plain pair)
   unchanged.
2. R-32 kit (`r32-shrinkto-bare-2026-11-06`, land/rtl-r32-kit): FIXED vector (SHRINKTO 64 over 256 reads 64).
3. R-67 kit: FIXED, controls unchanged (as C.5a).
4. hpm kit (7 images; hpm-8 not simulated): C.5a's values; cycles may move by the trim on LDC/STC arms.
5. ObjBounds kit (11 images): C.5a's verdicts and cost pattern.
Then one SPEC test stage (the runtime lane's choice; xz or mcf test, 1-2 h): rc 0, specdiff, outputs = C.5a's.

**Grading:** each kit's own grader against the stage-B column below; a value the sims did not predict is a finding,
a missing kit result is VOID for that kit. The flash itself: sealed sha256 checked locally, resident name read back,
C.5a's bitstream registered as fallback (unchanged).

## E5: the 20-bit lifetime index (bitstream 4's option A on stage-B) — the QEMU twin's gate (written before the runs)

RTL: capstone-ariane e5-idx20 443f67892 (local) = stage-B a481031ca + option A 741aff6a6 (cherry-picked d8626c14c) +
its arms (35c224153) + the PC tracker's kill compares at [19:0] (443f67892). Twin: capstone-qemu
virtual/flat-index-20 23f9df04e5 (flat-index-18 + the ABI constants re-split; model checks 0 failures, 4 failures on
the 18-bit sources as the negative control), binary 398a6544425df6b2.

**Gate (QEMU), the live smoke** of bitstream 4 (`f55/live-stage`, `live-T3` 1eb994b5fb979862: N 16-byte objects
alive at once, N = 1000, 400000, 1100000), two worlds, same sources and launcher (vexec b8d0673e55527639, runtime's
perf/univ-module 3f7c8e19fecc, runtime sources byte-identical):
- E5 world: twin20 + kq20 (ko ba57ac39b6003f15, INDEX_BITS=20, runtime's build);
- 18-bit world: twin18 (708bbff5663f0e86) + kq18 (ko 7ee19fd0382f2ac0, INDEX_BITS=18, built here from the same tree).
Each module only runs on its own twin's ABI, so the pair is matched by design, not one variable.

| N | E5 world | 18-bit world |
|---|---|---|
| params | index_bits=20, node_flat_records=1048576 | index_bits=18, node_flat_records=262144 |
| 1000 | LIVE-OK | LIVE-OK |
| 400000 | LIVE-OK, nodes 400,000-430,000, node_capacity 1048576 | LIVE-NULL at an index near 251,000-262,000, nodes 255,000-262,144 (bitstream 4's 18-bit control read 251,479 / 260,455) |
| 1100000 | LIVE-NULL at an index 990,000-1,047,000, rc 3, nodes 1,040,000-1,048,576 (B's F0 window) | as N = 400000 |

No domain fault lines in any cell. If live-T3 (built 30 Oct) does not run under today's launcher in EITHER world, the
gate is VOID and a fresh live-T3 build is the next step, not a reading.

**Result (8 Nov, perf/e5twin, both worlds on one boot each): every pre-registered cell holds.**

| N | E5 world (twin20 398a6544 + kq20 ba57ac39) | 18-bit world (twin18 708bbff5 + kq18 7ee19fd0) |
|---|---|---|
| params | index_bits=20, node_flat_records=1048576 | index_bits=18, node_flat_records=262144 |
| 1000 | LIVE-OK bad=0, nodes 1,111 | LIVE-OK bad=0, nodes 1,111 |
| 400000 | LIVE-OK bad=0, nodes 414,217, capacity 1,048,576 | LIVE-NULL at 251,479, nodes 260,455, rc 3 |
| 1100000 | LIVE-NULL at 1,011,158, nodes 1,046,942, rc 3 | LIVE-NULL at 251,479, nodes 260,455, rc 3 |

FAULT-LINES 0 in every cell. The E5 world's three numbers equal option B's flat-module F0 twin cells on bitstream 4
(414,217 / 1,011,158 / 1,046,942), and the 18-bit world's equal bitstream 4's 18-bit control (251,479 / 260,455): the
live smoke's allocation sequence is deterministic, so the table width is the only thing these cells read.

## E3 result: NOT a lever (8 Nov, perf/rvcm)

**First run VOID on control R1:** 11,982 flushes against the ~65 k pre-registered: the model flushed only when a lookup
saw the context flag change, but the RTL empties the cache on every entry and escape. Fixed before reading any number
(qemu-rvcm: `rvcm_ns_flush()` at every `save_state` / `restore_state` in capstone_supervisor.c; back-to-back flushes with
no lookup between them count once). Re-run, mcf test rc 0, specdiff 2/2, instret 27.04 G as the recorded QEMU run:

| geometry | entries | in-context misses (10.37 G lookups) |
|---|---|---|
| 64 x 4 (C.5a) | 256 | 180.58 M (1.74 %) |
| 128 x 4 | 512 | 178.37 M |
| 64 x 8 | 512 | 178.58 M |
| 256 x 4 | 1,024 | 172.15 M |
| 128 x 8 | 1,024 | 172.80 M |
| 1024 x 4 | 4,096 | 59.84 M |

- R1: 38,507 cold starts (flushes with lookups between them) for 32,513 context entries: a cold cache at every
  residency. The pre-registered "about 2x entries" counted escape and entry separately; with the dedup it is one per
  residency. The first run's under-flushed misses (180.43 M) differ by 0.15 M, so cold starts are not the miss source.
- R0: 10.37 G in-context lookups. The pre-registered comparison (the dcm plugin's Capstone U count) does not exist —
  that run was voided (below, E2) — so the comparison is to Plain's 10.66 G loads + stores on the same input: 0.973x,
  consistent with Capstone's in-context instructions at 0.99x Plain.
- Same totals at 64x8 and 128x4: the misses are capacity, not conflict. The hot set is between 1 k and 4 k ids.
- **Build rule:** best larger geometry (1024x4, 16x the entries) saves 120.7 M misses: under 194 M (12 cycles, 1 %) and
  under 137 M (34 cycles, 2 %). At most 4.1 G cycles = 1.8 % of mcf's in-context 233 G, for a 4,096-entry 30-bit tag
  array. 1,024 entries save 8.4 M (0.04-0.12 %). **E3 is not built.**

## E0 synthesis-only gate (board lane, apollo, Vivado 2024.2, 8 Nov): 6 of 7 keys, the FF key attributed

Against C.5a's own gate report: core LUT 145,391 -> 144,853 (-538, band ±1,000) PASS; ex+issue -2,320 (ex -3,719,
issue +1,399; ±2,500) PASS; flu_unit 7 -> 7 (the band cannot see SHRINKTO's change); cva6_mmu -314 PASS; Synth 8-326 on
load_unit 0, timing-loop headers 0 = C.5a's PASS; peak RSS 22.35 GB. **core FF 41,426 -> 41,730 (+304, band ±300): FAIL
by 4**, attributed by flop name (board lane's census of ex_stage_i's own flops in both synthesized checkpoints,
836 / 1,153 = the report's counts; checked here by stem): dyn_child_repr_q 1 -> 253 (the 1-bit representability flop
backward-retimed through the compress/decompress compare), capstone_dyn_ftval_q 0 -> 64 (byte-identical source in both
commits; merged with an equivalent register in C.5a's netlist, not in stage-B's), dyn_child_q renamed 64 <-> 64,
two fanout replicas +2, capstone_dyn_ready_q -1 (dead outside the trace block after the DYN trim, as predicted). No new
state without a source cause. Other |dLUT| > 500 rows are hierarchy churn (dom_switcher +2,128, dyn_unit -1,926,
csr_buffer -1,849, scoreboard +2,102, lsu_i -814, i_mult +580). Full build launched on the ruling (reversible; the seal
serves as the control once its own keys pass).

## E2 result (8 Nov): the model holds; 16 ways passes the build rule but is RATIO-NEUTRAL

**Instrument change, stated:** the dcm.c plugin VOIDED the Capstone run: with its memory callbacks the firmware takes
cause 24 at boot (pc 0x8002522a, an untagged base) right after the module loads, while the same kit without the plugin
runs mcf to its timeout with no fault (perf/dcm/q-cap-noplug). The Capstone side was re-measured with the same model
inside the twin QEMU (qemu-rvcm.patch, `dcm_access` at the end of `capstone_check_access`, every in-context capability
access, vaddr-indexed / vaddr-tagged). The Plain side stays the plugin's (paddr). C0 (helper accesses visible) is
replaced by construction: the hook IS the capability access.

| | Plain (plugin, all classes) | Capstone (hook, in context) | silicon Plain / Capstone in context |
|---|---|---|---|
| L1D 8-way random | 2.487 G | 3.384 G | 2.898 G / 3.796 G |
| L1D 16-way | 2.311 G (-7.1 %) | 3.195 G (-5.6 %) | |
| L1D 32-way | 2.017 G (-18.9 %) | 2.924 G (-13.6 %) | |
| L1D 8-way LRU | 2.420 G | 3.302 G | |
| DTLB 16 / 32 / 64 | 0.772 / 0.555 / 0.434 G | 0.906 / 0.663 / 0.524 G | 0.341 / 0.463 G (16) |

- C1 Plain -14 % PASS; C2 Capstone -11 % PASS; C3 ratio 1.361 vs 1.31 PASS (+0.05). C4 DTLB: the model is 2.0-2.3x
  above silicon on both sides, so its DTLB numbers are not used as predictions.
- **Build rule (Capstone side only, as pre-registered):** 16 ways saves 0.190 G model misses = ~0.21 G at silicon's scale
  (x1.12): 2.6 G cycles at 12, 7.2 G at 34 = 1.1 % / 3.1 % of 233 G. PASSES at both.
- **But Plain gains the same:** -0.176 G model, ~0.21 G at silicon's scale (x1.17). Equal absolute cycles off both runs
  leave Capstone/Plain where it is or slightly worse (1.268 -> ~1.27 at the mid penalty). 32 ways is the same picture
  at twice the size. E2 is a lever for absolute run time, not for the overhead ratio.
- **Status: PARKED — helps both arms equally** (the runtime lane's ruling, 8 Nov: the metric is the Capstone/Plain
  ratio). Recipe if absolute time ever becomes the metric: `core/include/capstone_cv64a6_imafdc_sv39_config_pkg.sv`
  CVA6ConfigDcacheByteSize 32768 -> 65536 and CVA6ConfigDcacheSetAssoc 8 -> 16 (4 KiB per way kept, VIPT); the line
  width stays 128 (the capability tag is one bit per line, wt_dcache_mem.sv:146). A Plain pair on the same bitstream is
  mandatory.

## E0 first-boot vectors (image sims, 8 Nov): every VALUE equals C.5a's; only LDC/STC cycle counts move

`e0-vectors.txt` (this folder): the 32 C.5a-gate images (slot 9, hpm 7 — hpm-8 has no ELF here and is not predicted —,
ObjBounds 11, R-67 4, R-32 1), each reproduced bit for bit from its kit (the slot kit rebuilt with its build.sh; every
hash checked against the kit's SHA256SUMS), simulated on fe4810dd1 clean (= stage-B a481031ca's RTL; S12_MEM_DELAY 12,
time_out 3 M, model rebuilt), vectors from the recorder area, beside C.5a's own sims of the same images.
- 19 images identical to C.5a in every field: slot-5/6/8/9, hpm-1/2/4/5/6/7, objb-pcc-ctrl/s1/stale/stale-w, the four
  R-67 arms; objb-pcc records nothing on BOTH trees (retired 2,859 both, the same trap pcs).
- 12 differ ONLY in their cycle fields, all on LDC/STC arms, all faster — the DYN handshake trim and LDC rs1 as
  intended: slot-2/3 0x170b -> 0x1310 (5,899 -> 4,880), slot-1/4 -1, slot-7 (ldc alone) 0xa5d -> 0x860; hpm-3 stall
  0x1ff -> 0x2 and cycles 0x55e -> 0x461; objb s2/s3/s4/s5/s6/s7 cycles down by 1-27 %; every count, address and
  verdict field equal.
- shrinkto: the R-32-fixed vector (equal to r32img-batch on the same RTL; C.5a reads the off-by-one).
**Board pre-registration for E0 = the E0 column**, graded per kit; a value field that differs from it is a finding,
a cycle field is graded against the E0 column with each kit's own tolerance. Then one SPEC test stage (the runtime
lane's choice).

## E0 SEALED (board lane, apollo, 8 Nov): PASS on every pre-registered key

Full build of stage-B a481031ca, exit 0, 1h18m47s, peak RSS 22.34 GB under the 40 GB guard. **caplifive_stageb_a481031ca.bit,
sha256 aecb81b20be64809a7ea69d809c7dceede17beac40c98cb282d11418842640f0, 11,443,722 B.** Against C.5a's seal (1fc2cd5aa):
placed congestion level 4 (C.5a 5) PASS; check_timing 0 loops PASS; WNS -11.007 (C.5a -13.758, +2.751, inside the null
band) PASS; routed clean, 215,311 / 215,311 PASS. Routing margin: placed LUT 169,275 (83.06 %, -463), FF 63,954 (+312),
slices 48,282 (94.76 %, -412), router congestion max level 5 (= C.5a), TNS -130.6 k (C.5a -218.8 k), WHS +0.055. E0 is
the control for the E variants; C.5a stays the registered fallback on every board. No flash before Wednesday 20:00.

## E5 RTL in simulation (c5a-wt .sup/e5gate, rtl 443f67892 dirty=no, Anvil prologue OK, model rebuilt, 9 Nov)

- **vp3-idx20 a1 / a2 / a3: every reading as the test header pre-registers** (17 / 19 / 15 readings, mcause 0 in
  each). That includes the allocation past 2^18 (s5 0x40001, parent 0x40001, header 0x40002 / 0x2_00040000 / 4), the
  generation wrapping at 1,023 and retiring, and the stale id {0, 300000} refused above 2^18 with the controls below it.
- **va-alloc-csret**: 0 0x100004 0 0x200004 1 0 0 4 3 0 0 8 3 4 1 8 0, the 20-bit split's {gen 1, idx 4} / {gen 2,
  idx 4}. Positions 11 and 15 read 8 = generation 2 at [59:50] through the test's own [59:48] decode, the layout's
  prediction (the 18-bit trees read 0x40004 / 0x80004 and 2 there).
- **E5's 32 first-boot kit images** (the E0 gate's ELFs, same model, recorder vectors by board-vector.py): 31 equal
  E0's vector exactly, every value and every cycle count; objb-pcc is empty on both trees (it never reaches its
  recorder, as on E0 and C.5a). So the E0 vector file is E5's too.
- **Lint** (vp9-wt at 443f67892, Anvil regenerated): PASS at the stage-B baseline on every counter: LATCH 44,
  MULTIDRIVEN 3, ALWCOMBORDER 0, COMBDLY 0, UNOPTFLAT 39, ANVIL_UNOPTFLAT 0, BLKSEQ 2, UNDRIVEN 25, UNUSEDSIGNAL 1430,
  CASEOVERLAP 2, SELRANGE 100. Synthesis on leffe2 (the synth lane, launched 16:58Z), gate and full build in
  parallel, pre-registered against E0's absolute rows.
- **claim-auditor over a481031ca..443f67892**:
  - "completely moves the id to the 20-bit split" SUPPORTED. The cherry-pick's +/- lines are byte-identical to
    741aff6a6's. Of the 802 lines stage-B added since e7af48457, the only old-split slices were pcc_inv_now's two
    [17:0] compares (443f67892 widens them); every other compare stage-B added is a full 30-bit equality. A pattern
    sweep over the tip found no id or record field left at the 18-bit split. The RVC tag is a full 30-bit compare
    (no aliasing on bits 18-19). The widened compares end at flops (no UNOPTFLAT stem).
  - "safe to synthesize" waited for the lint gate and a read of the generated Anvil, recorded below.
  - Non-blocking: stale 18/12-split prose in comments (ex_stage, rev_node, unit.anvilh, dyn_unit; code right); the
    physical-namespace record address now spans idx[19:0] * 16 (16 MiB past the base), reachable only if an id with
    index >= 65,536 reaches the unit while the physical namespace is held (UNRESOLVED, pre-existing, not E5's).
- **Generated Anvil read (the auditor's open codegen point)**: c5a-wt core/capstone_rev_node.anvil.sv, regenerated by
  the E5 batch's prologue (PROVENANCE rtl 443f67892 dirty=no): `10'd1023` 6, `12'd4095` 0, `[18 +:` 0, `[17:0]` 0,
  `input logic[19:0] _probe_ep_probe_req_0`. Control: the 18-bit trees' generated files (vp8-wt, vp9-wt) read `12'd4095`
  6 and `10'd1023` 0. Six generated instances against five source sites: the extra one is not attributed (not chased).

## e5-native: E5 + CSRUNV DETACH/ADOPT + the resume clamp (the train bitstream for this week; 9 Nov)

Scope from the lead, relayed by the runtime lane: "native goes to silicon". The minimal RTL is (i), the one slot as a
transport; raw-tag moves (ii) are not built (pinned arenas). The checker stays firmware emulation. The fallback is
C.5a; standalone E5 builds in parallel on leffe2 as the module-path fallback.

**RTL**: capstone-ariane e5-native (local, stage-B lineage): d4885fe38 detach/adopt, 274f670ad comments only,
ad2e7a80c strip fix + wrong-frame arm, 14b0a57df the resume clamp + the Anvil legality as slices. 2cf56f02f is tests
only. The synthesis hash is **14b0a57df**.

**ABI** (CSRUNV rd, rs1 = frame PA (16-aligned), rs2 = action; the runtime lane agreed):
- 0 entry or resume, 1 discard, 2 service reply: unchanged.
- 4 DETACH: granted iff the slot is used and paused, its frame == rs1, and no CSSUPERVISE arm. Frees the slot.
- 5 | kind << 8 ADOPT: granted iff the slot is free, nothing is armed, and kind (rs2[10:8]) is 1, 3, 4 or 5. Marks
  the slot used and paused with that frame and kind.
- Both: rd = 0 granted, -1 otherwise, never a trap, no memory read. The next RESUME reads the context from the frame
  as always.
- Legal rs2: <= 2, == 4, or rs2[7:0] == 5 with rs2[63:11] == 0. Anything else is ILLEGAL_INSTRUCTION from the DYN unit.
- The kind is in rs2, not read from frame +16: commit has no memory read, and +16 is a kernel-writable word either way.

**Assumptions (the semantics owner's rulings, 9 Nov)**:
- Finding 2: the kernel supplies the adopted kind. A wrong kind is a supervisor misbehaving toward its own context.
  An example: kind 3 on a quantum-paused context, then action 2, skips one instruction and zeroes the a0 slot. Out of
  scope. With the clamp it cannot escalate privilege.
- Residual 2b: RESUME takes satp's PPN/ASID (+928) and srevroot (+936) from the frame with no ENTRY-style precheck.
  The kernel chooses both at ENTRY anyway (+0 / +8). Recorded as kernel authority; a RESUME precheck (a switcher
  change) is in the backlog.
- **QEMU/silicon divergence**: QEMU keeps a parked context's satp/srevroot host-side, so it does NOT model a frame
  whose +928/+936 were rewritten between escape and resume. No QEMU result covers that case on silicon.

**Finding 1 and the clamp** (claim-auditor of d4885fe38):
- RESUME skipped the switcher's phase-7 precheck and wrote mstatus (+920) raw and satp (+928) unchecked. ADOPT made
  that reachable without a real escape.
- The ruling: a RESUME must never yield raw M or bare translation, whoever wrote the frame.
- 14b0a57df clamps the switcher's id-3/66 writes while sup_runv_q is set: mstatus := SXL/UXL + the frame's FS, and
  satp.mode := Sv39.
- The final audit (443f67892..14b0a57df) found the clamp's timing SUPPORTED. Every switcher write of id 3/66 was
  enumerated: the RESUME's run with sup_runv_q = 1, the ESCAPE's caller restore with 0. One more consequence: an
  unclamped MPRV = 1 with MPP != M would also have disarmed the LSU's capability check (armed only at ld_st priv M).
- **Measured** (va-runv-bare; M rewrites +928 to Bare and sets MPRV|TVM at +920 between the escape and the reply):
  - E5 443f67892: the next escape saved mode 0 and 0x120000, so the RESUME installed both. The context then ended
    terminal, kind 2, cause 5.
    - The pre-registered "the unmapped load completes" is REFUTED in that half. The auditor's source reading predicts
      completion too.
    - OPEN: what raises the cause 5.
  - e5-native 14b0a57df: kind 4, cause 13, mode 8, 0, as predicted.
  - The control (frame untouched): kind 4, cause 13, mode 8, 0 on both trees.

**Arms on 14b0a57df** (vp8 .sup/nat2, dirty=no):
- va-runv-xfer: every commit-written rd as pre-registered. The escape-written rds sit outside the retirement trace;
  2cf56f02f adds self-moves, and its rerun is recorded below. The memory side shows A's state crossing detach, a run
  of B in the slot, and adopt: data[16] 0x11, data[24] 0xa5a5, data[32] 0xb0b0, frame A +16 3, B +16 2.
- On E5 the same arm diverges from its first DETACH on (ILLEGAL 2), the control that the arm exercises the path.
- va-runv, va-runv-quantum, va-runv2 and va-runv3: every reading equals its header. vp3-idx20-a1 and va-alloc-csret
  equal E5's.
- Instrument: the scratchpad copy of capprint-readings.py still had the M-only write filter (fixed 2026-10-17 on
  virtual-profile-2-d). It was patched before these readings; the earlier E5 arms print only M-mode-written registers.
- Rerun on 2cf56f02f (self-moves; core = 14b0a57df), vp9 .sup/nat3: va-runv-xfer reads its header on EVERY position,
  the escape-written rds included: 3, -1, 0, -1, -1, 2, 0, 0, -1, 3, 0x11, 0xa5a5, 0xb0b0, 3, 0, 2; 0x1d; 2, 2, 2; 0,
  -1, 0, 3, 0, 9. va-runv-bare 3, 4, 4, 13, 0, 8, 0; 0, 9 and -ctrl the same, as pre-registered.

**Lint on 14b0a57df** (rtl-ids20-wt, Anvil regenerated): LATCH 44, MULTIDRIVEN 3, ALWCOMBORDER 0, COMBDLY 0, UNOPTFLAT
39, ANVIL_UNOPTFLAT 0, BLKSEQ 2, UNDRIVEN 25, UNUSEDSIGNAL 1432, CASEOVERLAP 2, SELRANGE 100. The gate read FAIL against
the 1431 ceiling on UNUSEDSIGNAL only. The +2 over E5 (1430) is named by a diff with anvilc's wire numbers normalised:
the two copies of rs2.cursor that anvilc emits for the legality slices. REF re-baselined at 54413bc16.

**Synthesis (board lane, apollo)**: one full run with the gate keys graded in flight.
- Run 1: FAIL 8/10, stopped at 01:54, with clean synthesis (23.22 GB peak, 0 loop headers, 0 8-326).
  - core FF -303 vs the pre-registered +35 ±300.
  - dyn_unit +1,598 vs ±1,500, all in _spawn_0.
- **RETRACTION (this lane's key, 9 Nov)**: the rev_node / dyn_unit / lsu ±1,500 bands, and the commit / csr ±500
  rows, were written BELOW the recorded per-module redistribution floor of ±2,500 LUT (measured 2026-10-18; dyn_unit
  moved +1,001 source-identical). It is the second instance after 2026-10-22. The dyn_unit reading is not evidence
  about the change.
- The board lane's grade_gate.py now refuses any band below MODULE_LUT_FLOOR = 2500 / TOP_LUT_FLOOR = 2000 (exit 2,
  never widened). Its controls fire: the old e5native and e5 specs are refused.
- Core FF -303 ATTRIBUTED by flop name (board census, apollo ffcensus2):
  - dyn_child_repr_q 253 -> 1 (stage-B's backward retime gone, C.5a's form);
  - dyn_child_q.start 64 -> 0, merged into capstone_dyn_ftval_q (same enable and reset; ex_stage.sv:954 / :1280);
  - rev_rd_idx_q +2 (the 20-bit index);
  - csr_regfile +26 is source-identical cpmp churn plus replicas.
  - NO new state from the change.
- Run 2 (relaunched under catastrophic keys only): in-flight PASS 3/3; the post-synth report is identical to run 1
  in every row. Full build 4/4 vs E0:
  - placed congestion 4; check_timing 0 loops; WNS -13.953 (E0 -11.007, inside ±3.74); routed clean;
  - placed LUT 168,933 (-342), FF 63,650; router congestion max 4;
  - .bit sha256 b6698b7c19a132a38c1196b77c713292d95940b50de858e69d93eeb5d7e5a475 (the seal follows the collector).
- The flash waits for the one-variable control (leffe2's E5 gate) and the vector gate below.

**First-boot vectors (pre-registered here, before any board run)**: the 32 image sims on 14b0a57df (.sup/nimgA/B/C,
three worktrees, model rebuilt each) are all EQUAL to E0's column, objb-pcc empty on both.
- Driver: tests/rtl-smoke/evariants-2026-11-08/e5ngate/ (e5n-gate.sh, e5n-grade.py; e5n-vectors.txt, sha256 3374af09...).
  On the host it lives at perf/e5ngate.
- Grader controls and dry runs on every branch: DRYRUN.txt beside it.
- The rule is E0's: VALUE fields exact; TIMING fields against C.5a's silicon with the noise floor.

**e5-native SEALED (board lane, apollo, 9 Nov 03:15)**:
- caplifive_e5native_14b0a57df.bit, sha256 b6698b7c19a132a38c1196b77c713292d95940b50de858e69d93eeb5d7e5a475,
  11,443,722 B. The rtl lane re-hashed it from the store; it is written into perf/e5ngate/e5n-bit.sha.
- Tarball sha256 636f666a09cee828..., .bit cmp-identical, PROVENANCE HEAD 14b0a57df.
- A stubbed dry run against the real store passes the preconditions.
- Flash conditions: (1) the FF census, met; (3) the vectors, met; (2) leffe2's E5 control, pending.

**The native route's CSCHECKR/W + CSKINDS monitor emulation EXECUTED on 14b0a57df** (capstone-sbi e04ecb8; until now
it had run nowhere, since QEMU lacks CAPCREATE):
- every row of the arm's pre-registered header holds, each refusal 24..29 once;
- result lines are in tests/rtl-smoke/evariants-2026-11-08/cscheck-emu-rtl-14b0a57df.result-lines.txt.

**E5 synthesis gate on leffe2 (synth lane, 443f67892)** — the one-variable control for e5-native:
- core 144,677 LUT / 41,381 FF; ex 38,523 + issue 48,785 = 87,308 (E0 87,153); rev_node 3,078; lsu_i 15,755.
- dyn_unit 5,893 (E0 10,766).
- 0 loop headers, 0 Synth 8-326; mapping 5,409.7 MB; peak 22.45 GB.
- Retiming summary 1 / 26 / 93 / 25, equal to C.5a's.
Two keys failed by letter; ruled PASS by this lane:
- **FF**: E0 is the retiming OUTLIER. Its own log (apollo stageb2-a481031ca-gate-out/synth_1.runme.log:24307) reads
  forward 3 / backward 56 / 1,387 registers added / 28 deleted; C.5a's (ref-c5a :24310) reads 1 / 26 / 93 / 25. E5,
  e5-native, b3, C.5a and stage A all sit at 41,381-41,427 FF. **A second correction to this lane's keys**: the
  e5-native and E5 FF bands were centred on E0, a retiming outlier, so a ±300 band around it could not hold for a
  tree that does not retime the same way.
- **dyn_unit**: a hierarchy flip between ex_stage and issue_stage (ex -5,714, issue +5,869; IRO 33,404 vs stage A's
  26,079). It is graded by the ex+issue stem, the recorded rule of 2026-10-22: +155, PASS.
- e5-native minus E5: core LUT -62, FF +46, ex+issue -251. That is flash condition (2), MET.
- **e5-native is cleared to flash on every condition**; the flash is the runtime lane's. E5's full build continues
  on leffe2.

**E5 FULL BUILD (synth lane, leffe2, 443f67892)**: every pre-registered key PASS.
- placer congestion 4, router 5; routed clean; check_timing 0 loops (the matcher fires on vp3g);
- WNS -14.418 (band -11.007 ±3.74, margin 0.329); TNS -288,110; WHS +0.060;
- placed LUT 169,135, FF 63,604;
- .bit sha256 87edca3550f08c72471f71e107ab31dec3e6d93e2fd1a76173664eaef72918c1;
- tarball synth-443f67892-exit0.tar.gz, sha256 2221d5cf7c397bcc05cb2ff9a7b110b3acb784091b1466f36d5b9e3594844c5e.
Standalone E5 is now the CONTROL: it carries R-69. The board lane holds read-only copies of its gate report and routed
reports (apollo synth/ref-e5, sha256-matched). Graded against E5, e5-native passes 4/4 (WNS +0.465).

## E5c: E5 + the resume clamp only, the module-path fallback (R-69; the semantics owner's yes, 9 Nov)

capstone-ariane e5c **246e7abc2** = E5 443f67892 + e5-native's csr_regfile clamp hunks, byte-identical, + va-runv-bare.S.
The switcher is unchanged, so the final claim-audit of the clamp (every id-3/66 write enumerated, timing SUPPORTED)
applies as is.
- Lint (rtl-ids20-wt): PASS, identical to E5 on every counter (UNUSEDSIGNAL 1430).
- Arms (vp8 .sup/e5carms): satp-only, bare and ctrl read 3, 4, 4, 13, 0, 8, 0 (clamped). The va-runv family,
  vp3-idx20-a1 and va-alloc-csret equal e5-native's.
- Vectors: 31/32 = E0, objb-pcc empty on both.
- Stopping keys, coded by the board lane as --spec=e5c-stem: core LUT ±2,000 and ex+issue ±2,500 vs E5, plus loop
  headers. Every module row is reported, not graded. The any-module key was dropped: across the dyn_unit/issue flip
  single rows move 6-7k (e5-native vs E5: issue -7,650, ex +7,399), so it would stop a healthy run at random.
- Full build graded against E5's routed WNS (-14.418 ±3.74).
- Flash gate: tests/rtl-smoke/evariants-2026-11-08/e5cgate (pre-registered here, before any board run).
