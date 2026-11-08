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
