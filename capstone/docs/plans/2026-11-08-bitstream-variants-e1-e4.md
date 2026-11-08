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
