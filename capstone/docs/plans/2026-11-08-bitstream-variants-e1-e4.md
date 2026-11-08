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
