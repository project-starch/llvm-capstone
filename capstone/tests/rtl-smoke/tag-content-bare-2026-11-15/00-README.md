# tag-content-bare-2026-11-15

Do the DRAM tag bytes come out right after plain overwrites of tagged memory, on silicon? The adapter frees a store's
tag write when its tag-word cache (TWC) holds the word and the cached byte already equals the store's tag
(capstone-ariane 9fd9d2530 `wt_axi_adapter.sv:299-301`). A stale entry reading 0 over a DRAM 1 would let a plain store
skip its clear. The stale tag then surfaces only after the entry dies, and no output check sees it. F2's longer-lived
entries and the overlap lever's concurrency both lean on that rule. The image reads the tag bytes back RAW (plain byte
loads from the tag region) and counts them against the bytes its store sequence implies.

`build.sh` rebuilds the image from `tests/tag-content.S`, a copy of capstone-ariane
`verif/tests/custom/capstone/tag-content.S` at sim/tag-audit 2702f75d4 (tests only; checked against `VP_SRC` when
given). The wrapper is `fp-rm-bare-2026-10-26`'s (inc/ copied). `images/SHA256SUMS` pins the image. `PREREG.md` holds
the vector simulated from THIS image on F, F2 and G, and on the mutant control.

Runner: `sup-bare-2026-10-03-ladder/run_sup_bare.py` (`SUP_IMG`, `SUP_REC_ADDR` from the `.rec`), the bare-boot path only.
Related: the sim-only audit (`TAG_AUDIT`, capstone-ariane sim/tag-audit e3fcf2b7d) and the twc-reach plan
(`capstone/docs/plans/2026-11-14-twc-reach.md`, section "G on silicon, mcf").
