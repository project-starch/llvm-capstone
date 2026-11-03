# C.5a verdict-cache hit rates on SPEC test (QEMU counting model), 3 November

Option C.5a lets LDC and STC take the revocation-validity cache's (RVC) cached verdict instead of the per-access
lifetime check. This run counts how often that cache would hit on the SPEC test inputs. It is an input to the C.5a
bitstream decision, not a gate.

**Instrument.** capstone-qemu 0f271af7befc with `x-capstone-lc-model=true`
(`target/riscv/capstone_lc_model.c`). The model copies bitstream 3's (e7af48457) RVC: 64 sets × 4 ways, it
installs only live evidence, a record written invalid clears its way, and a namespace change flushes the cache.
- Instance **R** is that cache.
- Instance **E** is the same cache, also flushed at every revocation (an epoch scheme).
- Windows open and close at the `slti` markers 0x7a1/0x7a2 placed around each SPEC command. Each window prints one
  `capstone: lc-model window=N ...` line to the guest serial log.

**Images.** These are the SPEC test stages of 3 November, the same images as the board:
- `capstone-lifetimes` is Temporal (Full, ObjLife).
- `capstone-spatial` is MapLife.
- Each stage ran as a snapshot-class QEMU session in its own slot (`CAPSTONE_QEMU_DOMAIN_LOCK`).
- Every run is rc 0 with specdiff PASS.

**Reading.** `lcsum.py <dir holding outlc/>` sums the windows of each arm, using the arm's command count from
`result.json`. It exits non-zero when it finds no data.

| benchmark | Temporal miss, R | conflict share of it | notes |
|---|---|---|---|
| lbm | 0.001 % | 0 | |
| nab | 0.004 % | 0.003 % | |
| imagick | 0.033 % | 0 | |
| xz | 0.062 % | 0 | MapLife identical |
| perlbench | 0.113 % | 0.037 % | E: 0.518 % |
| gcc | 0.169 % | 0 | context is small: 6.3 M LDC+STC |
| mcf | 1.883 % | 1.813 % | 173 M conflict misses on LDC; MapLife 0.006 % |

Full table: `SUMMARY.txt`.

**Neither event occurs in these runs.**
- `hit_dead` = 0: SPEC never accesses a dead object, so these runs show nothing about the dead path.
- `disagree` = 0 in every run.

**Estimate, not a measurement.** C.5a's measured miss cost is 12-13 cycles over ObjBounds; a hit costs what ObjBounds
costs.
- mcf: about 2.2 G cycles, roughly 0.7-0.8 % of its run.
- Every other benchmark: below 0.1 %.
- So C.5a predicts Temporal ≈ Spatial + < 1 % on SPEC test. Today, measured on the board, Temporal / Spatial is
  1.004-1.105.

**Caveats.**
- perlbench's MapLife arm executed 23 G LDC+STC against Temporal's 12 G, so the two arms did different work per
  window. Compare their rates, not their counts.
- The model has no timing. It counts outcomes only.
