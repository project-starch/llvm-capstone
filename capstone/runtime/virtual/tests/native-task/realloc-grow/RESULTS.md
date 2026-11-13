# Patch 0011 and the cache's evict-to-fit on the native twin: results (2026-11-13)

Pre-registered in PREREG.md (e9ecb21b888d for A/B, 3e70344d8c3f for C) before any arm ran. Native twin: kernel
perf/natio/images (native-io-align k3), 936M, direct arm, --no-sampler, capmix3 shift 24 around each command; the 0x80
bucket (kernel text at its physical address, plus OpenSBI ~0.01 M) is counted as kernel. Tools in tools/
(mkvm.py stages, compare.py readings, census.py diag_admit census).

A = nat-fair-cache eb73a2bf475c, B = A + musl 0011 (f097dd782aae), C = B + mr_make_room (3e70344d8c3f).

| benchmark | kernel A | B vs A | C vs A | pages A -> C | outputs A/B/C |
|---|---|---|---|---|---|
| 505.mcf_r test | 2,484,346,964 | -35.77% | -63.13% | 392,471 -> 138,140 | identical, 6 files, specdiff 2/2 |
| 502.gcc_r train01 | 472,207,605 | +0.25% | -3.19% | 71,511 -> 69,716 | identical, 5 files, specdiff 1/1 |
| 538.imagick_r test | 30,873,924 | +0.38% | +0.32% | 4,806 -> 4,808 | identical, 8 files, specdiff 1/1 |
| 557.xz_r test | 10,675,211,449 | +0.00% | -0.00% | 1,729,410 -> 1,729,393 | identical, 37 files, 12/12 |
| 500.perlbench_r test | 3,283,855,993 | +0.00% | -0.00% | 199,525 -> 199,514 | identical, 13 files, 1/2 on every arm |

C alone (plain twins): 519.lbm_r and 544.nab_r outputs identical to nat-fair-cache's own twins (5 and 3 files).
User instructions move by at most 0.7% (mcf: B -182 M, C -85 M against A).

Against the predictions:
- B, mcf: pages 248,406 (200k-270k: HIT); kernel -0.889 G (-0.8..-1.3: HIT); user -182 M (-20..-150 M: MISSED).
- C against B, mcf: pages -110,266 (-90k..-125k: HIT); kernel -0.680 G (-0.5..-0.8: HIT); user +97 M (+30..+150: HIT).
- gcc train01 guard: service_other A 2,600, B 2,593, C 2,303 (-11.2% against B; the guard was +15%).
- Every other benchmark: outputs identical, kernel within 5% (HIT).

mapping-reuse-probe on the native twin, A/B/C: the same verdicts in every mode (control, realloc, reuse, alias,
protect, stale200, stale16, pressure 253 blocks); realloc hits 2 on A and 1 on B/C, because a growth now moves into a
mapping with room, so the later same-size malloc meets a cached mapping of another length (its zero check passes).

## The riscv-base counterpart (riscv-base/)

The board arm compares Capstone with riscv-base musl whose plain_reuse.c caches mappings with the same bounds. That
cache never reused an individually mmapped chunk: malloc.c maps it with the exact request (n + IB + UNIT), free.c
releases it with maplen*4096, so a cached length never equals a later request. riscv-base mcf test (rv64fr objects
relinked; the plain relink reproduces the exe, 05e07af3ffd0ac36), kit-profq QEMU, plain-map-wrap.c logging:

| plain_reuse | large mappings reused | pgfault |
|---|---|---|
| as built | 0 of 13 | 132,539 |
| + evict (C's rule) | 0 of 13 | - |
| + round | 0 of 13 | - |
| + round + evict | 6 of 8 arcs_pointer_sorted | 85,654 |

specdiff 2/2 in every variant. The fair pair for C is riscv-base with plain-reuse-round-evict.diff.
