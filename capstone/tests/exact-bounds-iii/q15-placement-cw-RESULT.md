# Q-15 placement repair on the nomovc-cw line: the reading (2026-10-20), against q15-placement-cw-PREREGISTRATION.md

Runs at e02dd7f9908f's runners; images exactly the pre-registered hashes. Kits: qkit7-eb (QEMU 7a4f3a64, module 5bfc38f8,
exact bounds), qkit7-tB-mimic (7e5fe043, module a1fa252f; SWITCHES movc_null_scalar=1 compact_rounding=ON r60/r59 OFF),
qkit7-nf2 (7a4f3a64, module a1fa252f), and, added after the step-2 reading to attribute it, qkit7 (7a4f3a64, compact,
module 5bfc38f8: the original step 2's kit). Comparator fields as pre-registered (offset is not one of them).

| arm | step 1 (eb) vs record | step 2 (tB-mimic) vs step 1 | plain nf2 vs tB-mimic | cause=29 lines (tB / nf2) |
|---|---|---|---|---|
| pb13 | 8/9 identical + 08 NO-READING control-failed (pre-registered pad-masked) | 9/9 | 9/9 | 0 / 0 (1498 / 1497 lines) |
| pgm1 | 5/5 | 5/5 | 5/5 | 0 / 0 |
| pg-pools (sql) | 9/9 (4 offset-only moves: memcpy +200 -> +220, pg_popcount +288 -> +292) | **7/9: FAILS** | 9/9 | 0 / 0 (20416 / 20410 lines) |

Counter control: t3-q15-c-pb13's serial reads 4 `cause=29` lines.

**sql step 2 fails its prediction on qkit7-tB-mimic:** 03 (ltree_lquery_uint16_overflow) CAUGHT -> NO-READING
("this image cannot create ltree"), and 07 (pg_trgm_picksplit) from cause 28 at pg_popcount_optimized to cause 30
(INSUF_RESOURCES) at AllocSetAlloc. Plain nf2 reads the same, so no silicon switch moves them. Only the tB/nf2 consoles
carry `capstone_vm: collect skipped: collection is not available on bf3db3175 (Q-27, S-entry)` (5 lines; eb 0).
**Attribution run (one variable: the module):** the same image on qkit7 (compact, module 5bfc38f8, which collects on
QEMU) reads 9/9 identical to step 1, 0 cause=29, 0 collect-skipped (20648 lines). So the compact profile and the repair
pass; the two cells move because the board's module cannot collect (Q-27) and pg-pools exhausts the flat table without
collection, the shape cpython lt1 showed. pg-pools does not transfer to the current bitstream until collection exists.

Masked rows: 1 (pb13/08, no reading, pad-masked: first_oob_offset 0 < pad_bytes 16; 16400 -> 16416).
Q15_PAD: pb13 pad_bytes_max 16 (cache path); pgm1 0; sql up to 574 padded chunks, pad_bytes_total <= 10256, max 992.
