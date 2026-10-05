# The plain-malloc temporal baseline: 5 fixtures, 10 arms, 37 cells (2026-10-05)

**Why this run exists.** The inventory's question-(b) table had a row with a dash in it: the **five
not-nested temporal defects** — memcached fixtures 17 and 18, tshark 14 and 15, FFmpeg 24 — were
counted in the defect totals but had no arm readings beside them. A dash is indistinguishable from
an unexamined cell, and this row is the **denominator**: it is where a plain `malloc`/`free`
use-after-free can be contrasted against the nested ones, and the only place a quarantine-based
system can register a temporal hit at all.

**37 of 37 cells as predicted**, against each port's own unchanged `host/safety-expect.txt`.
`result-lines.txt` has every row.

## The row, read off the cells

| arm | caught | of | what it means |
|---|---:|---:|---|
| `level0` (no protection) | 0 | 5 | the control: nothing is enforced, nothing faults |
| `shrink` (**Capstone without extra protection**) | **0** | **5** | per-object bounds are no help here |
| `sublet` | **5** | **5** | revocation catches every one |
| `chunks` (tshark's nested port) | 2 | 2 | unchanged by the inner-allocator port, as expected |

Every `sublet` catch reads **cause 24** — revoked authority — not a bounds cause. The axis is clean.

**`shrink` returning on all five is the finding, not a gap.** These objects really do reach
`free()`; a bound still cannot see a dead object, because the stale address is in bounds by
construction. Filling this cell with a catch would mean the arm was broken.

**Controls.** fx2 `heap_neighbour` and fx3 `heap_one_past` ran in **every one of the ten boots**:
`level0` RETURN, and FAULT `oob` on `shrink`, `sublet` and `chunks` in all three ports (cause 7 on
the write probe, cause 5 on the read). So a changed reading in the rows above would have been
attributable to the fixture rather than to the boot, and the two axes are separated within each
boot — the same image that returns on a temporal fixture faults on a spatial one.

## The CheriBSD prediction for these five, stated so it can be refuted

Not measured here — no SDK, purecap sysroot or image on this host
(`ports/common/cmake/toolchains/cheribsd.cmake:4-9`). But unlike the 22 nested cases, these five
**are predicted to be CAUGHT**, and the reason is the same mechanism that predicts the nested zero:

> the nested cases complete under stock CheriBSD because the stale storage never reaches `free()` —
> it goes back on an inner allocator's own free list inside a block `malloc` still owns, so the
> quarantine never holds it. **These five have no inner allocator.** The object is a plain
> `malloc`/`g_malloc`/`av_malloc` allocation that is genuinely freed, so libc's quarantine does hold
> it and the revoker does sweep it.

**This is the first cell in the CheriBSD column predicted non-zero, and it is what gives the
measured 0 of 18 its meaning.** A system that caught nothing anywhere would be indistinguishable
from a dead instrument; one that catches the plain cases and misses the nested ones is measuring the
nesting. It is therefore **the first thing an SDK host should run**, and a miss here would refute
the mechanism rather than merely add a data point.

## Platform and caveats

- `deleg-gate2/qemu-12` with `pinned-platform/images-root`'s `fw_jump.elf`, `cma=1536M`,
  `process_cache_bytes=402653184`, `CAPSTONE_GP_NONLIN=1`, `CAPSTONE_REV_NODES=65536`.
- **QEMU only.** A `sublet` temporal fault here is the emulator untagging a revoked capability on
  reload (Q-11); deployed silicon lets such an access retire. These are detections of the *defect*,
  not claims about silicon behaviour.
- **N = 1 per cell**, ten boots.
- memcached's arms run through the per-arm safety images (`level0` `9beb8bbfebdf04a1`, `shrink`
  `c8c1a1596787f908`, `sublet` `b5b340c630e0ea9d`); tshark and FFmpeg through their per-fixture
  images in `domain`, `domain-shrink`, `domain-sublet` and `domain-chunks`.
- memcached is driven through the runner's memcached path, added in `eda506566052` and
  negative-tested there.
