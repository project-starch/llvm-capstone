# Bare-metal baseline works: I-2 solved, every overhead rises, and §3 INVERTS

**Date:** 2026-07-28
**Lane:** C (primary)
**Cost:** 4 board boots (3 failed on console bring-up), plus off-board QEMU validation.
**Status:** the baseline confound is **removed at the source**, not modelled.

---

## The proof that it works

`ctrsanity` exists precisely to answer "is a baseline cycle comparable to a domain
cycle". Its clean value was known independently from the capability side:

| | cycles | instret | cyc/iter |
|---|---:|---:|---:|
| capability domain | 600,309 | 500,033 | 6.003 |
| **bare-metal baseline** | **600,041** | **500,022** | **6.000** |
| old Linux baseline | 728,727 | 509,178 | 7.287 |

**Ratio 1.000.** The identical five-instruction loop now costs the same on both sides.
The entire 1.21× discrepancy was the operating system. I-2 is closed.

Measurement quality is categorically different: **15/15 passes tied at minimum instret
with spread = 0** on nearly every rung (`beebs_fac` spread 6). The Linux baseline never
managed better than 1/15 on anything long.

## Corrected overhead table — every number rises

| rung | opt | **cycles** | was | **instr** | cap CPI | base CPI |
|---|---|---:|---:|---:|---:|---:|
| `beebs_bs` | −O1 | **1.530×** | 1.274 | 1.058× | 2.581 | 1.785 |
| `beebs_prime` | −O0 | **1.683×** | 1.032 | — | — | 2.260 |
| `beebs_cnt` | −O1 | **1.353×** | 1.165 | 1.319× | 1.677 | 1.635 |
| `rv8_primes` | −O0 | **1.263×** | 1.050 | 1.130× | 1.970 | 1.762 |
| `beebs_recursion` | −O1 | **1.955×** | 1.801 | 1.458× | 6.439 | 4.802 |
| `ctrsanity` (control) | −O1 | 1.000× | — | 1.000× | 1.201 | 1.200 |

**`beebs_prime` went from 1.032× to 1.683×.** The paper's flagship "3.2 % scalar" figure
was understated by a factor of ~20 in overhead terms. Pervasive spatial safety costs
**26 % to 96 %** across these kernels, not 3–5 %.

Two caveats on provenance, stated rather than buried:
- `beebs_recursion`'s capability instret (2,944) is back-computed from the published
  1.458× ratio against the old baseline; it should be re-measured directly.
- `beebs_prime` has no capability instret at all (instrumenting its `domain_main`
  changes the value it computes — a separate, still-unexplained silicon effect).

## §3 INVERTS — "the overhead is ABI, not hardware" is REFUTED

The claim rested on `rv8_primes` retiring **more instructions than it cost cycles**, with
CPI falling — read as "bounds enforcement is near-free per instruction; the cost is our
globals ABI". Against a clean baseline it reverses:

| | cycles | instructions | CPI |
|---|---:|---:|---:|
| baseline (bare) | 13,679,903 | 7,764,899 | **1.762** |
| capability | 17,283,292 | 8,773,753 | **1.970** |
| ratio | **1.263×** | **1.130×** | CPI **RISES** |

Cycles grow **faster** than instructions. The earlier "CPI falls 2.07 → 1.98" was an
artifact of interrupts inflating the *baseline's* CPI — interrupt handling runs at ~14
cycles per instruction against real code's ~1.8, so it raised exactly the quantity the
argument turned on.

The prediction from the calibration estimate was baseline CPI ≈ 1.742; measured 1.762.
The estimate was sound, and so is its conclusion: **we cannot claim capability
enforcement is free per instruction.** On every rung with both counters, cycles grow
faster than instruction count.

## How the bare-metal baseline is built

OpenSBI boots our S-mode program in place of the Linux Image. Firmware is assembled by
reusing the known-good OpenSBI+FDT prefix of the existing `fw_payload` and substituting
our payload at offset `0x200000` (→ `0x80200000`, confirmed by OpenSBI's own
`Domain0 Next Address`). The rung kernels are the **same** `ladder_base_kern.c` objects
the Linux controller links, so the measured code is unchanged; only the harness differs.

Side benefit, and a large one: firmware drops from **15.4 MB to 2.1 MB**. The JTAG reload
dominates every board boot, so this makes the board substantially faster as well as
correct.

`-mcmodel=medany` is required (the payload sits above the low 2 GB that medlow's
`lui/addi` can reach). Static instruction counts match medlow on 3 of 4 sampled rungs;
`beebs_prime` differs by 4. Its measured bare instret (12,562) is within 0.03 % of the
Linux min-of-16 value (12,558), so the code model is not perturbing the comparison.

## Console bring-up: three failures, and the lesson

Three board sessions produced the OpenSBI banner and then silence.

1. **Legacy SBI console (EID 0x01)** — works under QEMU's OpenSBI v1.3.1, absent here.
2. **DBCN** — added in SBI **2.0**; this board reports **`Runtime SBI Version: 1.0`**, so
   it cannot exist. Also, the probe was reading `a0` (the error code) instead of `a1`
   (the value), so DBCN was never selected even where present.
3. **Direct ns16550a MMIO** — works. Parameters came from the firmware's own device tree
   (`/soc/uart@10000000`, `reg-shift=2`, `io-width=4`; PMP Region05 grants S/U access),
   not from guesswork. QEMU's virt has the same chip at the same address with shift 0,
   so the shift is a build parameter.

**The lesson: the device tree had the answer on disk the whole time.** Two board sessions
were spent guessing at firmware features when `fdt` parsing was free and definitive. Go
to the authoritative artifact before spending a scarce resource.

A fourth run then produced every label with **blank numbers** —
`BASE RESULT ctrsanity4 pass= retval= cycles= instret=` — because `puts_` had been routed
to the new console and `putu_` had not. The run had succeeded; only the digits were lost.

## Two instrumentation bugs worth remembering

- **The runner reported "0 UART chars" on a fully successful run.** It baselines
  `start = len(uart_text)` before `continue`, but the console clears its buffer on
  reboot, so the index exceeded the new length and every slice returned empty. Now falls
  back to the whole buffer. Monitoring that reports failure on success is worse than no
  monitoring.
- **A build check reported "BUILDS" for failing builds** because it grepped output for
  error strings without testing the exit code. Gate on exit status.

## What this unblocks

- `rv8_primes` is measurable for the first time — it was permanently out of reach of the
  repeat-and-take-minimum trick (16.5 M cycles ≫ one timer tick).
- The full 13-rung re-measurement can now be done against a trustworthy denominator, and
  with the R-3 one-boot workaround it costs ~2 boots rather than ~26.
- SQLite becomes meaningful: it runs long, so it was firmly in the regime where the Linux
  baseline was wrong.
