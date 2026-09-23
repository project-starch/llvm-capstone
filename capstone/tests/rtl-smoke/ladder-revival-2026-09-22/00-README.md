# Silicon-ladder revival, 2026-09-22 — the eight blocked rungs all measure

Bitstream **`caplifive_m1_054cea69b.bit`** (resident, not reflashed; WNS −8.307, timing does not
close). Monitor `2dcd3a5`. Pre-registration: `../ladder-revival-2026-09-22.prereg.md`, committed
before the first boot.

## The question

`docs/ref/fpga-silicon-measurements-for-paper.md` §2 publishes **eight** rungs and lists **eight
more that do not appear**, each with *"a clean, correct baseline half"*, blocked on R-1 (four named
directly), R-6, R-7 and R-9. `ISSUES-ARCHIVE.md:25` has recorded R-1 as **`GONE 2026-09-05`** for
seventeen days. The table was never re-run.

## Result — all eight return their correct oracle

Every rung in `capability-half.result-lines.txt` reads `correct = YES`, scored against the native
oracle rather than against "it returned":

| rung | prior blocker | now |
|---|---|---|
| `matmult_int` | R-1 | **correct** |
| `coremark_matrix` | R-1 | **correct** |
| `beebs_crc32` | R-1 | **correct** |
| `beebs_insertsort` | R-1 | **correct** |
| `beebs_janne` | **R-6** | **correct** |
| `rv8_sha512` | **R-7 → R-1** | **correct** |
| `rv8_sha512s` | R-1 | **correct** |
| `beebs_ns` | **R-9** | **correct** |

The three rungs whose blockers were **not** R-1 pass too. That is worth stating separately, because
the archive is explicit that *"two of the eight are not explained by R-1"* and that writing "all
remaining failures are the register-indexed-load defect" was falsified the next day. This run does
not re-assert that claim; it records that all eight now measure, whatever retired each blocker.

`rawhazard5/6/7` — R-1's own probes — read **5** on every live slot, so R-1 is absent on this
bitstream by the same instrument the archive used to retire it.

## The vehicle was validated before any new row was believed

Pre-registered refutation condition: *if any anchor's instruction count moves more than 1 % from its
published value, the vehicle is not comparable and no new row may be added.* It did not fire.

| anchor | instret now | instret 2026-07-28 | Δ |
|---|---:|---:|---:|
| `ctrsanity` (control, identical code both sides) | 500,030 | 500,033 | 0.0006 % |
| `beebs_prime` | 2,703 | 2,708 | 0.18 % |
| `beebs_aha_mont64` | 256,697 | 256,699 | 0.0008 % |

## ⚠ Cycles did NOT carry over, and the published table's cycles must not be reused

On the same three anchors, with instruction counts essentially identical:

| anchor | CPI 2026-07-28 | CPI now | Δ |
|---|---:|---:|---:|
| `ctrsanity` | 1.2005 | 1.4005 | **+16.7 %** |
| `beebs_prime` | 3.6123 | 3.5875 | −0.7 % |
| `beebs_aha_mont64` | 1.1300 | 1.1160 | −1.2 % |

`ctrsanity` is the control — identical code on both halves — and its CPI moved 16.7 % on unchanged
instructions. So **a new capability-half cycle count paired with a published baseline cycle count
would measure the bitstream, not the ABI.** The baseline half is therefore re-measured on this
silicon rather than reused.

The same conclusion falls out of the instruction counts alone. Pairing this run's capability instret
against the July baselines printed at §2 `:260` gives:

| rung | cap instret (now) | July base instret | ratio |
|---|---:|---:|---:|
| `rv8_sha512` | 460,950 | 462,646 | **0.996** |
| `rv8_sha512s` | 75,477 | 69,108 | 1.092 |
| `beebs_ns` | 61,299 | 62,097 | **0.987** |

Two of three land **below 1.0** — capabilities appearing to retire *fewer* instructions than plain
RISC-V. That is eight weeks of compiler movement, not a capability effect, and it is what a
cross-vintage pairing buys you.

## Deviation from the pre-registration, disclosed

The pre-registration said to run with `LADDER_ONE_BOOT=0`, one boot per rung, so that a wedge could
not cost the remaining rungs. **The runs used `LADDER_ONE_BOOT=1` / `LADDER_DISTINCT_VA=1`** — the
launcher's documented defaults and the standing R-3 workaround — which is **one power-cycle per
sweep** with each rung relocated to its own entry VA (`0x10000`, `0x20000`, … logged once per rung).
Verified in the logs: one `power-cycle + reload firmware` per sweep, and 7 of boot 2's 8 rungs report
`controller already on the guest, skipping transfer`.

The caution was unnecessary — R-3 is handled by the distinct VAs, not by separate boots — and no rung
wedged, so nothing was lost. Recorded because a pre-registration that is silently departed from is
not a pre-registration.

## The baseline half was measured on this silicon, and the CONTROL FAILED

**RETRACTION, same day.** This section first reported the control as 7.8 % out and attributed it to
scheduler interference. **Both the magnitude and the mechanism were wrong**, and the error was in
reading the runner's `warm` column as a measurement. It is one sample from a wide distribution, not a
floor. Corrected below; the *conclusion* — no new overhead row — is unchanged.

### The runner's `warm` column is not a clean measurement

The baseline runs each rung **16 times** and prints one `warm` value. Across those passes, on
identical code:

| rung | instret values seen | spread |
|---|---|---:|
| `beebs_prime` | 2,704 … 12,172 | **350 %** |
| `matmult_int` | 7,272 … 13,687 | 88 % |
| `ctrsanity` | 509,212 … 552,689 | 8.5 % |

The baseline half runs as ordinary Linux userspace, so its counters take scheduler interference the
domain half never sees. Interference only ever *adds*, so **the per-pass minimum is the clean
statistic** — and it verifies: `beebs_prime`'s minimum is **2,704, exactly the published July
baseline**. The `warm` column is not the minimum (`ctrsanity` warm = 538,935 against a floor of
509,212), which is where the retracted 7.8 % came from.

### Recomputed from per-pass minima

| rung | cap instret | base instret (min) | instr ratio | cycle ratio |
|---|---:|---:|---:|---:|
| **`ctrsanity`** (control) | 500,030 | 509,212 | **0.982** | 1.060 |
| `beebs_prime` | 2,703 | 2,704 | 1.000 | 1.017 |
| `matmult_int` | 9,722 | 7,272 | 1.337 | 1.591 |
| `coremark_matrix` | 31,560 | 25,666 | 1.230 | 1.391 |
| `beebs_crc32` | 29,723 | 31,001 | **0.959** | 1.115 |
| `beebs_insertsort` | 962 | 813 | 1.183 | 1.627 |
| `beebs_janne` | 198 | 211 | **0.938** | 1.855 |

### The control still fails, by 1.8 % rather than 7.8 %

`ctrsanity`'s defining property is that both halves run identical code; published, they were eleven
instructions apart (500,033 vs 500,022). Today the capability half is unchanged at 500,030, and the
baseline **floor** is 509,212 — **+1.8 % over its own published value**, on a rung whose two halves
are supposed to be identical. Interference cannot explain a raised floor, and the `-O` levels match:
`optlevels.txt` reads `-O1` on both sides.

So a residual remains after the interference is accounted for, it is real, and **its cause is not
established.** Two further rungs read instruction ratios below 1.0 — `beebs_janne` 0.938 and
`beebs_crc32` 0.959 — which the ABI cannot produce.

**No new overhead row is added**, because a control that does not read 1.000 cannot separate a
capability cost from whatever is moving the control. The plan's verification section set exactly this
condition before the run: *"If `ctrsanity` does not read 1.000×, the vehicle is not measuring what the
published table measured, and no new row may be added."*

**Note on which gate actually fired.** The committed pre-registration's own numbered gate was *any
anchor's instruction count moving more than 1 % from its published value* — and that gate **PASSED**
(0.0006 %, 0.18 %, 0.0008 %). It could not have caught this: it compares the capability half against
its own July value, so a drift on the *baseline* side is invisible to it. The condition that fired
came from the plan's verification section, not from the pre-registration. **The pre-registered gate
was the wrong discriminator**, and that is worth more than the result it missed: a gate on one half
cannot police a ratio.

### Four rungs produced no baseline

`beebs_aha_mont64`, `rv8_sha512`, `rv8_sha512s`, `beebs_ns` returned `--` / `correct=NO`, which is why
the runner exited 1. They stop after `beebs_janne` with a `^C` in the UART capture, which is
**consistent with the runner's per-rung timeout firing** rather than with a rung defect. Not
confirmed, but there is a signature and it is not size-ordered in the way a memory limit would be:
`ctrsanity`'s 500k instructions completed while `beebs_aha_mont64`'s 256k did not.

### What this means

The coverage result above stands — it is about correctness, scored against native oracles, and needs
no denominator. The *overhead* half of §2 cannot be regenerated until the baseline vehicle reports a
floor rather than a sample and the control's residual 1.8 % is explained. Both are host-side
measurement work, not board time, and the first is cheap: take the minimum across the passes the
runner already performs.

## What this does NOT establish

- **No ratios are claimed here, and the numbers in the table above must not be quoted as overheads.**
  They are printed to show the control failing, which is the finding.
- **No wall-clock or MHz-normalised figure.** Timing does not close on this build (WNS −8.307).
- **Nothing about why each blocker retired.** Four rungs' entries name R-1, which is recorded GONE;
  the other three name R-6, R-7 and R-9 and are not investigated here. "It measures now" is the
  claim; "R-1's fix is why" is not.

## Why `bare-metal-baseline.result-lines.txt` carries all 65 rungs, not the paired subset

Only a handful of these rungs pair against a capability half in this folder, so the file looks like
an over-capture. It is kept whole deliberately, for two reasons.

**It is the instrument's own certificate.** The sweep is what established that the bare-metal
baseline reproduces the July denominators *to the digit* after two months and several bitstreams
(`ctrsanity` 600,041/500,022, `beebs_prime` 9,283/2,704, `rv8_sha512` 540,073/462,646), at `15/15`
passes tied at min instret with spread 0 on nearly every rung. That claim is about the **instrument
across its whole range**, and a trimmed file cannot support it — the rows that do not pair here are
exactly the ones that make it more than an anecdote.

**It is the denominator source for the next rung.** These are 65 floors measured in one 81-second
boot on one bitstream. Any future capability-half rung drawn from `ladder-rungs.spec` has its
baseline here already and needs no second board session.

This is **result lines, not a capture** — 65 parsed rows, not the 1040-row UART log they came from,
which is not committed.
