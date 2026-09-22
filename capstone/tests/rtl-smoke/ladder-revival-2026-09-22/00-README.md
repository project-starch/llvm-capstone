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

The baseline half ran on the same bitstream (`baseline-half.result-lines.txt`). Pairing its **warm**
column — the one the tool's own header names as comparable, *"the domain half has no paging, so WARM
is the comparable column"* — against the capability half gives:

| rung | cap instret | base instret | instr ratio | cycle ratio | idem |
|---|---:|---:|---:|---:|---|
| **`ctrsanity`** (control) | 500,030 | 538,935 | **0.928** | **0.877** | YES |
| `beebs_prime` | 2,703 | 2,704 | 1.000 | 1.000 | YES |
| `matmult_int` | 9,722 | 7,272 | 1.337 | 1.541 | YES |
| `coremark_matrix` | 31,560 | 28,071 | 1.124 | 1.055 | YES |
| `beebs_crc32` | 29,723 | 33,752 | 0.881 | 0.844 | **no** |
| `beebs_insertsort` | 962 | 813 | 1.183 | 1.484 | YES |
| `beebs_janne` | 198 | 211 | **0.938** | 1.090 | YES |

**`ctrsanity` is the control, and its defining property is that both halves run identical code.**
Published 2026-07-28 it read 500,033 against 500,022 — eleven instructions apart. Today it reads
**500,030 against 538,935 — 38,905 apart, 7.8 %.** The halves have diverged.

The pre-registration named this exact condition in advance:

> If any anchor's instruction count differs from its published value by more than 1 %, the vehicle is
> not comparable with the published table. **No new row may be added to §2.**

and the verification section said the same of the control specifically: *"If `ctrsanity` does not read
1.000×, the vehicle is not measuring what the published table measured, and no new row may be added."*

**So no row is added.** Two further readings agree with the control rather than against it:
`beebs_janne` at **0.938** and `beebs_crc32` at **0.881** are instruction ratios *below 1.0* —
capabilities retiring fewer instructions than plain RISC-V, which the ABI cannot do. And the raw
16-pass data varies wildly on identical code (`matmult_int` instret 7,272 … 13,687), because the
baseline half runs as ordinary Linux userspace and its counters take scheduler interference that the
domain half does not.

Four rungs produced no baseline at all — `beebs_aha_mont64`, `rv8_sha512`, `rv8_sha512s`, `beebs_ns`
(`--`, `correct=NO`), which is why the runner exited 1. The cause is **not** established and is not
size-ordered: 538k-instruction `ctrsanity` completed while 256k `beebs_aha_mont64` did not.

**What this means.** The coverage result below stands — it is about correctness, and the capability
half is scored against native oracles that need no denominator. The *overhead* half of §2 cannot be
regenerated until the baseline vehicle is repaired. That is a separate piece of work and it is
cheaper than it looks, because it is a host-side measurement problem, not a board one.

## What this does NOT establish

- **No ratios are claimed here, and the numbers in the table above must not be quoted as overheads.**
  They are printed to show the control failing, which is the finding.
- **No wall-clock or MHz-normalised figure.** Timing does not close on this build (WNS −8.307).
- **Nothing about why each blocker retired.** Four rungs' entries name R-1, which is recorded GONE;
  the other three name R-6, R-7 and R-9 and are not investigated here. "It measures now" is the
  claim; "R-1's fix is why" is not.
