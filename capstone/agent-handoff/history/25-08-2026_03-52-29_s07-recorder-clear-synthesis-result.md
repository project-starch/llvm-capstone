# S-07 recorder clear: synthesises with a bitstream, and costs 2.887 ns it probably did not cause

**Build:** `84ed6eafb`, branch `s07-recorder-clear-39b`, base `39b21639d`. exit 0, 1h31m58s
synthesis, 2h09m01s including the collector. Peak synthesis RSS 21.00 GB.
**Bitstream PRESENT** — `write_bitstream completed successfully`, 11,443,722 bytes.
**No `DRC LUTLP-1`** (count 0, with the same matcher returning 669 on DRC/Rule lines, so the file
was read and the zero is real).

## The numbers, on the CPU clock

Always `clk_out1_xlnx_clk_gen`. The first "Failing Endpoints" line in these reports is `eth_rxck`,
an unrelated Ethernet I/O clock that reads a healthy **+4.374 ns / 0 failing** while the CPU clock
is failing badly. Quoting it is a trap this project has already nearly fallen into once.

| build | S-10 fix | lint gate | WNS | failing endpoints |
|---|---|---|---|---|
| `39b21639d` (base) | absent | PASS | **−10.629** | 96,727 |
| `84ed6eafb` (base + clear) | absent | PASS | **−13.516** | 103,197 |
| `80843404c` (flown for most of the investigation) | present | FAIL | **−16.400** | 102,769 |

**The change is 2.887 ns worse than its own base**, +6,470 failing endpoints.

## Why the change is very likely NOT the cause

- **The instrument's own nets are a PASSENGER.** Worst slack through the fix's paths is **−9.608**,
  against a design WNS of −13.516 — nearly 4 ns *better* than the critical path. By the forensics'
  own criterion the added logic sits on already-failing paths rather than on the critical one.
- **103,193 of the 103,197 failing endpoints launch from ONE register bit**,
  `dom_switcher/cur_idx_q_reg[5]` (bit [4] contributes the other 4). That is the long-known
  dom-switcher cone, root-caused earlier as `cur_idx` combinationally muxed onto the async regfile
  address port. It has nothing to do with a debug-mux clear in the load unit.
- **There is precedent for multi-nanosecond swings from small edits on this design.** The
  `store_buffer.sv` S-10b comment records *"a 62-line leaf-module change was followed by a 5.8 ns
  WNS regression that has not yet been attributed."* A 32-line additive change moving 2.887 ns is
  the same phenomenon at smaller scale, in a cone dominated by a single bit whose placement is
  evidently sensitive.

**Stated honestly: this is ONE run, and place-and-route is stochastic, so a single sample cannot
separate 2.887 ns from run-to-run variation.** The attribution above is evidence, not proof. A
repeat run would cost ~2 h to answer a question about noise, and the passenger result already
addresses the question that matters — whether the instrument is on the critical path. It is not.

## Is it flashable?

**Not by the project's own rule** — `run.tcl:93-99`, a timing-failing bitstream behaves
intermittently and data-dependently, which is indistinguishable from the silicon defects under
investigation. That rule applies here.

But it applies to **every** bitstream in this table, including the one flown for most of this
investigation. **On WNS, `84ed6eafb` at −13.516 is BETTER than the currently-resident
`caplifive_s10fix_80843404c.bit` at −16.400**, and it additionally passes the lint gate that the
resident image fails. So flying it would not be a regression against what is on the board today;
it would be an improvement on two axes at once. That is a judgement for the project lead, not a
lane, and it does not make the image *good* — only better than the incumbent.

## Two operational findings from the run, neither about this change

**1. The collector is now the memory-critical phase and is close to the ceiling.** It peaked at
**37.18 GB against the guard's 40 GB ceiling** — 2.82 GB of headroom, and materially above the
~32 GB previously observed. Synthesis itself peaked at 21.00 GB. On a busier machine the collector
could trip the guard and **destroy the artifact it is in the middle of collecting**. The ceiling
was sized against synthesis, and synthesis is no longer the peak.

**2. A watchdog reported "bitstream ABSENT" against a path this flow never writes.** It checked
`ariane.runs/impl_1/runme.log`; this flow writes under `work-fpga` and `ariane.runs` does not exist
at all. It was caught before being reported because the **negative control also printed nothing**,
which distinguished "the file is missing" from "the string is missing". Same failure class as the
`pgrep -f` matching its own shell and the `eth_rxck` first-line read: *the query returned
something, so it answered the question.* The defence that worked was, again, a second reading that
had to agree.

## Not missing: timing-forensics section 7

The artifact has sections 1-6 only. Section 7 is 5 commits deep in the S-10 lineage and 0 commits
in `84ed6eafb`'s, so the chosen base predates it. Nothing this build could have produced is absent.
