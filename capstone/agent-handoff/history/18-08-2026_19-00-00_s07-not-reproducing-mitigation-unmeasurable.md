# S-07 is NOT reproducing today, so the mitigation A/B measured nothing

Date: 2026-08-18. Bitstream `caplifive_s06s08fix_s07tag2_618f4ce.bit`. Compiler: `capstone-bootstrap`
with `CapstoneCapGlobalInit.cpp` reverted (C-19), which restores `XU` to the historical
`f1214600d0dac351`.

## The numbers

Two boots, same session, same bitstream, control passing in both.

| arm | what it is | reps | outcome |
|---|---|---|---|
| `S7T` | control | 1 per boot | PASS both boots (`obs=1460078339`) |
| `XU` | unmitigated, **byte-identical to the historical reproducer** | 4 | **4 PASS, 0 wedges** |
| `XD` | `-mllvm -capstone-double-ldc` | 2 | 1 PASS, **1 WEDGE** |

Both `XU` and `XD` passes are full completions: `__CAPSTONE_SQLITE_EXTENDED_PASSED__` and
`__CAPSTONE_SQLITE_MEMORY_PASSED__`.

## What this does and does not say

**The mitigation is UNMEASURABLE today, and that is the headline.** The A/B was designed around a
pre-registered rule: a boot where the unmitigated arm does not wedge carries no evidence about the
mitigation. `XU` did not wedge in four consecutive reps, so there was nothing to mitigate. Any
verdict on `-capstone-double-ldc` from these boots would be manufactured.

**The rate is not a property of the binary, and this is now nailed down.** `XU` here is
byte-identical (`f1214600d0dac351`) to the artifact that wedged repeatedly in earlier sessions.
Same bytes, same bitstream, four clean runs. Whatever modulates S-07 is board or environment
state, not the image. The repro folder already said the rate moved 23% -> 0% -> back
unexplained; this is the same phenomenon observed against a hash-verified control.

**Weak evidence AGAINST the double-load, not for it.** `XD` wedged once in two reps while `XU`
passed four. That is N=2 against N=4 and nowhere near significance, but the direction is the wrong
one for a mitigation, and it has a plausible mechanism: the transform adds ~43k extra capability
loads and 192 KiB of text, so it changes both memory traffic and layout. It should not be
described as promising on this evidence.

## Consequence for the plan

The mitigation-first plan assumed a reliably wedging baseline. That assumption is currently false,
so the three-boot A/B cannot proceed. Options, in the order they are worth trying:

1. **Wait for the baseline to wedge again** and re-run the identical pair. Cheap, but the trigger
   is unknown, so this is not schedulable.
2. **Find what modulates the rate.** This is now the most valuable open question and it was never
   really asked: the same bytes wedge on some days and not others. Board temperature, boot
   history, DRAM contents, domain placement (`DBAS` differs per boot) are all candidates, and
   `DBAS` is recorded in every transcript already, so a retrospective look across sessions costs
   nothing.
3. **Structural minimization** (`CAPSTONE_EXT_STOP` ladder, `MRO`/`MRR`) is also blocked for the
   same reason: it bisects by observing which arm wedges, and nothing wedges today.

## Instrument note

The driver mislabelled `XD`'s full pass as "no obs= marker, expected for a truncation arm". `XD`
is not a truncation arm; the full build reports via `__CAPSTONE_SQLITE_*_PASSED__` markers rather
than `obs=`, which the driver does not know about. Read at face value it would have hidden a
genuine full-workload pass. Per-domain UART was read instead.
