# The per-boot run ceiling was the region table, and it is fixed (needs a push)

Date: 2026-08-18. Bitstream `caplifive_s06s08fix_s07tag2_618f4ce.bit` (unchanged — no RTL involved).

## Result

**12 of 12 SQLite-class domains in one boot, all PASS, `rgid` reaching 56, zero region overflows.**
Before the change the same boot stops at domain 5 or 6.

Board sessions were capped at ~5 useful runs. They are now capped at ~21. Since the cost of `n` is
what made measuring a RATE unaffordable — and unaffordable rates are why single-sample wedges kept
being written up as findings — this is the enabling fix for `ref/RATE-RULE.md`.

## What it was

`region_n` reaching `CAPSTONE_MAX_REGION_N = 32`. Proven by the monitor's own tags on the domain
that failed to enter: `RGNN:00000020` (= 32) and `RGNO:0000E00C` (`ERR_REGION_OVERFLOW`). The
guard is a `while(1)` in M-mode, so the symptom is the *next* domain never entering — which reads
exactly like an entry stall and was recorded for weeks as an unknown exhausted resource.

It had already been diagnosed in a comment beside that guard ("region ids reach 24/25 by the
FOURTH SQLite domain against CAPSTONE_MAX_REGION_N = 32") and never connected to the ceiling.

## Two wrong diagnoses on the way — both worth not repeating

1. **"The ceiling does not exist"**, from 10 identical rungs passing in one boot. Rungs consume
   ~1 region id each and could never reach 32. A negative that could not fire.
2. **"It is rev-node pool exhaustion"**, from reading `rev_node_head = 1005` against a 1024 pool.
   The debug mux exposes only `head[9:0]` while the RTL sentinel is 65535 — a **16-bit** head — so
   1005 is a truncated view, and the `overflow` bit reads 0. Caught before acting on it.

## The size is capped by the compiler, not by memory

256 does not build:

    sbi_capstone_dom.c.S:6049: Error: illegal operands `addi t1,t1,-4096'

capstone-c emits a plain `addi` for the offset into `regions[]`, and 256 x 16 B = 4096 is outside
the 12-bit signed immediate. **96** keeps the largest offset at 1536 B. Safe to raise at all
because it bounds the SOFTWARE tables only: the hardware resident set is `CPMP_COUNT = 16` and
regions load into CPMP lazily.

## Build gotcha that cost a boot

The `.c.S` generation rule's input is `sbi_capstone.c`, so **editing the header regenerates
nothing** and the firmware relinks stale — the first attempt booted an unchanged monitor while
reporting success. `touch` the `.c` after any header change. Also: the successful build's log still
contains the failure from its own earlier pass, so exit code 0 alone is not evidence; check the
generated asm.

## NOT PUSHED — needs the project lead

The monitor lives four submodules deep
(`caplifive-system` -> `sw/buildroot` -> `components/opensbi` -> `lib/sbi/capstone-sbi`) and this
credential has **no write access**:

    remote: Permission to project-starch/capstone-sbi.git denied
    fatal: ... The requested URL returned error: 403

Both commits are local on branch `capstone-bootstrap` in that innermost repo:

| commit | what |
|---|---|
| `6423d23` | raise `CAPSTONE_MAX_REGION_N` 32 -> 96 |
| `0bd1fec` | the `DBAS`/`DENT`/`MSTA` trace tags, **found uncommitted in the working tree** |

The second one matters independently of the ceiling: `DBAS` is the domain's load base, and every
board verdict this week — including the analysis that refuted physical placement as the S-07
modulator — was built on it. It was sitting in a working tree only, which this project has already
paid for once.

## Rate so far, since it is now cheap to extend

`XU` at hash `f1214600d0dac351`, bitstream as above: **k=2 wedges in n=14** reps across five boots,
including one boot that wedged on its first rep an hour after four consecutive clean runs.
