# First boot on the S-10 bitstream produced no domain output — PROVISIONAL, N=1

**Date:** 2026-08-21
**Status: NOT A FINDING.** One boot, one arm, on a shared console. Recorded because it is the
first silicon evidence about `caplifive_s10fix_80843404c.bit` and because the analysis below is
reusable, **not** because anything is established.

## What was observed (reported by the board lane, not by me)

Board powered on, SBI banner, Linux 6.4.14, shell prompt, device check `DEVOK`/`DN_0`. Driver
dispatched `TEST 1/4 /test-domains/sqbase.dom`. **After that dispatch: zero printable UART**, then
binary garbage. The run was stopped on a verified pid; `release_board` ran cleanly.

`sqbase.dom` is byte-identical to the plain-SQLite build that previously passed 3/3, run under its
matching 4 KiB-region host. So the domain image is not the variable; the firmware is.

## Why this may be the S-08 class, and why that is more than a guess

The board lane's hypothesis, reached from the symptom alone: `Ok, good file` (libcapstone's first
line, via `write(2)`) and the shell's command echo are **both syscall-mediated**, so if U-mode
ecalls stop being delegated they go silent together, and the failure presents as "nothing at all"
rather than "markers up to the share" the way S-08 did.

That is not a story fitted to the evidence. It is **S-08's actual mechanism**, from its own fix
commit (`9fd5507be`):

> *"userspace ecall no longer delegated; every syscall dies in M-mode's unhandled handler"*
> *"medeleg=0 kills ecall delegation — exactly the measured S-08 signature"*

**But the S-08 fix IS in this bitstream** — `9fd5507be` is an ancestor of `80843404c`, as are S-06
(`25035c4c0`) and S-07 (`5c5f4e3a7`). So if this is that signature it is either a regression of
that fix or a **second path to the same architectural state**. Two hypotheses, not one.

## S-10 checked as a cause: unlikely on source analysis

S-10's only behavioural addition is that a write-buffer granule-mate with `ctag == 0` forces a
read's tag to zero (`wt_dcache_mem.sv:305,319,387`). The worry is that this untags a capability
context row during a domain switch. The layout says it cannot reach one.

On the reachable path — `is_full` is hardwired `1'b0`, so `capstone_dom_switcher.anvil:121-125`
takes `val_n = 7'd7`:

```
rows 0-2   16 bytes, metadata_en=1 (capabilities)  ->  +0, +16, +32
rows 3-7    8 bytes, metadata_en=0 (scalars)       ->  +48, +56, +64, +72, +80
granules:  cap rows own granules 0,1,2 EXCLUSIVELY
           scalars pair up:  g3={+48,+56}  g4={+64,+72}  g5={+80}
```

**A capability row never shares a granule with a scalar row.** The scalars that do pair up —
including mstatus at +48 and mideleg at +56, the S-08 slots — are `ctag=0` already, so forcing
their tag to zero changes nothing.

**Limits, so this is not over-read:** source analysis of the layout, not a proof; it assumes a
granule-aligned base; it says nothing about the LSU-side path or about timing. It lowers the prior
on S-10; it does not eliminate it.

## The larger suspect remains timing

This image misses setup by **5.8 ns more** than the one every prior board result was taken on
(WNS -16.400 against -10.629). `corev_apu/fpga/scripts/run.tcl:93-99` states that a timing-failing
bitstream *"behaves intermittently and data-dependently — the exact signature of the S-07 defect
under investigation, with no way to separate the two afterwards."* That fits "boots, completes a
shell command, then the first domain dispatch produces nothing and the console degrades to
garbage" at least as well as any architectural explanation.

**This is the confound that was accepted when the image was flashed.** For the record: the flash
was not on this lane's recommendation — the recommendation was to hold until the acceptance arms
existed and `wr8`'s carve cost had been counted, neither of which was true at flash time.

## What would settle it, cheapest first

1. **A boot with the console not shared.** `user_count` was 3. Until that is removed, external
   interaction stays in the hypothesis set and it is the cheapest one to eliminate.
2. **Any monitor-side marker in a boot where everything user-side is silent.** The monitor's own
   output does not go through ecall delegation, so its presence separates "syscalls stopped being
   delegated" from "the core is wedged" — with no extra tooling. If even monitor-side output is
   absent, it is **not** the S-08 class and the timing hypothesis moves up.
3. **The trap-register read** — `EXCX:0000E002`, `MCAU:00000008`, `MSTA` with MPP=0, the constants
   recorded in `tests/fpga-repros/S08-s06fullfix-bitstream-cannot-run-domains/`.

## Consequence for S-10 acceptance

**The acceptance arms are built and HELD.** Staging them on an image that may be unable to run
domains would produce a ladder of void arms that looks like an S-10 result — the same failure the
slot-budget correction just avoided. They live in `tests/rtl-smoke/wbuf-arms/` with
`READING-RULE.md`; waiting costs nothing.

---

# RESOLVED, same day: the image runs domains. Boot 1 was infrastructure.

**Second boot, same image, same four arms.** Two changes only: the console was confirmed
**unshared**, and `SQLITE_IDLE_S` was dropped from 1800 to 240.

**Control GREEN, complete marker chain.** `sqbase.dom` — byte-identical to the pre-SLT plain build
— returned `row name=alpha value=11` / `beta=22` / `gamma=33` / `EXTENDED_PASSED` /
`MEMORY_PASSED`, `rc=0`, with the full sequence inbound: `Ok, good file`, `DBAS:`/`DENT:`,
`A/dom-ok`, both region shares with their `LC:`/`ECSA:`/`SHA0-6:` monitor output, `G/enter`,
`H/return`.

So on `caplifive_s10fix_80843404c.bit`: **SQLite runs, domains run, region sharing works, and both
M-mode and U-mode output flow.**

## The discriminator worked, and it ruled out the S-08 class

The cheap discriminator proposed above — *any monitor-side marker while user-side is silent
separates "delegation stopped" from "core wedged"* — was applied by the board lane to the log they
already had. Boot 1 had **zero M-mode output as well as zero U-mode output**, with a positive
control proving the matcher fires (164 monitor-side markers in the replayed prior boot). Both
classes silent is not the S-08 signature, which leaves monitor output intact.

**So boot 1 was neither S-08 nor "cannot run domains"** — the latter now refuted by direct
demonstration.

## What boot 1 WAS, and what it was not

One void boot out of two, with the console shared during the bad one, sits inside the documented
~1-in-5 infrastructure control-failure rate. The board lane declined to attribute it to timing
without a reproduction, which is right: **N=1 with a known infrastructure failure mode available is
not evidence for anything else.**

One correction they also recorded: at `SQLITE_IDLE_S=1800` the driver would have waited thirty
minutes before declaring the domain silent and performing its wedge read. Killing it during that
wait is why no trap registers exist for boot 1. The value is now 240.

## The distinction that must NOT be lost

**The timing hypothesis is retired for BOOT 1. It is not retired.** This image still misses setup
by 5.8 ns more than the one every prior board result came from, and `run.tcl:93-99` still says a
timing-failing bitstream behaves intermittently and data-dependently with no way to separate it
afterwards. That caveat stands for every future anomaly on this image. "Explained once by
infrastructure" is not "explained".

## Also demonstrated on this silicon, and it is the property this project keeps needing

The SQLLogicTest **negative control passed field for field on hardware**: 21 records, all six
deliberately-wrong arms fired for the right reasons — wrong value, wrong md5, right-md5-wrong-count,
too-few-values, statement-ok-that-errors, statement-error-that-succeeds — plus an unparseable record
counted rather than skipped. The MD5 the board computed over a 500-value result set is bit-identical
to the host's.

That is exactly the property built into `wb0`/`wb2`: **an instrument proven able to report failure
before any clean result from it is believed** — and it now has a silicon demonstration.

## Consequence

**The S-10 acceptance arms are UNBLOCKED.** The two-boot plan stands as written — `wb0`/`wb2`
before `wb1` in one boot, then `wb4`/`wb3`/`wr8` — within the four-slot budget. Nothing here
establishes anything about S-10's correctness; it establishes only that a control-validated boot is
achievable on this image.
