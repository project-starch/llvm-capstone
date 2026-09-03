# Task: strip the S-07/S-12 debug instrumentation from the RTL, properly

**Status:** NOT STARTED. Deferred deliberately on 2026-09-04 so it could not contaminate the
S-12 fix bitstream cycle. Do this once S-12 is closed.

## Why this is a real task and not tidying

The instrumentation is in **every bitstream the board results rest on**, unconditionally — it is
not behind any `ifdef`. Measured on `80843404c`:

    debug_led_o refs in cva6.sv                        66
    S-07 LDC recorder (s07_ldc0_*)   16 in load_unit.sv, 11 in cva6.sv
    S-07 STC recorder (s07_stc*)                       17
    selftest / tracer / trap log / dom-switch log      8 / 12 / 36 / 3

On a design at 84% LUT occupancy that fails timing on every endpoint, that is not free. What it
costs has never been measured — which is the immediate reason for the measurement arm described
below.

## What was done INSTEAD, and why it is not this task

Arm 2 of the S-12 synthesis (`s12-fix-noinstr`, `6f8345fdb`) adds ONE line —
`debug_led_o = 8'b0` as the last assignment in the debug mux's `always_comb` — so the whole
aperture tree becomes dead logic that synthesis removes.

That was the right call for a bitstream cycle whose job is measuring one variable, and it is NOT
a substitute for this task:

* it removes the **logic**, not the **source**. The code stays, and every future reader still has
  to understand it.
* it kills the trap-summary apertures too, so that build cannot read `mcause`/`mepc` at a wedge.
  It prices the instrumentation; it cannot replace the debug vehicle.
* the tracer survives it, because that drives `uart_debug_tx_o` independently of the LED mux.

## Why the obvious routes do not work — measured, so nobody re-derives them

**Branch from before the instrumentation: IMPOSSIBLE.** The debug LED/switch infrastructure dates
from **2025-07-22** (`bcbbbc236`), 177 commits before `80843404c`, and the S-07 recorders from
**2026-08-17**. The S-10 fix is **2026-08-20**. So the instrumentation is *underneath* every fix
we need; there is no commit with the fixes and without it.

**Revert the instrumentation commits: 7 of 8 CONFLICT.** Tested, not assumed:

    b30b93fab CONFLICT   a2ef8ebae CONFLICT   9490baf0b CONFLICT   d65c67589 CONFLICT
    8c75d899b CONFLICT   ecfda99b0 CONFLICT   d2b7c14f0 CLEAN      1fc34e158 CONFLICT

Because they are ~11 interleaved commits over five days on the same two files, several of which
are corrections or partial strips of each other (`8c75d899b` withdraws the gen-3 probe,
`a3dbae618` removes two dead instruments, `83a7d061f` strips a never-synthesised one). Two also
carry NON-instrumentation content: `a3dbae618` turns retiming back on and ungags three warning
classes; `39111e119` puts `tval` in the trap latch, which is what makes a wedge's faulting operand
readable at all.

`83a7d061f` looks like precedent but is not: it stripped an instrument that had **never been
synthesised**, so nothing depended on it.

## What the task actually is

A hand-written removal, on its own branch, validated on its own terms:

1. **Decide what STAYS.** Not everything here is debug scaffolding. The trap latch
   (`recent_nontrivial_mepc/mcause/tval`) is how a wedge names its own faulting pc, and a domain
   fault does not reach the monitor — `cva6.sv` says the debug mux is "the ONLY way to recover the
   faulting pc". Removing that would make every future wedge unclassifiable. Candidates to KEEP:
   the trap latch and the minimum aperture to read it. Candidates to GO: the S-07 LDC/STC
   recorders, `s07_gran_match`, the selftest, the granule apertures, the census bytes.
2. **Remove by hand, one concern per commit**, so a bisect can attribute a regression.
3. **Validate as a change in its own right** — lint signal-set diff against base, the full
   in-regime suite, the delay sweep. It is a large diff touching files in the standing
   combinational-loop cone; the fact that it only *deletes* is not a safety argument.
4. **Rebaseline `verif/sim/rtl-lint.REF.txt` deliberately**, with the new counts justified in the
   commit message. Note the gate's stored `UNOPTFLAT 39` is ALREADY stale against this lineage's
   genuine 40 — see below — so that number must be re-derived, not carried.
5. **Synthesise and census it.** Removing logic from a design whose usability rests on every
   failing endpoint originating in the dom-switcher cone is not automatically safe: placement
   moves, and a census that gains an originating register outside that cone means the build is not
   usable regardless of how much smaller it is. See
   `ref/bitstream-usability-is-the-census-not-the-slack.md`.

## Sequencing

**After S-12 is closed.** Doing it now would put a large deletion into a cycle whose entire
purpose is measuring one four-line change, and forfeit the attribution that makes the result
readable.

The arm-2 measurement should be read first: if the instrumentation turns out to cost little in
LUTs and nothing in the census, this task is housekeeping and can wait. If it costs materially,
it is on the critical path for any future bitstream on a design with under 1% margin.

## Related

- `ref/bitstream-usability-is-the-census-not-the-slack.md` — the acceptance criterion any stripped
  build must still meet.
- The lint gate's stored baseline is stale independently of this work: it holds `UNOPTFLAT 39`
  from a lineage without the S-10 fix, while `80843404c` genuinely has 40. It fails identically on
  the unmodified base. Do not rebaseline it to make a gate pass; re-derive it when this task
  changes the counts for a real reason.
