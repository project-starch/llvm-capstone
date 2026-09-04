# The S-07 tag-history recorder is unusable in practice: "first since reset" is the wrong trigger

Found 2026-08-24, on the boot that was meant to use it. **This is an instrument defect, not a
limitation of one run** — the aperture cannot answer its question for any workload that runs
after boot software, which is every workload we have.

## What the instrument is

`cva6.sv` switch 208, the S-07 tag-history verdict byte, with the granule addresses alongside it:

| aperture | switch | contents |
|---|---|---|
| verdict byte | 208 (UART-safe) | `[7] ldc0_valid`, `[6:5] ldc0_src` (0 = L1 hit, 1 = miss refill, 2 = wbuffer forward), `[4] stc_valid`, `[3] stc_ctag`, `[2] gran_match`, `[1] stc_clobbered` |
| LDC granule | 205 / 206 | `s07_ldc0_paddr[11:4]` / `[19:12]` — post-run only |
| STC granule | 207 / 209 | `s07_stc_paddr[11:4]` / `[19:12]` — post-run only |

It is meant to separate a tag genuinely lost between store and load (naming the cache path that
served the load) from a granule that was stored untagged in the first place — a fork that decides
whether a fault lives on the reload side or the spill side.

## The defect

`core/load_unit.sv:66` and `:766`:

```systemverilog
// S-07 probe: sticky record of the FIRST LDC whose response tag was 0.
if (ldc_result_back && !req_port_i.data_rtag && !s07_ldc0_valid_q) begin
```

One-shot, first-wins, and **there is no clear** — the only reset is `rst_ni`. Checked
specifically: `dom_switch_log_clear` exists at bank `3'b101` reg 31 for the domain-switch log,
and there is no s07 equivalent anywhere.

**An untagged LDC is not inherently a fault.** A capability load over a zeroed stack slot, or over
plain scalar data, legitimately returns tag 0. Boot software does this as a matter of course. So
the one slot is consumed before any domain starts, and the subject's load can never be recorded.

## Measured, not inferred

Boot 31, halted reads:

    PRE-RUN baseline, before any arm ran:   sw=208 = 0xb8   ldc0_valid ALREADY SET
    after sqpad10 wedged:                   sw=208 = 0xb0   ldc0_valid=1, stc_ctag=0, gran_match=0
      s07 ldc0 granule paddr[19:4] = 0x81170
      s07 stc  granule paddr[19:4] = 0x9f370      <- different from each other, neither is the subject slot

## Why this was nearly expensive

**The liveness gate passes.** `ldc0_valid` set was being used — by both lanes — as the byte's
positive control, on the reasoning that at a wedge an untagged LDC is known to have occurred, so
a clear valid bit means the recorder never fired. That is sound and it guards exactly one
direction. It cannot see **attribution**: bit 7 set proves *some* untagged LDC was recorded, never
that it was the subject's.

**And the branch it decodes into is the worst one available.** `stc_ctag=0` reads as "the granule
was stored untagged, so the fault is on the SPILL side" — which would have moved the whole
investigation onto the wrong half of the problem, on the strength of an unrelated boot-software
load. The gate had been positive-controlled five ways offline and none of the five could see
attribution, because every one of them tested decoding rather than provenance.

**What caught it** was cross-checking the recorded granule addresses against the subject slot,
which is possible only because the addresses are on the mux. The recorder is well-built in that
respect; the trigger is what is wrong.

## The fix is RTL, and it is small

No ordering of arms helps — the monitor runs before every domain regardless. Two candidates:

1. **Add a clear on the debug mux**, mirroring `dom_switch_log_clear` exactly (a proven in-tree
   pattern). Purely additive, changes no existing reading, and lets a driver domain-scope the
   record by clearing immediately before the arm. **Preferred.**
2. **Drop the `&& !s07_ldc0_valid_q` term** so the recorder is last-wins. This is a *strict
   reduction* — less logic, which CLAUDE.md prefers for observation-only changes — and after a
   wedge the last untagged LDC before the core died is the subject's. But it silently changes
   semantics any existing analysis may rely on, and it churns during a long run.

Either is bitstream-gated: useless on the board without synthesis and a reflash, and per
CLAUDE.md a hash is not ready until synthesis has RUN. **Not started; it competes for the same
synthesis slot as the latent dom-switcher defect and that priority is the project lead's call.**

## The generalisable lesson

**A positive control proves a detector can FIRE. It does not prove the firing was YOURS.** For any
sticky, one-shot, or first-wins recorder, liveness and attribution are two different controls, and
the liveness one is the easy one to mistake for both. Where a recorder exposes an address, a tag,
or any identity alongside its verdict, cross-checking that identity against the subject is the
attribution control — and it is the one that catches a false positive, which is the direction a
liveness gate structurally cannot guard.
