# New bitstream: fault delivery does NOT fire for S-07, and 204/208 are VOID on it

Date: 2026-08-18. Bitstream **`caplifive_s07debug_18august.bit`** = `capstone-ariane` `6882b265f`.
Every silicon number taken before this flash is baseline-invalid.

## What this bitstream actually is

**The gen-3 probe was WITHDRAWN — it never synthesized.** It drove `synth_design` past 100 GB, the
same phase and signature as the earlier 343 GB failure. So there is **no census, no LDC/STC PCs,
no correlation bit, no dwell counter and no synchronizer on the trigger**.

The entire RTL change is one line, removing the first-capture guard so the tag-history record
**ROLLS** instead of latching once:

```
-  if (ldc_result_back && !req_port_i.data_rtag && !s07_ldc0_valid_q) begin
+  if (ldc_result_back && !req_port_i.data_rtag) begin
```

That is worth the flash on its own: the one-shot record was spent by boot-time software before any
test domain ran, so it produced nothing usable across two sessions. Rolling cannot be spent, and a
wedge freezes it.

Apertures revert to the gen-1 map (193/194 gen-1, `mepc` back to eight bytes at 196-203, 219/221/
222/223 absent). The driver detected this **by itself**: the generation check is keyed on 193,
whose bit 7 is hard zero, so it read gen-1 and assembled `mepc` from eight bytes correctly. Keyed
on 216 it would have mis-declared the generation.

**With the dwell counter gone there is no debounce and no synchronizer on the 220 selftest
trigger.** The driver's transit-avoiding switch walk is now the only protection against an
accidental injection.

## 1. The merged monitor's fault delivery does NOT fire for S-07

The peer monitor branch was merged locally, keeping both sides (their `fault_return_from_domain`
plus our UART diagnostics). The hope was that an S-07 fault would become a **returned fault code**
rather than a dead board — which would have removed the defining obstacle of this investigation,
since a wedge destroys its own evidence.

**It does not.** Verified the merged monitor is genuinely in the running firmware —
`fault_return_from_domain` appears 5x in the linked ELF and 10x in the generated assembly — and
the domain still wedges: `SQ: G/enter`, no return. The path is present and not reached for this
cause.

Clean negative, and good for continuity: the wedge classification, `s07-rate.py` and all
accumulated k/n stay valid and need no restatement.

## 2. 204 and 208 decode to IMPOSSIBLE values — every verdict from them is VOID here

| aperture | reading | decode |
|---|---|---|
| 208 after `S7T` | `0x9c` | `ldc0_valid=1 src=0` — "(b) GENUINE TAG LOSS" |
| 208 after `XU` | `0xfe` | `src=3` — **UNDEFINED, illegal** |
| 208 after `XU` | `0x0c` | `ldc0_valid=0` with other fields set |
| 204 at the wedge | `0x0c` | `count=12` with `ldc_seen` **CLEAR** — **illegal by construction** |

The 204 reading is decisive: a non-zero count with the seen bit clear is precisely the combination
the closed encoding was designed to make impossible. So the decoder is reading bits that do not
mean what it thinks on `6882b265f`.

**This is the closed encoding earning its keep.** The first line decodes cleanly and says exactly
what everyone wants to hear — a genuine tag loss, the answer to the whole investigation. Without
the self-check it would have been reported as a breakthrough. Asked the RTL lane for the
field-by-field layout of 204/208 as synthesized, including whether removing the first-capture
guard changed the meaning of any field that was previously written only on the first capture.

Until that is answered: **no S-07 verdict from 204 or 208 on this bitstream.** The pass/wedge rate
is unaffected — it comes from domain markers, not the mux — so the investigation is not blocked.

## Fresh baseline

`XU` (`f1214600d0dac351`) on this bitstream: **k = 1 wedge in n = 3**. Accumulating.

## Lesson worth keeping

The aperture map handed over was accurate about **which** apertures exist, and the generation check
built on it worked. The gap was the **bit layout within** an aperture, which was described as
"unchanged" and was not. A hash handed between lanes carries its lint numbers by existing rule;
this says it should also carry the field layout of any aperture whose semantics could move, even
when the aperture number does not.
