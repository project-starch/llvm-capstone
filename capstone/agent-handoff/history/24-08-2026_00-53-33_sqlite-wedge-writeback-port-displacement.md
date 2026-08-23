# The SQLite spill/reload wedge is NOT writeback-port displacement — excluded on a fired control

> **VERDICT, 2026-08-24, added after the body below was written and pushed. Displacement is
> EXCLUDED.** The board lane already had the reading and did not need a boot: switch 204 reads
> `0x00` at the wedge on all six recorded boots (21, 22, 24, 25, 26, 29), with the switch-220
> selftest firing in every one of them — `post-204 = 0x41`, `ldc_seen` set and the count moved
> by exactly one. So the zero is a **controlled negative**, not an unfired instrument. The
> localization argued below is wrong; the mechanism it describes did not occur.
>
> **Narrower than first stated: the exclusion rests on the HALTED reads only.** The aperture is
> read on two paths and they disagree — halted `0x00` at the wedge, running `0x0c` which the
> driver self-flags VOID. The selftest reads are halted too. The running path is not merely
> weaker evidence, it is *invalid*: `cva6.sv:1026` computes
> `s07_gran_match = s07_ldc0_valid && s07_stc_valid && <granule compare>`, a combinational AND
> requiring both valid bits, so a byte with `gran_match` set and `ldc0_valid` clear is
> **structurally impossible** for the logic to produce. That sample is a readout artifact, and
> it also refutes the "running reads only OR in extra bits" model, so subset reasoning must not
> be used to promote a running read. Halted reads are the only ones that carry a verdict.
>
> **Coverage was then checked for holes and there are none.** `include/ariane_pkg.sv:237-243`
> gives `FLU_WB=0, STORE_WB=1, LOAD_WB=2, FPU_WB=3, CAP_WB=4`. Ports 0 and 4 carry capability
> data and cannot erase. **Port 3 is the FPU port**, driven by a single direct assignment from
> the FPU (`cva6.sv:1495-1498`) with no mux and no LSU or DYN path into it, so an LDC cannot
> land there. Ports 1 and 2 are watched, each for its matching op, and are driven solely by the
> store and load units respectively. The exclusion is complete, not partial.
>
> **Also corrected:** the body says the board carries `caplifive_s07fix.bit`. It carries
> `caplifive_s10fix_80843404c.bit`, and the detector is present and demonstrably alive in that
> image — which the selftest proves rather than asserts.
>
> **Where it goes next.** With the pipeline excluded, the NOT_CAP came from memory or tag state.
> The instrument for that is also already synthesized and UART-safe: **switch 208**, the
> tag-history verdict byte — `[7] ldc0_valid`, `[6:5] ldc0_src` (0 = L1 hit, 1 = miss refill,
> 2 = wbuffer forward), `[4] stc_valid`, `[3] stc_ctag`, `[2] gran_match`, `[1] stc_clobbered`.
> `match=1 & stc_ctag=1` means the tag was written and read back zero and `ldc0_src` says by
> which path; `match=1 & stc_ctag=0` means it was stored untagged and the fault is on the SPILL
> side, which would move the investigation. **The 220 selftest does not control this byte** —
> it drives the displacement counter only. But bit 7 is its own liveness check: at a wedge an
> untagged LDC is known to have occurred, so `ldc0_valid` clear means the recorder never fired
> and the byte is void.
>
> **But bit 7 guards only one direction, and the other is a live false-positive risk.**
> `load_unit.sv:66,766` records the **FIRST** LDC whose response tag was 0 —
> `if (ldc_result_back && !req_port_i.data_rtag && !s07_ldc0_valid_q)` — one-shot, and there is
> **no clear**: the only reset is `rst_ni`. (`dom_switch_log_clear` exists for the domain-switch
> log at bank `3'b101` reg 31; there is no s07 equivalent.) An untagged LDC is not inherently a
> fault — a load over a zeroed stack slot returns tag 0 legitimately — so any earlier arm, or
> the monitor, can permanently claim the slot. Bit 7 then reads 1, the liveness gate passes, and
> every other bit describes an unrelated earlier load.
>
> **The cross-check that closes it is already on the mux:** `s07_ldc0_paddr` at reg 13/14
> (switch 205/206) and `s07_stc_paddr` at reg 15/17 (switch 207/209). Not UART-safe, but the
> documented use is post-run and a halted read after a wedge is exactly that. Requiring the
> recorded granule to match the subject slot upgrades bit 7 from "an untagged LDC was recorded"
> to "the subject's was", and reading the STC address validates `gran_match` directly instead of
> trusting it.
>
> **The body below is kept as written**, because the reasoning was sound and the exclusion is
> what makes it worth keeping: the signature it matches — correct cursor, NOT_CAP metadata, next
> capability consumer raising 25 — is genuinely the documented S-07 signature, which is exactly
> why an argument could not retire it and a controlled negative could.

Written 2026-08-24, connecting the board lane's measurements to a mechanism already named in
this tree. **This is a localization on quoted RTL, not a confirmed root cause** — the deciding
measurement is described at the end and has not been taken.

## What the board lane established

A capability spilled by `stc` and reloaded by `ldc` comes back `NOT_CAP`, and the next
capability consumer — a `cincoffsetimm` — raises mcause 25. Four boots direct, `tval != mepc`
excluding the commit-stage producer, instruction confirmed by disassembly at `mepc`.

Then two results that between them break the drain-latency reading:

| variant | pad | faulting insn | delay | outcome |
|---|---|---|---|---|
| un-probed | none | `0x104A54` | 0 | WEDGE |
| pad10 | 10 nops | `0x104a9c` | ~10 | WEDGE |
| loop3 | 3-iter loop | `0x104a94` | ~10 | completes |
| loop200 | 200-iter loop | `0x104a94` | ~600 | completes |
| pad600 | 600 nops | `0x1053d4` | ~600 | completes |

`loop3` and `pad10` have the **same dynamic delay and opposite outcomes**, and inside the loop
instrument — where the faulting instruction is pinned at `0x104a94` for every n — a 66x delay
sweep changes nothing. Delay cannot carry this.

## The mechanism the tree already names

`core/scoreboard.sv:328-330` ties `cap_data` to `'0` on writeback ports 1, 2 and 3; only port 0
and port 4 (`CAP_WB`) carry capability data. The detector's condition in the same file:

```systemverilog
cap_wb_displaced_o[0] = wt_valid_i[2] && mem_q[trans_id_i[2]].issued && op == LDC
cap_wb_displaced_o[1] = wt_valid_i[1] && mem_q[trans_id_i[1]].issued && op == STC
```

with the in-source comment: *"the dyn unit's load/store syncer displaced a capability op's
response onto the scalar bypass: the instruction retires with a correct cursor and NOT_CAP
metadata, and the next capability consumer raises mcause 25 — the S-07 silicon signature."*

**A correct cursor with NOT_CAP metadata, and the next capability consumer raising 25**, is the
observed signature exactly.

## Why that reconciles every arm

The variable is **writeback-port contention**, and delay, branch and address are proxies:

- **A taken backward branch** redirects the frontend and drains wrongly-fetched instructions,
  keeping the issue/writeback window sparse. Less contention, so the `ldc` lands on port 4 rather
  than port 2. This accounts for `loop3` completing where `pad10` wedges at the same delay,
  with no appeal to alignment.
- **The faulting instruction's address** changes fetch grouping, which changes what is
  co-resident in the writeback window, which changes contention. Real and reproducible, but a
  proxy — a 4-byte alignment sweep would bisect a correlation rather than the mechanism.
- **Delay** is a passenger, which is why the sweep inside one instrument is flat.

It also explains why simulation found nothing: the sim tests measured memory-side behaviour
(drain latency, buffer occupancy) and this is a pipeline-routing effect that never touches the
write buffer at all.

## The deciding measurement, which has NOT been taken

A synthesized detector for this is already on the debug mux — no new arm, no new binary:

| what | where | notes |
|---|---|---|
| displacement byte | bank `3'b110`, reg `5'b01100` -> **switch 204** | `204 & 3 == 0`, UART-safe, samplable mid-run |
| contents | `{STC-via-STORE_WB seen, LDC-via-LOAD_WB seen, 6-bit saturating LDC count}` | non-zero = pipeline erased it, memory was fine |
| **selftest** | bank `3'b110`, reg `5'b11100` -> **switch 220** | `cva6.sv:1025`, one-shot per reset, drives the SAME sticky and counter through the REAL path |

**Fire the selftest first and require the count to move.** The board currently carries
`caplifive_s07fix.bit` and it has not been verified that this image contains the detector. If it
does not, switch 204 reads zero, and that zero is indistinguishable from "no displacement
occurred" — a false negative of the exact class this project keeps paying for. A reading of 204
is void unless 220 has fired in the same boot.

Take it on any variant that already wedges. Non-zero LDC count confirms the localization and
collapses the delay, alignment and contention lines into one mechanism; zero with a fired
selftest excludes displacement and makes the alignment sweep worth its arms.

**Not claimed:** that displacement is confirmed, that it is the only mechanism, or that the
detector has been watched firing. The condition and mux placement are read from source; nothing
here rests on having seen it pulse.
