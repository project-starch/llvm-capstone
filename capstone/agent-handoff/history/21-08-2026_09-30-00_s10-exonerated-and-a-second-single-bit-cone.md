# S-10 is exonerated on timing, the narrow constraint is sound, and there is a SECOND single-bit cone

**Date:** 2026-08-21
**Build:** `76b7f2afc` = `80843404c` RTL byte-identical + the destination-scoped multicycle only.
exit 0, 1h38m57s, peak 21.25 GB, `write_bitstream completed successfully`, bitstream 11,443,722 B.

```
WNS               -12.084     (S-10 unconstrained: -16.400   control: -10.629)
failing endpoints  93,241     (S-10 unconstrained: 102,774   control:  96,727)
```

## 1. The S-10 fix is NOT the timing problem — measured directly for the first time

Section 5 of the forensics only ever asked this for S-07, so S-10 had never been asked. Asking it:

```
gran_oh     denominator      8 nets    worst slack through:  -1.996
gran_clr    denominator      0 nets    (see caveat)
dcache_mem  denominator  27349 nets    worst slack through: -12.084
```

**The S-10 fix's own comparator nets are at -1.996 against a WNS of -12.084 — about 10 ns clear.**
That answers the question that was put to me directly: *if the fix is not on the critical path,
why did WNS degrade after it?* The fix's own logic is nearly meeting timing. The degradation is
placement of cones it does not belong to.

`i_wt_dcache_mem` shows -12.084, i.e. the critical path passes through that module — but the
module is 27,349 nets of pre-existing cache logic, so that is a statement about the module, not
about the 62 lines S-10 added.

**Caveat, flagged rather than read as a result:** `gran_clr` matched **zero** nets. The signal was
optimised away or renamed, so that probe was UNANSWERABLE, not clean.

## 2. The narrow constraint is SOUND — unlike the broad one

The check its own commit mandated, finally run:

```
7a  filters, denominator first:  from_cur_idx 7 cells | to_iro 3637 | to_scoreboard 7083 | to_tracer 49100
7b  must-still-fail groups:      dom_switcher 305 | mechanism_fsm 35 | debug_log 2 | lsu_domswitch 0
```

Nothing is inert, and the switcher's own registers are **still failing** — 305 endpoints under
`dom_switcher/`, 35 on the unconditionally-clocked `_thread_0_event_*_q_reg` FSM registers, 2 on
the debug log registers. Contrast the broad version, which took WNS to -0.046 and made
`dom_switcher` vanish from the report entirely.

**The one zero is MY OWN MATCHER'S FAULT, and 7b is where I violated my own rule.** Every other
block in that script prints its denominator first; 7b printed counts without one. The denominator,
obtained afterwards: the string `sel_dom_switch` appears **0 times anywhere** in the forensics,
while `i_store_unit` appears 6 times, `i_load_unit` 12, `lsu_bypass_i` 35 and `lsu_i` 83 — and
`dom_switch_data_resp_q0_data_q_reg` appears 3 times. So LSU dom-switch registers ARE failing;
the pattern simply does not match any synthesised cell name. **A naming miss, not a masking
finding.** 7b must be fixed to print denominators before its next use.

## 3. NEW: a second single-bit startpoint, and it is not the switcher

```
7c  failing paths NOT launched from cur_idx_q_reg: 79,167 over THREE distinct startpoints
       78,790   ex_stage_i/lsu_i/lsu_bypass_i/status_cnt_q_reg[0]
          336   dom_switcher/_init_0_reg
           41   i_ddr/.../cmd_pipe_plus.wr_data_addr_reg[2]
```

**One bit of the LSU bypass status counter is the startpoint of 78,790 failing paths** — 99.5% of
everything outside the switcher cone. Structurally the same pathology as `cur_idx_q_reg`, in a
different module, and it was completely invisible before: `-nworst 1` reports one worst path per
endpoint, so while the switcher cone was worse, these paths never surfaced.

So the design has **two** single-bit cones that between them account for essentially all timing
failure, and neither is S-10.

## 4. The channel question is settled: DATA, not REG

```
data_read_q_paths.rpt   _reg_ch_  0 occurrences
                        _data_ch_ 60 occurrences        (control: 20 Slack lines present)
worst path  Source: dom_switcher/cur_idx_q_reg[4]/C
            Destination: dom_switcher/data_read_q_reg[36]/CE
            47.398 ns, logic 6.978 (14.7%), route 40.420 (85.3%)
```

Two things confirmed. The destination is the **CE**, exactly as traced from the RTL — the path
runs through the clock enable, not the D input. And the **data** channel dominates, with the reg
channel absent, which retires the "register the reg-channel round trip" framing for good; the
correction made earlier on grep evidence is now confirmed at the design level.

85% route delay on a 47 ns path in a design at 84% occupancy is a placement and congestion
problem, not a logic-depth problem.

## What this changes

* **S-10 is exonerated as a timing regression cause.** Ship-blocking objection withdrawn. Note
  the design failed timing badly *before* S-10 too — the control is -10.629 — so the question was
  never "does S-10 make it fail" but "does S-10 make it materially worse", and its own nets say no.
* **The remedy is RTL on two cones**, not a constraint and not a rework of S-10. The switcher's
  fix already exists unmerged on `capt-implementation` (`42ff49cf6`, `c09469628`). Nothing
  equivalent is known for `lsu_bypass_i/status_cnt_q`.
* **The reflash question is now only about acceptance staging**, which is the board lane's list:
  rebuild the acceptance arms and count `wr8`'s carve cost against the 1021-entry pool.
