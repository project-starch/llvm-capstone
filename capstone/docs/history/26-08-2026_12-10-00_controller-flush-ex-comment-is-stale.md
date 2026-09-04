# `controller.sv`'s domain-switch deadlock warning describes a hazard that is closed elsewhere

**Date:** 2026-08-26
**Verdict:** NOT a live defect. **Documentation defect only** — but a load-bearing one, because
the comment tells a future reader that the line beneath it is dangerous, and it is not.

## What it looks like

`core/controller.sv`, in the `else if (CVA6Cfg.RVA && flush_commit_i)` branch:

```
      // Do not flush EX during a domain switch: the dom_switch_busy signal is
      // a level (not a pulse), so flush_commit_i stays high for the entire
      // domain switch operation.  Asserting flush_ex_o here would continuously
      // kill the d-cache transaction that the domain switcher is waiting for,
      // causing a deadlock where req_en_q never clears.
      flush_ex_o             = 1'b1;
```

The line does exactly what the comment says causes a deadlock.

## It is not a stale comment left over from a rewrite — it is a dated reversal

    901c45ce1  "Applied fixes to hanging problems in domain switching"
               introduced  flush_ex_o = ~dom_switch_busy_i;

    030378a66  2026-04-21, "Fixed an issue that caused the instruction after CALL to be
               executed"
               changed it to  flush_ex_o = 1'b1;  and left the comment untouched.
               Its ONLY other change is 3 lines in core/commit_stage.sv.

So a fix for a domain-switch hang was undone by a fix for a CALL-sequencing issue, and the
warning was left standing over the reversal.

## Why it is nevertheless not reachable

The comment's premise is correct. `core/commit_stage.sv:494` forces the overlap
unconditionally:

    flush_commit_o = flush_commit | dom_switch_busy_i;

so for every cycle of a domain switch, `flush_commit_i` is high and `flush_ex_o` is asserted.
And the switcher does block on a d-cache transaction: `capstone_dom_switcher.anvil:26-30` does
`send data_ch.req >> let data_resp = recv data_ch.resp`, with no timeout and no retry.
`req_en_q` in the comment is `dom_switch_busy_i` itself
(`capstone_dom_switcher.anvil:131` drives `busy_ch.busy (*req_en)`).

What closes it is a SECOND, independent exemption in the LSU, which `030378a66` never touched
and which is present unchanged today:

    core/load_unit.sv:279-284    the flush loop raises ldbuf_flushed only for
                                 !ldbuf_q[i].is_dom_switch
    core/load_unit.sv:697-701    the response path passes a dom_switch entry through
                                 REGARDLESS of kill_req, with its own comment: "dom_switch
                                 entries must always pass through regardless of kill_req,
                                 since the load unit cannot re-issue them and dropping the
                                 response causes a permanent deadlock"
    core/load_unit.sv:667-673    WAIT_FLUSH is entered only when !sel_dom_switch
    core/store_unit.sv:290-291,359 and core/store_buffer.sv:232-238  the same shape on the
                                 store side; a dom-switch store bypasses the speculative
                                 queue that the flush clears

So two different mechanisms were written to close one hazard, in different files, and removing
one left no visible symptom. That is exactly why the reversal survived: it was safe, and
nothing said so.

## The real cost, and it is not zero

The comment now actively misinforms. Anyone touching this branch reads a specific deadlock
warning attached to a line that does the forbidden thing, and has three options — "re-apply the
guard" (a regression, it would re-break the CALL fix), "the comment is nonsense" (dismisses a
correct hazard analysis), or spend an hour rediscovering the LSU exemption. This note exists so
the third is unnecessary.

**Recommended edit, NOT applied here** because the branch tip is currently in synthesis and
changing it would invalidate a verified hash: rewrite the comment to say the hazard is real but
closed in the LSU, naming `load_unit.sv:697-701`, so the two mechanisms point at each other.

## Provenance note

An RTL audit attributed the LSU exemption to `901c45ce1`. That is wrong; `git log -S"is_dom_switch"`
gives `7f8e7f956`, `d2395f803`, `d3e61b761`. It does not change the verdict — what matters is
that the exemption is present in the current tree, which was checked directly rather than
inferred from history.

## What is NOT claimed

Nothing about the unexplained hangs on record — the `create_domain` hang and the R-16 entry
stall. Those were observed on RTL revisions not established here, and whether the LSU exemption
was present at the time was not checked. UNRESOLVED, and deliberately not folded into this note.

## Found how

Incidentally, while auditing an unrelated S-12 recorder change. Recorded separately per the
one-finding-per-commit rule.
