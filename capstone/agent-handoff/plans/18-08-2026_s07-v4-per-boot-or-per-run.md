# S-07 v4: decide whether the wedge is PER-BOOT or PER-RUN

## Context

The v3 plan's goal — make measuring a rate affordable — is achieved. The per-boot run ceiling was
`CAPSTONE_MAX_REGION_N = 32`; raised to 96 and demonstrated at **12 domains in one boot, `rgid` 58,
zero region overflows**, against 5 before. A rate now costs a few boots instead of a session.

What is known, all of it negative:

* **Not the image.** `XU` at hash `f1214600d0dac351` has both passed and wedged on the same
  bitstream — 4/4 clean one hour, wedging on rep 1 the next.
* **Not physical placement.** Same domain, same hash, same `DBAS 0x84400000`, same position,
  wedged at 03:42 and passed at 03:50.
* **Not boot position**, and not bulk tag loss (zero in 2.1M reloads, though that sweep is
  underpowered for a rare single bad load).
* **Not the C-19 compiler regression** — that was a separate, deterministic, now-reverted defect.

Current measured rate: **k = 2 wedges in n = 16** reps.

**The one question that has never been asked is whether the wedge is a property of the BOOT or of
the RUN.** Everything above rules out static properties; nobody has tested whether "bad boots"
exist. It is now cheap to find out, and the answer decides where to look next.

## The experiment

Six boots, each `S7T` control followed by **`XU` repeated 12 times**, and record **reps until the
first wedge**. A wedge ends the boot, so that statistic is what the censoring allows — and it is
enough:

| observed shape | reading |
|---|---|
| **bimodal** — some boots wedge in the first reps, others survive all 12 | **per-BOOT state.** Something set at power-up or drifting within a session. Next: diff good vs bad boots using data already logged (`DBAS`, region ids, timing), and vary power-cycle dwell. |
| **geometric** — first-wedge position scattered uniformly, ~1 in 8 per rep | **per-RUN randomness.** No boot-level state; the trigger is a rare event inside the workload. Next: hunt the site, not the environment. |

Six boots at ~14% per rep gives roughly 8-10 wedges, enough to tell those apart. If they are
indistinguishable at n≈70, that is itself worth recording — it bounds any per-boot effect.

**Use ONE domain repeated, never a mixed interleave.** Mixed boots hit the second, still-unfixed
ceiling: `SPLB:0000E010` = `CAPSTONE_ERR_SPLIT_EXACT`, the `split_out_cap` exact-fit spin, which
stopped two boots at domain 6 today. Uniform domains reuse the same carve geometry and avoid it —
`S7T` x 12 ran clean. This also means today's `tsq` interleave yielded 2 usable reps instead of 8.

## Then, and only then

* **If per-boot:** the modulator is environmental and the existing transcripts already contain the
  candidate variables. This is where a power-cycle-dwell or thermal test becomes worth a boot.
* **If per-run:** the `tsq` meter is the wrong instrument (it samples bulk memory, not the rare
  site) and effort goes back to localising the faulting load — with the advantage that reps are
  now cheap enough to bisect the workload by rate rather than by single samples.

## Deliberately NOT doing

* **No merge of the remote monitor branch.** Its `fault_return_from_domain` handles
  `RISCV_EXCP_INVALID_CAP` — S-07's own cause family — and would convert a wedge into a returned
  fault code, orphaning every S-07 classification and the k/n collected so far. Valuable later, as
  its own task with board re-validation; fatal to do mid-measurement.
* **No fix for the SPLB exact-fit ceiling.** Avoidable by using uniform domains; fixing it is
  monitor work that would change the firmware mid-experiment.
* **No mitigation A/B.** The double-load arm is one sample and stays uncited until there is a
  baseline to compare against.
* **No RTL, no bitstream, no reflash.**

## Files

* Run: `tests/rtl-smoke/fpga_driver/run_sqlite_stages_fpga.py` (`SQLITE_STAGE_DOMS` with one
  domain repeated), classify with `tests/rtl-smoke/s07-rate.py`
* Record: a dated note in `agent-handoff/history/`, and the running k/n in
  `agent-handoff/ref/RATE-RULE.md`

## Verification

1. **Control first in every boot**; a boot whose control fails is VOID.
2. **Classify, do not just count.** Only a domain that ENTERED and never returned counts toward k;
   `NO-ENTRY` and `SPLB` stops carry no verdict and are excluded from both k and n. `s07-rate.py`
   does this and is negative-tested.
3. **Report k and n, never "it wedges".**
4. **Verify each domain by content hash inside the cpio** before trusting a boot.
5. `precommit-scan.sh`, and `git commit -o <paths>` so a concurrent lane's staged work cannot ride
   along.
