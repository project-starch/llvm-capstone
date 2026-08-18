# S-07 v3: make debugging fast enough to measure a rate, then root-cause it

## Context

This replaces the v2 plan, whose premise — a reliably wedging baseline — turned out to be false.

Today, `XU` at hash `f1214600d0dac351` — **byte-identical to the artifact that wedged repeatedly in
earlier sessions** — passed 4 of 4 full extended-workload runs. So the wedge rate is not a property
of the image; something in board or environment state modulates it. Meanwhile the double-load arm
wedged once in two reps, which is a single sample and therefore means nothing.

That is the real blocker, and `SILICON-BLOCKER.md` already diagnosed it and was never acted on:
*"Any single-sample wedge means nothing… The measured unit must be a RATE, with n reported."* Most
recorded wedges in this project are single samples **by construction**, because a wedge ends the
board session. The "unexplained build-to-build sensitivity" running through the whole
investigation is, at least partly, that background rate sampled once per image.

**So the blocker is not S-07. It is that we cannot afford to measure a rate.** One boot costs
~5 minutes and yields at most ~5 domain runs. Everything below is aimed at changing that first,
because every other question — is the mitigation real, does placement matter, did any historical
"X wedges" claim ever exceed background — is unanswerable until n is cheap.

No RTL change is available (the new bitstream is still generating), and none is needed for
Priority 1.

## Priority 1 — make the boot loop fast (this is the whole point)

### 1a. Diagnose the ~6-run ceiling — one boot, decisive

Runs beyond ~5 die **host-side in region creation**, between `SQ: B/mkregion1` and
`SQ: C/mkregion2`, with no monitor tag — a different site from every S-07 wedge, which die *after*
`SQ: G/enter`. Two candidate resources are exhausted at almost the same point, and we do not know
which binds:

| candidate | evidence | cost to fix |
|---|---|---|
| `CAPSTONE_MAX_REGION_N = 32` (`sbi_capstone.h:53`) | `rgid` climbs ~4-5 per run from ~12; 32 is reached at ~run 5. The failure is *in region creation*. | **one `#define` + firmware rebuild — free** |
| rev-node pool, 1024 entries, bump-allocated with no reclamation | head read 425 after 2 domains → ~212/run → exhausts ~run 5 | RTL constant — needs a bitstream |

**The experiment:** one boot running ~8 small domains (rungs, not SQLite — fast), and at the
failure read `rgid` from the transcript plus rev-node head/overflow from switches **249/250**,
which the driver already reads at a wedge. Whichever counter is at its limit is the ceiling.

### 1b. Raise it

If region/domain ids bind: raise `CAPSTONE_MAX_REGION_N` and `CAPSTONE_MAX_DOM_N`
(`sbi_capstone.h:52-53`), rebuild firmware, re-run 1a to confirm reps/boot actually increased.
If the rev-node pool binds instead, that is an RTL ask to bundle into a *future* bitstream — do not
disturb the one now generating — and 1c still applies.

### 1c. Cut the per-boot cost

The JTAG image load is **133-227 s** and dominates a 5-minute boot. Two independent savings:

* **Prune the image.** Each SQLite domain is ~1.5 MB and we routinely carry three. Stage only what
  a given boot runs. `bake-sqlite-doms.sh` already prunes from a manifest; use it deliberately.
* **Skip the reload entirely when the image is unchanged.** This is the big one: repeated reps of
  the same firmware do not need a 200 s re-upload. The only thing blocking a warm restart is stale
  `.data` — `_relocate_lottery` and `_boot_status` (`fw_base.S:507-511`) are writable words that a
  reload happens to clear, and `fw_base.S:61-64` sends the boot hart into a wait loop when the
  lottery is non-zero. **Zeroing those two words over GDB before `monitor reset halt` should give a
  warm boot with no upload.** Contrary to the harness comment, there is no DDR init to re-run:
  `ariane_early_init` is an empty stub and no DDR/MIG code exists in the firmware.

## Priority 2 — accumulate the rate opportunistically, never in dedicated boots

Per the decision: no boots spent purely on statistics. Instead, **every** boot from here on appends
reps of the unmitigated `XU` after its real payload, and the driver appends one line per rep to a
cumulative log (domain, hash, outcome, DBAS, rgid, boot id). Two rules make the numbers mean
something:

* **Classify the failure, do not just count it.** An S-07 wedge dies after `SQ: G/enter`; a ceiling
  failure dies at `mkregion` before entry. Only the former counts toward k.
* **Report k/n, never "it wedges".** Anything that does not exceed the accumulated background is
  not a finding.

## Priority 3 — the root-cause lead: physical placement

`DBAS` is the domain's **true physical address** (`capstone.c:119` `__pa()`, traced to
`sbi_capstone.c:761`), and it varies per boot because the buddy allocator's state depends on boot
history — the module never frees. That makes placement the most promising modulator to test, and it
is controllable **without any code change**:

* **dummy domains first** shift the real domain's block (pages are never freed);
* **domain size** sets the allocation order and hence the alignment.

Once Priority 1 makes reps cheap: run the same image at deliberately different `DBAS` values and
compare rates. A correlation would be the first mechanistic handle on S-07 in weeks; its absence
strikes placement off the list. Cap: `CAPSTONE_MAX_DOM_N = 32` per boot.

## Priority 4 — audit the stale docs (delegated, proposals only)

`SILICON-BLOCKER.md` is ~6000 lines and its own most important conclusion contradicts most of the
document above it. Run **separate audit agents** over `agent-handoff/` to identify what is stale,
superseded, or retracted, and produce a **proposed** deletion/merge list with per-file evidence.
Nothing is deleted in this plan — the audit reports, the decision is the project lead's.

## Files

* **Diagnose/raise the ceiling:** `caplifive-system/sw/buildroot/components/opensbi/lib/sbi/capstone-sbi/sbi_capstone.h` (`CAPSTONE_MAX_REGION_N`, `CAPSTONE_MAX_DOM_N`)
* **Warm boot:** `tests/rtl-smoke/fpga_driver/run_ladder_perf_fpga.py` (`cold_boot`), reusing the existing `gdb_cmd` path
* **Rate log:** `tests/rtl-smoke/fpga_driver/run_sqlite_stages_fpga.py` (it already parses outcomes, DBAS and per-domain markers)
* **Reuse unchanged:** `bake-sqlite-doms.sh`, `verify-and-stage-rung.sh`, `check-capinit-slots.py`

## Verification

1. **The ceiling fix must be demonstrated, not assumed** — reps/boot before vs after, same domains.
2. **The warm boot must be proven by its own output**, not by absence of error: a warm-booted run
   must produce a fresh boot banner and a passing control. If it half-boots, fall back to the power
   cycle; that costs one boot's opening and nothing else.
3. **Every domain still verified by content hash inside the cpio**, never on the filesystem.
4. **Control first in every boot**; read no further than the first failure.
5. **Rate claims carry k and n.** No "X wedges" without both.
6. `precommit-scan.sh` before every commit and push.

## Explicitly not doing

No RTL change and no disturbance of the bitstream now generating. No dedicated statistics boots.
No deletion of documents inside this plan — Priority 4 proposes, it does not act. No further
mitigation A/B until a background rate exists to compare against; the double-load result stands as
one sample and is not to be cited as either success or failure.
