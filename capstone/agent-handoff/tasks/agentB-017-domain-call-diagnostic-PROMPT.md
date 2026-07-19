# Follow-up prompt for Agent-B — task 017: localize the domain-CALL stall (monitor vs RTL)

*Paste below the line into `claude-b`. Self-contained. Self-serve GDB boot works;
the block now is the Capstone domain CALL stalling after `create_dom`. Before any
owner hand-off, determine whether the stall is in **our** OpenSBI monitor (fixable)
or the **RTL** (escalate).*

---

You are Agent-B, continuing task 017 in `/home/alexey/dev/llvm-capstone-b`,
branch `capstone-bootstrap-b`.

```bash
cd /home/alexey/dev/llvm-capstone-b && git fetch origin && git pull --rebase
source capstone/tests/capstone-test-env.sh
```

## The reframe that changes scope

You've been treating "the monitor" as the owner's, but **GDB-boot loads *our*
`fw_payload`, whose OpenSBI is *our* build from `genesys-testing`.** So the Capstone
SBI handler stalling on the domain CALL is **our binary** — a *software* stall there
is in-scope to fix (rebuild our OpenSBI, keep the change as a patch/flag in our
staging), exactly like the RFENCE/SMP fixes. Only an **RTL/hardware** stall (the CVA6
capability-CALL itself not completing) is the owner's. This diagnostic decides which.

Note on validity: keep using **GDB-boot** — you detach before the `.dom`s, so the
measured loop runs on bare silicon and `mcycle` is native. We do **not** need
resident-flash for correct numbers; don't pursue it.

## Task 1 — localize the stall

1. **Simplest domain first.** Run the **borrow-cost** `.dom` (no region transfer) on
   the board. Does *its* CALL complete and emit `RESULT`, or also stall? This
   separates the **domain CALL / entry** itself from the **region-share** op:
   - borrow-cost also stalls → the CALL/domain-entry is the problem (leans RTL).
   - only revoke-cost stalls → the **region-share** SBI op is the problem (leans our
     monitor).
2. **Instrument our OpenSBI** (rebuild `fw_payload`): UART prints bracketing
   `create_dom` → region-share → the CALL/domain-entry. Also a print at the domain's
   **entry point** (first thing the `.dom` runs). Then:
   - last monitor print is *before* the CALL and the domain-entry print never appears
     → control left to the domain and didn't return → **hardware/domain-switch**.
   - a monitor SBI handler prints entry but not exit → it's **spinning in our
     monitor** → software, ours to fix.
   (GDB desyncs across a domain switch, so use UART prints, not stepping.)

## Task 2 — test the icache hypothesis (our-fixable if it hits)

RFENCE and `insmod` both hung on this CVA6's icache/text-patch path. A domain CALL
switches execution context, which also needs icache coherence — the stall may be the
**same class**. Try a monitor-side local `fence.i` (and/or the CVA6's icache-flush
sequence) on domain entry/exit in our OpenSBI, rebuild, retest. If the CALL then
completes → self-fixed.

## Task 3 — config comparison (Q3 angle)

Check whether **our** OpenSBI build fully enables/configures the Capstone domain
CALL + region-share path, vs the `genesys-testing` reference / whatever the resident
`jasonyu` firmware uses. A build-config or feature-flag difference in *our* monitor
(not the RTL) would be the easiest fix of all.

## Decision

- **Software (our monitor):** apply the fix (icache fence / config / handler bug) in
  our OpenSBI build, rerun the sweep, and report the **cycle numbers** (revoke-cost
  bump/norevoke/revoke → delta vs QEMU +5; borrow-cost raw/borrow/copy).
- **RTL (hardware):** STOP. Produce a precise characterization: exact stall point
  (create_dom OK → which op → control left to domain / never returned), the icache
  hypothesis and what you tried, the minimal repro. Note — but do **not** attempt —
  the heavier path (an RTL patch on a new `capstone/capstone-ariane` branch →
  Vivado synth → bitstream flash), which needs the lab's toolchain + a persistent
  flash and is a separate decision.

Time-box this: the paper's load-bearing perf result is the QEMU comparison; the RTL
figure is confirmatory. If it's RTL, escalate cleanly rather than grind.

## Guardrails

- **In-scope:** rebuilding our `fw_payload` OpenSBI/kernel (patches/flags in our
  staging), GDB-boot, non-persistent board ops, UART capture — all under the standing
  authorization.
- **STOP and ask** before any **non-volatile / SPI write** (bitstream or firmware
  flash) and before any **`capstone-ariane` submodule-source commit** — keep RTL
  experiments, if any, on a scratch branch, not committed, and don't flash.
- Token never enters anything committed/pushed (in-lab transient exposure is fine).
  Good-citizen on the board (Lock, release, back off). UART RX is lossy — throttle.
- Commit on `capstone-bootstrap-b`, exact paths, **no `Co-Authored-By:`**, no
  worker/agent identity in messages, no debug files. Additive test tooling + our-image
  build changes only.

## Deliverables

- The verdict: **monitor-software (fixed → numbers) or RTL-hardware (precise
  hand-off)**.
- If fixed: the real cycle-accurate breakdown + committed driver/image updates + a
  history note (`.../history/DD-MM-YYYY_HH-MM-SS_fpga-domain-call.md`).
- If RTL: the surgical characterization + whether an `capstone-ariane` RTL fix looks
  feasible, for a go/no-go on that heavier path.
