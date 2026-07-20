# Follow-up prompt for Agent-B — task 017: the board run (console is back)

*Paste below the line into `claude-b`. Self-contained. The console outage is over —
this is the real cycle-accurate perf run, the whole point of the task.*

---

You are Agent-B, continuing task 017 in `/home/alexey/dev/llvm-capstone-b`,
branch `capstone-bootstrap-b`.

```bash
cd /home/alexey/dev/llvm-capstone-b && git fetch origin && git pull --rebase
source capstone/tests/capstone-test-env.sh
```

## The gate is cleared

The board console `the FPGA web console` is **back up** (a colleague restarted the
backend; no more HTTP 503). Everything else is ready: the driver is
authoritatively confirmed against `socketio-api.md`, the history-seq bug is fixed,
and your `fpga_driver/` has been merged to `capstone-bootstrap`. **Run the real
sweep now.** The board URL (with its access token) is supplied separately at paste
time — runtime only, never commit or log it.

## Verify against the API before any hardware action (do these IN ORDER)

The point of this gate is to minimize the chance of the driver doing anything
unexpected on real hardware. Do **not** issue any state-changing action
(`power_toggle`, `POST /api/load-image`, `/api/reset-board`, `switch_toggle`, …)
until Stages A and B below are clean.

**Stage 0 — image staged?** Confirm `/tmp/capstone-b/fpga-image/fw_payload.bin`
exists and matches the recorded sha256 (`aadd213f…`, phase-3b build record
`c9b8127`). If the scratch was cleared, rebuild per that record.

**Stage A — offline conformance (no hardware).** Re-run
`python fpga_driver/test_dryrun.py` against the API-faithful `mock_server.py` →
must be green. This exercises the whole flow (Lock → upload → reset → sweep →
`--parse-uart` → release) plus the seq/history reconnect gating, entirely against
the mock that models `capstone/tests/rtl-smoke/socketio-api.md`. If it is not
green, stop here.

**Stage B — live protocol check, READ-ONLY.** Connect to the real board and
**observe only** — issue no state-changing event. On `connect` the server pushes a
snapshot (`load_state`, `flash_state`, `power_state`, `led_state`, `switch_state`,
`trace_state`, `auto_shutdown_state`, `gdb_state`) then `user_count`. Assert each
payload's **keys/shape match `socketio-api.md` exactly** (e.g. `power_state.state ∈
{on,off}`, `switch_state.states` has 8 entries, `auto_shutdown_state` has
`{timeout, locked}`). A benign `request_history{last_seq:-1}` read is allowed;
`power_toggle`/upload/reset/`switch_toggle` are **not**. Also the courtesy check:
if `user_count` or an unexpected non-idle state suggests someone else is
mid-session, **back off and report** — the board is shared and just recovered.
**If any live event diverges from the doc, STOP and report before touching
hardware.** (If the driver has no read-only/`--check-only` path, add a tiny one or
script a bare `FpgaConsole` connect that logs the snapshot and asserts shapes — do
not reuse the full run path for this.)

Only when Stage A is green **and** Stage B matches the doc do you proceed.

## The run

```
python fpga_driver/run_rtl_smoke.py --url '<token-URL supplied at paste time>' \
    --image /tmp/capstone-b/fpga-image/fw_payload.bin
```

Good-citizen cycle (the driver does this — verify it happens): **Lock**
(`set_auto_shutdown{locked:true}`) → power/upload (~2 min JTAG) → reset → wait for
the UART login prompt → run the `.user`+`.dom` pairs (borrow-cost + the three
revoke configs: bump / norevoke / revoke) → capture the UART `RESULT` lines →
`run-revoke-cost-fpga-qemu.sh --parse-uart` → **release the Lock**
(`locked:false`). Don't sit idle holding the Lock; release as soon as the sweep is
captured.

## What to report — the numbers are the deliverable

Report the parsed **cycle** breakdown next to the QEMU instruction-count reference,
so the paper can use it directly:

- **Revoke-cost (the headline):** bump / norevoke / revoke cycles/op, and the
  **revoke-at-free delta** (revoke − norevoke). QEMU reference was bump 7 / norevoke
  60 / revoke 65 → **+5 instr, O(1)**. State whether the RTL delta confirms O(1)
  contract-point revocation in real cycles.
- **Borrow-cost:** the raw / borrow / copy cycles/op (this also fills the C1
  spatial-overhead number, previously unmeasured).
- Any live-protocol surprise vs `socketio-api.md` (chunked upload, ack timing,
  reconnect behavior). If something differed, adapt `config.py`, re-green the mock,
  and note it.

## After the run

- Commit the results to `capstone-bootstrap-b`: the parsed breakdown + a short
  results doc, and a history note
  `capstone/agent-handoff/history/DD-MM-YYYY_HH-MM-SS_fpga-board-run.md`. Raw UART
  captures are fine to commit as evidence **only after confirming they contain no
  token** (they won't — the token is in the URL path, not the serial stream). Push.
- Report readiness for the small follow-up merge of the results into
  `capstone-bootstrap` and whether the numbers are paper-ready.

## Guardrails (unchanged)

- Additive **test tooling only** — no `llvm/`, monitor, submodule, or RTL changes.
- Token never enters the tree or any log; good-citizen on the shared board (Lock,
  short hold, clean state, back off if it's in use).
- Commit on `capstone-bootstrap-b`, exact paths, **no `Co-Authored-By:`**, no
  worker/agent identity in commit messages, no debug/report files.
