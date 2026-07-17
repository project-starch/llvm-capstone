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

The board console `fpga.corank.info` is **back up** (a colleague restarted the
backend; no more HTTP 503). Everything else is ready: the driver is
authoritatively confirmed against `socketio-api.md`, the history-seq bug is fixed,
and your `fpga_driver/` has been merged to `capstone-bootstrap`. **Run the real
sweep now.** The board URL (with its access token) is supplied separately at paste
time — runtime only, never commit or log it.

## Pre-flight (cheap, do these first)

1. **Image still staged?** Confirm `/tmp/capstone-b/fpga-image/fw_payload.bin`
   exists and matches the recorded sha256 (`aadd213f…`, from the phase-3b build
   record `c9b8127`). If the scratch was cleared, rebuild it per that record before
   running.
2. **Live connectivity + courtesy check.** Connect and read `user_count`,
   `power_state`, `load_state`. If `user_count` shows someone else likely mid-session
   (an unexpected non-idle state you didn't create), **back off and report** rather
   than stomp — the board is shared and just recovered.

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
