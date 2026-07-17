# Follow-up prompt for Agent-B — task 017: boot our image via GDB (self-serve, no owner)

*Paste below the line into `claude-b`. Self-contained. The board doesn't boot our
JTAG-loaded image because a **reset** reloads the SPI-resident firmware. But the
console's **GDB** path lets us start our loaded image WITHOUT a reset — a self-serve
route that needs no board owner and flashes nothing.*

---

You are Agent-B, continuing task 017 in `/home/alexey/dev/llvm-capstone-b`,
branch `capstone-bootstrap-b`.

```bash
cd /home/alexey/dev/llvm-capstone-b && git fetch origin && git pull --rebase
source capstone/tests/capstone-test-env.sh
```

## The reframe — don't go to the owner yet

Your finding stands: **Boot Images** is a JTAG load into DRAM (`0x8000_0000`,
volatile), and the console's **reset-board** makes the bootrom reload the
SPI-resident firmware, clobbering our load. **Bitstreams** flashes the FPGA `.bit`
(hardware design), not firmware, and would need Vivado + the private RTL — out of
reach. So neither the reset path nor the bitstream path is ours.

**But the reset is the enemy, not the load.** The console exposes a **GDB** session
(`gdb_start` / `gdb_input` / `gdb_output` / `gdb_state` in `socketio-api.md`;
OpenOCD + `gdb-multiarch`). That lets us **start our image directly, skipping the
bootrom reset** that reloads the resident firmware. This is self-serve,
**non-persistent** (flashes nothing, so zero shared-infra impact), and needs no owner.

## Pre-step (offline, cheap) — confirm the UP image is good

Before spending board time, QEMU-boot `fw_payload_up.bin` (SMP=n, sha `6991c0f7…`,
staged at `/tmp/capstone-b/fpga-image/`) to a shell and confirm the `.doms` emit
`RESULT` lines. QEMU has RFENCE so it won't test the fence fix, but it proves the
image is internally healthy before we drive hardware. (Also copy the image + recipe
somewhere durable, out of `/tmp`, so it survives your scratch being reaped.)

## Primary task — GDB-driven boot (do this live)

Goal: get **our** `fw_payload_up.bin` running to a Linux shell on the board without
the console's reset-board button. Work out the exact mechanics against the live GDB
session; a plausible recipe to start from (verify each step):

1. `gdb_start` to open the OpenOCD + GDB session.
2. In GDB, **`monitor reset halt`** — this halts the hart at the reset vector via the
   debug module **before** the bootrom fetches/runs the SPI-resident firmware. (This
   is the key: the debug reset-halt is *not* the console's reset-board button.)
3. Load our firmware into DRAM: `restore /path/fw_payload_up.bin binary 0x80000000`
   (it's a raw `.bin`, so `restore`, not `load`). ~15 MB over JTAG ≈ a couple of
   minutes, like the Boot-Images load. (Alternatively use Boot Images to JTAG-load
   first, then attach GDB — figure out which ordering the console actually allows;
   watch for OpenOCD contention between load-image and `gdb_start`.)
4. Set entry state: `set $pc = 0x80000000`, `set $a0 = 0` (hartid). The fpga/ariane
   OpenSBI **embeds its own DTB**, so `a1` likely doesn't matter — but if OpenSBI
   complains, point `$a1` at the DTB. Ensure the hart is in M-mode (it is after
   reset-halt).
5. `continue`, and watch `uart_output`/`gdb_output` for OpenSBI → Linux → **shell**.
   With SMP=n there's no RFENCE dependency, so `/init` should exec cleanly this time.

If a plausible variant of this reaches a shell, **you've unblocked the whole thing
with no owner.** Then wire it into the driver: add a `--boot-method=gdb` path to
`run_rtl_smoke.py` (gdb-restore + set-pc + continue in place of load+reset), re-green
the mock, and run the full sweep → capture `RESULT` → `--parse-uart` → report the
cycle breakdown (revoke-cost bump/norevoke/revoke → delta vs QEMU +5; borrow-cost
raw/borrow/copy).

## Secondary (only if GDB-boot stalls) — quick checks before escalating

- Re-scan the REST API / `app.js` for any **persistent boot-image write** (flash a
  `.bin` to SPI), distinct from the bitstream path — did we miss one?
- Capture the **current resident** boot log in full: does `jasonyu`'s firmware even
  reach a shell, or does it SD-init-loop forever? (If it never boots, the board is
  non-functional as flashed regardless — relevant to any eventual owner ask.)

## Guardrails — what's authorized vs what stops

- **Authorized (non-persistent, under your existing board authorization):** the GDB
  session, `monitor reset halt`, JTAG/`restore` loads into DRAM, set-PC + continue,
  UART capture. Drive these live yourself.
- **STOP and ask the user first:** anything that writes **non-volatile / SPI** — a
  bitstream flash, an SPI firmware write, or the console's persistent-flash path if
  one exists. That's shared infrastructure; do not do it autonomously.
- Good-citizen on the shared board: take the Lock while working, release on exit,
  back off if someone else is mid-session.
- Token never enters the tree or any log. Commit on `capstone-bootstrap-b`, exact
  paths, **no `Co-Authored-By:`**, no worker/agent identity in messages, no debug
  files. The driver stays additive test tooling; no `llvm/`/RTL/submodule-source
  changes.

## Deliverables

- The verdict on GDB-boot: **does our image reach a shell that way?** If yes — the
  cycle-accurate numbers (and `run_rtl_smoke.py` wired with the `--boot-method=gdb`
  path), committed + a history note. If no — a precise, evidenced account of what
  failed at which step, plus the secondary-check findings, so we know whether *any*
  self-serve path remains before considering the owner.
- History note → `capstone/agent-handoff/history/DD-MM-YYYY_HH-MM-SS_fpga-gdb-boot.md`.
