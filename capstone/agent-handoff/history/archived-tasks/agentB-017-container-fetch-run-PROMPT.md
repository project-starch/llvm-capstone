# Follow-up prompt for Agent-B — task 017: container-fetch + GDB boot, run the sweep

*Paste below the line into `claude-b`. Self-contained. The halt-before-bootrom half
works; the only gap is getting our image into DRAM while halted. This authorizes the
container-fetch route to close it and run the real sweep.*

---

You are Agent-B, continuing task 017 in `/home/alexey/dev/llvm-capstone-b`,
branch `capstone-bootstrap-b`.

```bash
cd /home/alexey/dev/llvm-capstone-b && git fetch origin && git pull --rebase
source capstone/tests/capstone-test-env.sh
```

## Where we are

Proven: `gdb_start` → `monitor reset halt` halts the hart in M-mode at the reset
vector **before** the bootrom reloads SPI. The only missing piece is loading our
15 MB `fw_payload` into DRAM while halted — the container can't see our uploaded file,
and `load-image` poisons the TAP and gets clobbered by the SPI reload. The route that
closes it: **have the gdb container fetch the image over the network, then `restore`
it while halted** — no `load-image`, no clobber, no TAP contention.

Use the **corrected** image: `fw_payload` sha **`5aed4793`** (SMP=n, `capstone.ko`
rebuilt against the UP kernel — the `6991c0f7` one had a vermagic mismatch and would
fail `insmod`). It is QEMU-validated end to end. Confirm it's durably staged (not just
`/tmp`).

## Token handling — the one hard rule

The board **access token** (in the console URL path) is a *credential*, not code.
The user has cleared its **transient, in-lab** exposure (e.g. appearing briefly in
`gdb_output`, which broadcasts to the co-lab colleagues already on the console) — so
that is **not** a blocker. The absolute rule that remains:

- **The token must never enter anything committed or pushed** — not a source file,
  not `config.py`, not a history note, not a committed log or UART/GDB transcript,
  not a commit message. The repo is public; a committed token goes global. Verify
  `git status`/`git grep` show zero occurrences before every commit, as you have.
- Light hygiene (best-effort, not a gate): if a **token-free** internal fetch route
  is trivially available (localhost / an internal endpoint / an unauthenticated
  `/api/images` variant), prefer it. Otherwise just use the token — pass it via a
  `curl -K <cfgfile>` config file (`chmod 600`, `shred -u` after) rather than in
  argv, so it isn't sitting in `ps`/shell history. Don't gate on any of this.

## Do it — fetch, boot, run

1. In the gdb container, fetch `5aed4793` into container-local storage (e.g. `/tmp`)
   via `curl`/`wget` from the console's image endpoint (`/api/images/<name>` or an
   internal equivalent).
2. `gdb_start` → `monitor reset halt` → `restore /tmp/<image> binary 0x80000000`
   (raw `.bin`, so `restore`). Then `set $pc = 0x80000000`, `set $a0 = 0` (hartid;
   the fpga/ariane OpenSBI embeds its own DTB, so `a1` likely doesn't matter) →
   `continue`.
3. Watch `uart_output` for OpenSBI → Linux → **shell**. With SMP=n there's no RFENCE
   dependency, so `/init` should exec cleanly. The rootfs should `insmod
   /capstone.ko` (you wired this) before the `.dom`s run.
4. Run the full sweep — the `.user`+`.dom` pairs (borrow-cost + bump/norevoke/revoke)
   — capture the UART `RESULT` lines → `run-revoke-cost-fpga-qemu.sh --parse-uart`.
5. Wire this into `run_rtl_smoke.py` as the `--boot-method=gdb` path (container-fetch
   + restore + set-pc + continue in place of load+reset), re-green the mock.

Respect the timings you learned: power-cycle + ~25 s settle before load paths, ~20 s
settle after `gdb_start` or OpenOCD reads all-ones, and the board wedges if left
continuously powered.

## Report — the numbers are the deliverable

Cycle breakdown next to the QEMU reference:
- **Revoke-cost:** bump / norevoke / revoke cycles/op and the revoke-at-free delta
  (QEMU: bump 7 / norevoke 60 / revoke 65 → **+5 instr, O(1)**); state whether the
  RTL delta confirms O(1) in real cycles.
- **Borrow-cost:** raw / borrow / copy cycles/op (fills the C1 spatial number).

## Guardrails

- **Non-persistent board ops are authorized** (GDB session, reset-halt, network
  fetch into the container, `restore` to DRAM, set-PC/continue, UART capture) under
  the standing board authorization.
- **Still STOP and ask** before any **non-volatile / SPI write** — a bitstream flash
  or a persistent firmware flash. That's a shared-infrastructure change, separate
  from the token question, and not autonomous.
- Good-citizen: Lock while working, release on exit, back off if someone else is
  mid-session.
- Commit on `capstone-bootstrap-b`, exact paths, **no `Co-Authored-By:`**, no
  worker/agent identity in messages, no debug/report files, token never committed.
  Additive test tooling only.

## Deliverables

- If GDB-boot reaches a shell: the real cycle numbers + `run_rtl_smoke.py` wired with
  `--boot-method=gdb`, committed with a history note
  (`capstone/agent-handoff/history/DD-MM-YYYY_HH-MM-SS_fpga-gdb-boot-run.md`).
- If it stalls: a precise, evidenced account of which step failed (fetch endpoint,
  container network, `restore`, entry state, or boot), so we know whether anything
  self-serve remains before the owner-flash fallback (the `5aed4793` image is ready
  to hand over).
