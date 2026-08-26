---
name: board-run
description: >-
  Run a program on the Capstone CVA6 FPGA board: bake it into the buildroot image, boot,
  invoke it, classify the result, release the board. Use for ANY board execution — silicon
  ladder rungs, SQLite staged probes, a new minimal repro — and before interpreting a run
  that produced no result. Covers the ordering rule that makes verdicts valid, the
  entry-stall vs real-wedge distinction, and the checks that must pass BEFORE spending a
  boot. Never ship a program over UART.
---

# Running a program on the Capstone FPGA

Board time is the scarce resource and most wrong verdicts in this project came from the
harness, not the hardware. This skill is the decision procedure. Full background:
`capstone/agent-handoff/ref/HOW-TO-LAUNCH-ON-FPGA.md`.

## 0a. FIRST: search the handoff docs for the symptom AND the failing function name

Before designing any experiment, grep `capstone/agent-handoff/` for the symptom and for the name
of the function/construct involved. On 2026-08-06 hours went into re-deriving a localization that
already had a **dedicated design doc** (`design/cap-local-aggregate-init-plan.md`), a **four-variant
board experiment** that had already isolated the shape (`ref/ISSUES.md` ~:1543), and **two landed
compiler fixes**. A single grep would have redirected the whole day.

```bash
grep -rn "<failing function>" capstone/agent-handoff/ | head
grep -rln "<symptom phrase>"  capstone/agent-handoff/
```

Prior art also tells you which prior results are trustworthy: several were measured with
instruments later shown to be broken, and the docs say so.

## 0b. RUN THE PREFLIGHT GATE

    bash capstone/tests/preflight-board-run.sh      # exit 1 = BLOCKED

Deterministic, never delegated, same standing as `precommit-scan.sh`. It checks the construct is
in the artifact, images are distinct, the control has a published passing record
(`ref/known-good-controls.md`), the DTS matches the resident bitstream, and the slot budget.

## 0. Before you spend a boot — three offline checks

Each of these has cost real board time when skipped.

1. **Verify the artifact does what the source says.** Disassemble and confirm the construct
   under test is actually present. A "repeat the load N times" ladder was CSE'd into ONE
   `ldc` regardless of N — memory barriers did not stop it — so the whole set tested nothing.
   ```bash
   llvm/cmake-build-debug/bin/llvm-objdump -d --triple=capstone64-unknown-elf <x>.dom
   ```
2. **Make every run RETURN a number.** A probe that hangs yields one bit ("somewhere after
   the last marker"). A probe that returns a wrong value tells you which slot, how many, and
   is bisectable. Prefer a clamp/early-return/sentinel over observing a hang.
3. **Give each arm a distinct sentinel.** Never reuse a value that a legitimate control also
   returns, or "not compiled in" reads as "control passed".
4. **Is the construct REACHED, not just present?** The costliest error of the 2026-08 session:
   five probes were built against a glue that never calls the code they tested. The symbol was in
   the binary and disassembled fine — it was dead. `DOMAIN_GLUE` defaults to **`generated`**
   (`build-ladder-domain.sh:22`), and `start-gp-captable-generic.S` has **zero** references to
   `cap_init`, while `start-gp-captable-interp.S` has 15. So:

   * a probe with **capability-bearing initialised globals** (`static char *p = data;`) MUST be
     built `DOMAIN_GLUE=interp`, or those globals silently never get a tag — correct under QEMU,
     wrong on silicon, with no build-time signal;
   * confirm reachability, e.g. `grep -c cap_init` on the glue you actually built with, or a
     probe whose value can only be produced by the code under test.

   Note the QEMU differential CANNOT catch this class: code that never runs passes emulation too.

5. **Sizing knobs — use the ones that exist.** `DOMAIN_WINDOW=<bytes>` sets the globals window
   (`build-ladder-domain.sh:42-45`) and `DOMAIN_BASE_VA=` relocates the entry (`:66-67`); they
   are different things. `DOMAIN_WINDOW=0x150000` reproduces SQLite's geometry on an 11 KB rung.
   **Do not hand-write a linker script:** `link-gpfree-32k.ld` does NOT place
   `.capstone_gp_initdesc`, so lld orphan-places it, `globals_off` reads as ~0x3f0 and the
   monitor aborts with `capstone_error 0xB10B`. Use the DEFAULT `link-gpfree.ld` plus
   `DOMAIN_WINDOW`.

## 1. Bake it in — never UART

UART delivery is 16 chars per HTTPS round trip: minutes per ~10 KB domain. Baked, ten
domains run in ONE boot in ~5 minutes, because a 10 KB file is free inside a JTAG upload
that happens anyway.

```bash
O=capstone/caplifive-system/sw/buildroot/overlay/test-domains
T=capstone/caplifive-system/sw/buildroot/build/target/test-domains
cp -f <artifact> "$O/" && cp -f <artifact> "$T/"      # BOTH — buildroot packs $T
cd capstone/caplifive-system/sw/buildroot
make build LINUX_PAYLOAD=1 A=linux-rebuild  CAPSTONE_CC_PATH="$(realpath ../../../capstone-c)"
make build LINUX_PAYLOAD=1 A=opensbi-rebuild CAPSTONE_CC_PATH="$(realpath ../../../capstone-c)"
```

`A=linux-rebuild` **first**: buildroot does not track `overlay/` → cpio, so an OpenSBI-only
relink silently ships the OLD initramfs.

**Retire before you stage, every time — the image is DERIVED from the run, not accumulated:**

```bash
bash capstone/tests/stage-board-domains.sh --apply $BAKED_RUNGS   # dry-run without --apply
```

It makes overlay **and** the buildroot target dir hold exactly {controller} ∪ {package files} ∪
{this run's rungs}, moving everything else to `/tmp/capstone/overlay-attic` — by explicit name,
never a glob (a prefix glob once deleted the package-installed `sbi.dom`), and nothing is deleted
because a `.dom` is cheap to keep and expensive to rebuild once its flags are forgotten.

**This applies to EVERY staging path, including a purpose-built bake script.** The rule is easy
to skip precisely because skipping it is invisible: each bake works, and the image just grows.
On 2026-08-13 a SQLite bake script staged its variants directly and never retired the previous
set, so fourteen ~1.5 MB domains accumulated across one afternoon — **23 MB of domains of which
four were live**, and the initramfs went 21.7 MB → 27.9 MB. Nothing failed; every boot after that
just paid for it on the JTAG upload, and the growth is fastest exactly when boots are most
frequent. A bake helper that does its own staging must retire its own previous set — from a
**manifest of what it staged**, so it can only ever remove what it can prove it created — or call
the script above. `bake-sqlite-doms.sh` now does the former and prints the image size delta.

Check the size when a boot feels slow, and treat growth as a bug rather than as weather:

```bash
ls -la capstone/caplifive-system/sw/buildroot/build/images/rootfs.cpio
du -sh capstone/caplifive-system/sw/buildroot/overlay/test-domains
```

(Image size is **not** a known cause of the R-16 entry stall — that was measured and ruled out:
zero bytes transferred, 12 loads inside a 300 s budget, pruning every domain saved 0.47 %. Shrink
the image to keep boots cheap, not as a remedy for a stall.)

Why a script rather than "prune when it looks big": the overlay reached 111 files / 35 MB in one
session, costing 30.6 min of JTAG across 39 boots plus a boot lost to HTTP 413. The gates added
afterwards did not close it — measured against the next session's 26 files / 1.91 MB, C9 would
have blocked on **nothing** (25 were under its per-file threshold, and the 1.5 MB one was on its
exemption list). Accumulate-then-prune fails because the prune depends on someone noticing.

Firmware identity is the **hash, never the size**: buildroot pads in 2 MiB steps, so two
different images routinely have identical sizes.

## 2. Run it

```bash
export FPGA_URL="$(cat ~/.claude-c/secrets/fpga-console-url)"   # credential — never commit/echo
export FPGA_FW=.../opensbi-custom/build/platform/fpga/ariane/firmware/fw_payload.bin
```

| Running | Driver | Key env |
|---|---|---|
| ladder rungs | `fpga_driver/run_baked_rungs_fpga.py` | `BAKED_RUNGS`, `BAKED_TIMEOUT`, `LADDER_FPGA_DIR`, `BAKED_OUT` |
| SQLite / staged probes | `fpga_driver/run_sqlite_stages_fpga.py` | `SQLITE_STAGE_DOMS` (`dom:selector`), `SQLITE_STAGE_TIMEOUT`, `PROBE_SCOPED_OUT` |

**`SQLITE_HOST` applies to EVERY arm, and the two hosts take different argument orders.**
`lpc` (ladder) is `lpc <name> <dom>`, so ladder entries are written `name:/path/x.dom`.
`sqlite_host.user` is `sqlite_host.user <dom> [--slt <test>]`, so its entries are `/path/x.dom`
or `/path/x.dom:--slt /path/case.test` — **no label**. Put a `label:` on a SQLite entry and the
label becomes argv[1], i.e. the domain path: the loader prints `Failed to open the file.` and
the domain never loads. Because one host serves the whole run, **a ladder rung and a SQLite
domain cannot share a boot** — pick a control that uses the same host (for SQLite, the same
domain with no `--slt` runs its built-in workload and returns).

The driver already hard-stops on this and calls the arm a PHANTOM, refusing to read the
`SQ: G/enter` / `SQ: H/return` markers that follow — they belong to a domain that was never
loaded. Trust that guard; it is there because those markers otherwise look like a real run.

```bash
cd capstone/tests/rtl-smoke
BAKED_RUNGS="ctl_rung unknown_rung" python3 -m fpga_driver.run_baked_rungs_fpga
```

`FPGA_FW` has no default on purpose — an implicit one silently boots whatever was built last.

**If you supervise the run with `board-watchdog.sh` (the SQLite staged path does), keep
`ENTRY_STALL_S` ≥ 260.** The JTAG upload is **133–227 s of entirely legitimate UART silence**
(~130 KiB/s), so a lower threshold aborts healthy runs *mid-upload*. The old 45 s default did
exactly that and produced a session's worth of false "will not boot" / "cyclic boot" /
"firmware is broken" diagnoses. Scope any log scan to **this run's own `load_image`** — the
console replays ~548 KB of the previous boot on connect, so grepping the whole log matches
stale markers.

### Switch values are not just a mux selector — the low three bits steal the console

Setting the virtual switches to read a debug register also drives three control bits, so a
**switch value is UART-safe only when `(value & 0b11) == 0`**:

| bit | effect |
|---|---|
| `sw[0]` | hands the **console TX pin** to the tracer (`uart_debug_takeover = sw[0] \| …`, and `uart_debug_tx_o = switches_i[0] ? tracer_uart_tx : uart_debug_tx`). Shell output is not lost — it is never transmitted. |
| `sw[1]` | **arms a one-shot trace dump** over that same pin. Edge-triggered, streams the whole buffer, and **outlives the switch value that armed it**. |
| `sw[2]` | tracer ring-buffer mode. Harmless. |

Cost of learning this the hard way: **one boot read as a kernel hang** (switches left at 255 —
odd — so the next domain's shell line was never echoed; the core was fine), and two more killed
pre-emptively. A domain that goes quiet because its TX pin was taken is **indistinguishable from
one that wedged**.

Rules that follow:

* **Never HOLD an odd value while a domain runs**, and park back at 0 after any read. Most useful
  apertures are odd — `191` (the trap-log clear), `197/199/201/203`, `225`, `249`, `255` — while
  `196`, `200`, `204`, `224` are safe.
* **The trap summary (`255`) has no safe aperture.** It exists only at bank `3'b111` reg
  `5'b11111`; `254` is a different field. Do not sample it mid-run.
* **Beware the armed dump.** `191` sets `sw[1]`, so every trap-log clear arms a dump. Holding any
  odd value later reconnects the tracer **mid-stream** and injects binary into the console.
* **Post-run odd reads are safe by construction**: `debug_led_o` goes to the LEDs, a different pin
  from the console TX, so an odd aperture cannot corrupt the reading itself — it can only steal
  console output happening at the same time.

## 3. THE ORDERING RULE — this is what makes a verdict valid

**The drivers do not reboot between programs, and a wedged program takes the core.**
Everything after the first failure is collateral, not a result.

* **At most ONE unknown per boot**, placed **last**, after a known-good control.
* **Read a run no further than its first failure.**
* Everything expected to return goes first, ascending.

This produced a wrong verdict within an hour of the baked path existing: an arm failed at
position 4 of a 6-program boot, and positions 5 and 6 were recorded as failures. Re-tested
one-per-boot, position 5 **passed**.

**Always run a known-good control FIRST in every boot.** It separates "this image failed"
from "board/firmware/boot failed" — and the control has its own failure rate, so a boot
whose control fails is **VOID**, carrying no verdict about anything.

If using `run_ladder_perf_fpga.py` with `LADDER_ONE_BOOT=1`, rungs must be linked at
**distinct entry VAs** (`DOMAIN_BASE_VA`), or R-3 silently hangs a second domain reused at
the same VA — a hang that looks exactly like a rung result.

## 4. Classify — three things are NOT results

Read the **last marker** in the run-scoped transcript, never "did not return":

| Evidence | Meaning |
|---|---|
| `SQ: obs=<n>` / `RESULT … retval=` | a result — record it |
| `SQ: G/enter` present, no `H/return` | **NOT proof of entry** — see below |
| `ENT1` present, no `ENT2` | control left M-mode: the DOMAIN owns the wedge — a **real** result |
| `ENT0` present, no `ENT1` | died in the monitor's `call_domain`, before the switch |
| no `SQ: G/enter` (ends at `SHA5:`) | **entry stall** — the domain never ran; says nothing about the code |
| no boot banner / JTAG errors | infra — retry |
| `ConnectionError`, HTTP 5xx on connect | **console/web-UI outage** — not a board fault |
| connect OK but **zero** UART bytes all run | **AMBIGUOUS** — dead firmware *or* dead UART relay |
| GUI terminal shows only `?`/mojibake, no text | **web-UI fault** — not board output, not a baud problem |
| `Proxy Error … Error reading from remote server` | the GUI is **rebooting** — transient, wait and retry |

**`SQ: G/enter` does NOT mean the domain entered.** It is printed by the HOST
(`sqlite_host.c:144`) *before* it calls in, and the monitor's `call_domain`
(`sbi_capstone.c:838`) was uninstrumented until 2026-08-05 — every `SHA*`/`ECSZ` tag belongs to
the region-share path. So "G/enter then silence" was consistent with dying in `call_domain`, in
the domain switch, at the first instruction, in the carve loop, or in cap-init, and calling it
"entered and wedged — a real result" was an unattributed guess. Use **`ENT1`** (about to leave
M-mode) and **`ENT2`** (domain returned) instead; for a ladder rung the equivalent entry proof
is **`SHA6`**, whose absence means it died in the FIRST region share, before entering.

`SHA5` last does **not** by itself mean an entry stall: a domain that enters and wedges
immediately also leaves `SHA5` last. **Distinguish on `SQ: G/enter`.**

A wrong *value* is a result too, and a valuable one — it is bisectable where a hang is not.

**Entry stalls (R-16) were FIXED IN SILICON on 2026-08-04** by `caplifive_fixed_forward.bit`
(capability operand forwarding, `capstone-ariane 7aac52f93`) — as was R-14, the same defect.
Keep the row above: the classification still governs every run, and the failure returns if the
board is reflashed to a bitstream without that fix. `capstone/tests/fpga-repros/R16-entry-stall/`
checks a bitstream in one boot. If you *do* see an entry stall, suspect the bitstream first.

**If it recurs, retrying the same binary is futile — it was per-image.** REDRAW instead:
rebuild with a harmless constant varied (e.g. a different compiled-in default stage) so the
code under test is byte-identical across draws. Always `sha256sum` the set and abort if any
two match.

## 4b. Waiting for a run to finish

**Do not spawn a second task to poll for something the runner already reports.** The background
run notifies on its own completion; a separate waiter duplicates that and can fail in ways the
runner cannot. Five such waiters ran 7-11 minutes each and never exited.

If you do need to wait (polling something outside the harness), use
`bash capstone/tests/wait-run.sh <logfile> [max_s] [stall_s]`, which cannot hang. Hand-rolled
`until grep ...; do sleep; done` loops have failed twice over, both worth naming:

* **The terminal pattern assumed the run reaches its LAST arm.** `<-- TEST 3/3` never appears
  when arm 1 wedges — and the driver stopping there is *correct*, "everything after this is
  lost". So the waiter hung precisely on the runs that carried the most information.
* **It grepped for a string never written to that file.** `EXIT=$?` goes to the task's stdout,
  not into the driver log being polled, so the pattern could not match at all.

Key on markers the driver emits on EVERY exit path — `BOARD_RELEASED`, `preflight BLOCKED`,
`Traceback`, `GUEST_RC` — and bound the wait two independent ways (no log growth for N seconds,
plus a hard cap), so a future pattern mistake degrades into an early exit rather than a hang.
Watch **log growth**, not process exit: a hung driver never exits.

## 5. Release the board, always

Lock → power-cycle → run → **power off + unlock in `finally`**. The drivers do this via
`safe_cleanup.release_board`.

* **Never `kill -9` a runner.** It orphans the server-side GDB session in `error`; every
  later run then times out before `load_image`, and it survives a power cycle. Only
  `gdb_stop()` clears it.
* Signal by **verified PID**, never a bare `pgrep -f <pattern>` — that matches your own
  shell, and once matched an editor with the script open and closed it.
* Confirm `gdb_state` is `idle` before blaming the board.

### The board looks dead: OpenOCD cannot claim the JTAG adapter

Symptom — every `gdb_start` fails server-side and `cold_boot` burns all three retries with
`timed out waiting for event 'gdb_state'`:

    [GDB] Start failed: OpenOCD exited during GDB startup (code 1)
    Error: libusb_claim_interface() failed with LIBUSB_ERROR_BUSY
    Error: unable to open ftdi device ...
    Error: [riscv.cpu] Unsupported DTM version: -1     <- CONSEQUENCE, not a second symptom
    Error: [riscv.cpu] Could not identify target type. <- likewise

**The FPGA is usually fine** — check the UART, which will show a clean `Hello World! …
booting!`. Only JTAG is unavailable.

**RECOVERY, and run it in this order:**

1. **Clear every switch to 0**, bit 7 first. `sw[0]` muxes the console TX to the tracer and
   `sw[1]` triggers a dump, so an aperture walk can leave the console owned by the tracer.
2. `gdb_stop()` — documented to terminate OpenOCD on the host.
3. **A real power cycle**: off, settle ~6 s, on, settle ~12 s. Not `POST /api/reset-board`,
   which succeeded and changed nothing.
4. **`gdb_start()` as a PROBE.**

**Step 4 is the only step that answers anything.** Steps 1-3 all report success whether or
not the adapter is free. `gdb_stop()` taking `gdb_state` from `error` to `idle` looks exactly
like a recovery and is not — the next `gdb_start` goes straight back to `error`. **Clearing
state is not clearing the device.**

**Probe with the board POWERED ON.** With power off there is no DTM, so `gdb_start` fails
whether or not the FTDI is free, and it fails with the same two trailing lines. A probe taken
with power off cannot separate *"USB is busy"* from *"there is no target"*, and reading its
failure as evidence about the adapter is a wrong verdict.

**Honest limit on this recipe:** when it first worked, switches were cleared AND a full power
cycle was done in the same attempt, so **which step recovers it is not established** — an
earlier attempt with `gdb_stop` + `reset-board` + power-on (no switch clear, shorter settles)
did NOT recover it. If you hit this, consider varying one step at a time and recording which
one worked; nobody has done that yet.

If it survives all four steps, the adapter is genuinely held by another process. Check
`user_count` for other sessions, then it needs host-side intervention — killing the stale
OpenOCD on the board server — which the socket.io/HTTP surface does not expose.

`/tmp/capstone/recover_full.py` implements the sequence.

## 5b. After a RE-FLASH: the memory map may have moved

A new bitstream can change the reserved capability-memory constants, and the device tree must
match or Linux dies in early init — every boot, at the same point, just after
`riscv-intc: 64 local interrupts mapped`. It looks identical to a dead board.

On 2026-08-05 the 65536-node bitstream moved `CAP_TAG_MEM_BASE` from `0xBC3C_0000` to
`0xBC2D_2D2D`; the DTS still said `0x3c3c0000`, so Linux was handed ~971 KB of shadow-tag memory
as RAM. Two boots were spent before the map was suspected, and **extending the boot window only
hid it** — the window must stay fixed so a stall reads as a stall.

So after any reflash:

1. Read `CAP_TAG_MEM_BASE` from `capstone-ariane/core/include/ariane_pkg.sv` **at the bitstream's
   commit** (`git show <commit>:core/...`), or run `calculate_memory.py`.
2. Set `reg = <0x0 0x80000000 0x0 (BASE - 0x80000000, page-aligned down)>` in BOTH
   `caplifive.dts` and `configs/caplifive.dts`.
3. Verify the value is in the **built firmware's DTB**, not just the source.
4. Also update `FPGA_BITSTREAM` — the drivers hard-stop on a mismatch, which is the gate working.

Full recipe: `agent-handoff/ref/HOW-TO-LAUNCH-ON-FPGA.md`.

## 6. Before concluding the hardware is broken

Rule out the harness first. On 2026-08-02 the large majority of apparent board failures were
self-inflicted; the same firmware file was declared dead six times, then booted five times
with no rebuild in between. The discriminator that settled it: the variable was the harness,
not the firmware — diff what actually changed on our side before spending a rebuild or a
reflash.

**A CONTROL THAT FAILS THE SAME WAY TWICE IS THE HARNESS, NOT THE 1-IN-5 FLAKE.** The control
fails on its own roughly 1 boot in 5, so one failure is genuinely a void boot worth retrying.
But that flake is an entry stall — it varies. **Two boots whose control dies with an identical
signature are a variable you introduced**, and retrying spends a second boot to learn nothing.
Diff the driver log of the failing boot against the last boot whose control PASSED and read
forward to the first line that differs; on 2026-08-20 that line was
`[s07] early halt control failed (ActionTimeout) -- no verdict, and the run continues`, an
optional pre-run diagnostic that timed out with the core left HALTED, after which every stage
timed out at `SQLITE_STAGE_TIMEOUT` and read exactly like a wedge.

Generalising, because this is the expensive half: **an optional diagnostic must never be able
to cost a boot silently.** Anything that halts the core, drives the switches, or opens GDB
before the arms run has to either prove it put the core back — the shell answering is the
proof — or abort the boot loudly. "No verdict, and the run continues" is fail-OPEN, and a
fail-open pre-run step converts itself into N false wedges. When a diagnostic has already
answered its question in an earlier boot, turn it off rather than re-running it.

**THE BOOTROM BANNER IS THE FIRST DISCRIMINATOR. Check it before suspecting anything you built.**

The FPGA bootrom prints, on power-on, BEFORE the JTAG firmware load:

```
Hello World!
Hit any key to enter update mode .. booting!
init SPI / SPI initialized! / initializing SD...
```

In a healthy run this appears at UART events ~25-28, ahead of `emit gdb_start` and well ahead of
`monitor load_image`. **It is emitted before your firmware is written to DDR, so if it is
ABSENT the fault is upstream of everything you built** -- device tree, `.dom` staging, monitor,
`.c.S` regeneration, branch, image size. Do not bisect a build against it.

    grep -c "Hello Wo" <run.log>     # 0 with power-on in the log => board/console serial fault

On 2026-08-05 a session went from "the board will not boot" through a bitstream reflash, two
device-tree reverts, a firmware rebuild and a monitor bisection before anyone checked this. The
banner was absent the whole time; the console was at fault, which the board owner confirmed.

**Check the console is actually up before diagnosing anything.** On 2026-08-05 the backend
returned **HTTP 502** and a session was spent diagnosing "the board will not boot" — reflashing,
reverting the device tree, rebuilding firmware, bisecting the monitor — against a console that
was not talking to the hardware at all. The board owner confirmed it was a web-UI error.
`connect()` now names this explicitly, but check it first whenever runs stop reaching the board:

```bash
curl -s -o /dev/null -w '%{http_code}' "$FPGA_URL"      # 200 = up; 5xx = outage, stop here
```

**A run that connects but receives ZERO UART bytes does NOT prove the firmware is broken.**
That is the failure mode that misled the same session. Silence is equally consistent with a
console whose UART relay has failed while its HTTP side still answers — and it looks identical
from the GUI, so "the user sees nothing either" is corroboration of nothing. Before blaming an
image, establish that the console can carry UART at all: the connect-time replay of the previous
boot (~548 KB) is the cheap positive control — if even that is absent, suspect the console.
**Garbage in the GUI terminal is a WEB-UI symptom here, not board output.** When the console
shows only replacement characters / mojibake and no readable text, that has been the web UI
failing — confirmed with the board owner 2026-08-05. Do NOT read it as a baud-rate or
clock-frequency mismatch and do not go editing `current-speed`/`clock-frequency` in the device
tree over it; that is a wild-goose chase this session started. (A short undecodable prefix at
the very start of a *working* session is separate and harmless — the console replays the tail of
the previous boot on connect. The signal is mojibake *instead of* boot text, not *before* it.)

The proxy sits in front of the console, so during a GUI restart you get an Apache
`Proxy Error … The proxy server received an invalid response from an upstream server` /
`Error reading from remote server`. That is the GUI coming back up: wait and retry, do not
diagnose hardware, do not reflash. Same for a 502 that clears on its own.

**NEVER call `console.trace_dump()` — it hangs the board hard.** Measured 2026-08-05 on a
wedged core: `trace_result` never arrives, the wait expires, and the board is left needing
manual recovery. It cost a reflash. The function is still in `fpga_console.py` and reads like
exactly the instrument a wedge investigation wants, which is precisely the trap. Same caution
for the GDB tab: an orphaned session wedges every later run and only `gdb_stop()` clears it.

**Do NOT scan a raw UART transcript with `grep` — use `python3` byte search.** `grep` here is
`ugrep`, and on a transcript containing control bytes it prints **nothing at all** — not `0`,
no output whatsoever — and exits 1, for a string a byte search finds repeatedly. Reproduced on
this session's own data:

```
$ python3 -c "d=open('/tmp/capstone/boot30-raw.txt','rb').read(); print(d.count(b'SQ: obs='))"
4
$ grep -c "SQ: obs=" /tmp/capstone/boot30-raw.txt     # prints NOTHING, exit 1
```

An empty output is worse than a silent zero: it reads as "absent" *and* as a failed command. Same family as the `awk strtonum` returning 0 and the
hex-constant-emitted-in-decimal incidents.

```python
python3 -c "d=open('/tmp/capstone/bootN-raw.txt','rb').read(); print(d.count(b'RESULT'))"
```

Verdicts taken from a driver's own stdout summary are unaffected — the runners parse run-scoped
text with Python regex. This applies to hand-grepping the raw capture.

**Bitstream re-flash is ask-first, always.**

## Re-flashing a bitstream

**Two stores, and using the wrong one wastes a session.** The console keeps boot images and
bitstreams separately:

| store | endpoint | holds | wrapped in our driver? |
|---|---|---|---|
| images | `/api/images/upload` | `fw_payload.bin` | YES — `upload_boot_image()` |
| **bitstreams** | `/api/bitstreams/upload` | `.bit` | **NO — deliberately not wired** |

`flash_bitstream(name)` names a **server-side** bitstream. So:

* **If the `.bit` is already server-side** (most are — the board owner puts them there), no upload
  is needed and a flash is a few lines. `caplifive_s10fix_80843404c.bit` and ~25 others are there.
* **If it is only local, our driver CANNOT put it there.** `upload_boot_image()` will happily
  accept an 11 MB `.bit` and file it under **images**, where `load_image` JTAG-loads to
  `0x80000000` as firmware — a hazard sitting next to the `fw_*.bin` files, and the flash then
  has no such name to resolve. Use the GUI Bitstream Manager or ask the board owner.

**Sequence, in this order.** Each step exists because skipping it has cost a session:

```python
c.power(True); time.sleep(15.0)      # POWER ON FIRST -- a cold board races the JTAG programmer:
                                     # flash_state -> error in ~1 s with NO SPI write
c.lock()                             # the flash takes ~90 s; auto-shutdown is 600 s
c.flash_bitstream(NAME)              # server-side name
c.power(False); time.sleep(8.0)      # POWER CYCLE IS MANDATORY -- the flash writes SPI only, the
c.power(True);  time.sleep(15.0)     # FPGA keeps the old config until it reconfigures at power-on.
                                     # Skip it and the DTM comes up IDCODE 0x00000001.
rb = flash_state["nv_bitstream_name"]   # RE-READ. Never trust the call.
```

**Reading the resident name is itself a trap, twice over.** `_current_state()` returns only the
`state` field and **drops `nv_bitstream_name`**; and registering the event handler *after*
`connect()` misses the initial state burst. Both return `None`, which is indistinguishable from
"a non-Capstone design is resident" — so both read as a failed flash that actually succeeded.
Register the handler **before** connect, take the full payload, and settle before reading.

**After any re-flash, before trusting a result:**

* **Check whether the memory map moved** — `CAP_TAG_MEM_BASE` / `CAP_REVNODE_MEM_BASE` in
  `capstone-ariane/core/include/ariane_pkg.sv` **at the new bitstream's commit**. If it moved,
  update `reg = <...>` in BOTH `caplifive.dts` and `configs/caplifive.dts` and verify in the built
  DTB. (`git show <commit>:core/include/ariane_pkg.sv` answers this in seconds and is worth doing
  every time — it cost two boots once.)
* **Pass `FPGA_BITSTREAM` explicitly per run** while experimenting. The drivers' defaults go stale
  and a stale default burns a launch on a HARD STOP; update them once, after the experiment
  settles on a resident image.
* **Name the bitstream in every recorded result.**

**Only `run_ladder_perf_fpga.py` flashes** (gated on `FPGA_ALLOW_FLASH=1`). In
`run_base_bare_fpga.py` that same variable **bypasses the HARD STOP without flashing**, so the run
proceeds on the wrong silicon — do not set it there expecting a flash.

## Reporting a result

State which arms were **reachable**, not just which failed — an entry stall biases *which
constructs can be measured at all*, so "arm X fails and arm Y does not" is unsupportable
unless Y actually entered. Quote the control's verdict alongside every result.

**Name the bitstream in any result you record.** R-14 and R-16 were both fixed by a reflash on
2026-08-04, which invalidated every board measurement taken before it. A result without a
bitstream is not re-checkable later.
