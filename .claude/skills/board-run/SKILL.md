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
relink silently ships the OLD initramfs. Prune stale big `.dom` files by **explicit name**
(never a glob — a prefix glob once deleted the package-installed `sbi.dom`).

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
from "board/firmware/boot failed" — and the control itself fails roughly 1 in 5, so a boot
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

## 5. Release the board, always

Lock → power-cycle → run → **power off + unlock in `finally`**. The drivers do this via
`safe_cleanup.release_board`.

* **Never `kill -9` a runner.** It orphans the server-side GDB session in `error`; every
  later run then times out before `load_image`, and it survives a power cycle. Only
  `gdb_stop()` clears it.
* Signal by **verified PID**, never a bare `pgrep -f <pattern>` — that matches your own
  shell, and once matched an editor with the script open and closed it.
* Confirm `gdb_state` is `idle` before blaming the board.

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
`ugrep`, and on a transcript containing control bytes it printed *nothing at all* for a string
that a byte-level search found 8 times. A silent zero reads exactly like "absent", which is the
worst possible failure mode for a verdict. Same family as the `awk strtonum` returning 0 and the
hex-constant-emitted-in-decimal incidents.

```python
python3 -c "d=open('/tmp/capstone/bootN-raw.txt','rb').read(); print(d.count(b'RESULT'))"
```

Verdicts taken from a driver's own stdout summary are unaffected — the runners parse run-scoped
text with Python regex. This applies to hand-grepping the raw capture.

**Bitstream re-flash is ask-first, always.**

## Reporting a result

State which arms were **reachable**, not just which failed — an entry stall biases *which
constructs can be measured at all*, so "arm X fails and arm Y does not" is unsupportable
unless Y actually entered. Quote the control's verdict alongside every result.

**Name the bitstream in any result you record.** R-14 and R-16 were both fixed by a reflash on
2026-08-04, which invalidated every board measurement taken before it. A result without a
bitstream is not re-checkable later.
