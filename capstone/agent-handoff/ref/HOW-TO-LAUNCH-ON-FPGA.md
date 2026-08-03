# How to launch something on the FPGA (short instruction)

Paste this to an agent (or follow it yourself). The board is a browser/websocket
CVA6+Capstone FPGA — **no SSH**; an agent drives it via the Python driver, a human
can also use the browser GUI. Every step and gotcha is in the KB files below.

**There is a skill for this:** `.claude/skills/board-run/SKILL.md` carries the decision
procedure (bake → order → classify → release) and auto-loads when a task involves running
something on the board. This file remains the full reference behind it.

## One-line instruction to give an agent

> Run `<domain/binary>` on the Capstone FPGA. Read
> `capstone/agent-handoff/ref/HOW-TO-LAUNCH-ON-FPGA.md` first, then the runbook
> `ref/gp-free-silicon-smoke-runbook.md` and KB
> `history/22-07-2026_18-05-00_gp-free-silicon-smoke-firmware-fixed-createdomain-hangs.md`.
> BAKE the program into the buildroot image (never ship it over UART — see
> §"UART TRANSFER IS RETIRED"), build the firmware with the recipe in memory
> `project_fpga_fw_payload_build_recipe`, lock the board, power-cycle, boot, invoke the
> baked program from the shell, harvest the result, then **power off + unlock**. Board is flaky/slow (~2 min to JTAG-load 15 MB); one
> persistent write only (bitstream re-flash) and only if authorized.

**Running SQLite, or any staged probe?** Read
§"Running SQLite (and any staged probe) on the board" below FIRST — it carries the
`dom:selector` mechanism, the mandatory `:0` control, the result-classification table
(an entry stall is not a result), and the ten hazards that cost hours of board time
on 2026-08-02.

## STOP — the five settings that have wasted the most board time

Check these before every run. Each one has produced a confident, wrong conclusion about
the *hardware* when the actual fault was in the harness.

| # | Setting | Wrong value costs |
|---|---|---|
| 1 | `ENTRY_STALL_S` ≥ **260** | JTAG upload is **133–227 s of legitimate silence** (~130 KiB/s). The 45 s default aborted runs mid-upload and produced "the board won't boot" / "cyclic boot" / "firmware is broken" — all false. |
| 2 | Scan only **this run's** output | The console replays ~548 KB of the previous boot on connect. Grepping the whole log finds stale `SHA5`/banner markers. Split at this run's `load_image`. |
| 3 | Unique `PROBE_SCOPED_OUT`, `rm -f` first | The transcript is written only at the end, so a killed run leaves the previous one in place and it reads as current. |
| 4 | `gdb_state` is `idle` before starting | A `kill -9`'d runner orphans the GDB session in `error`; every later run then times out before `load_image`. Survives power-cycle; only `gdb_stop()` clears it. |
| 5 | Firmware identity by **hash** | Buildroot pads in 2 MiB steps, so same-size ≠ same-image. Overlay edits also need `A=linux-rebuild`, not just an OpenSBI relink. |

**Before concluding the board or firmware is broken, rule out the harness.** On 2026-08-02
the large majority of apparent board failures were self-inflicted by rows 1–4; the same
firmware file was declared dead six times and then booted five times with no rebuild in
between.

## The 4 KB files (read in this order)

1. `ref/HOW-TO-LAUNCH-ON-FPGA.md` — this file (the map).
2. `ref/gp-free-silicon-smoke-runbook.md` — step-by-step runbook (build domain +
   controller, boot, run).
3. KB `history/22-07-2026_18-05-00_gp-free-silicon-smoke-firmware-fixed-createdomain-hangs.md`
   — full findings: firmware traps + fast build recipe, the create_domain root cause,
   exact reproduction, board etiquette.
4. Memory `project_fpga_fw_payload_build_recipe` — the exact firmware relink recipe
   (fpga/ariane from caplifive-system, embed FDT+kernel, regen `.c.S`). Plus
   `project_silicon_gp_delivery_boardowner_guidance`, `reference_fpga_rtl_platform`.

**Reproducers to hand over live in `capstone/tests/fpga-repros/`** — one directory per issue,
each with its own README and (where the images are small enough) the frozen `.dom` files pinned
by `SHA256SUMS`. `R14-frame-pad/` is the current R-14 hand-over package and runs entirely
through the baked-image path below.

## Tooling (already on disk)

- Driver + venv: `/tmp/capstone/fpga-venv/bin/python`, drivers under
  `tests/rtl-smoke/fpga_driver/`. The sanctioned ones boot the firmware and invoke a
  program ALREADY BAKED INTO THE IMAGE (`run_baked_rungs_fpga.py`,
  `run_sqlite_stages_fpga.py`); the UART-transferring ones are deprecated — see
  §"UART TRANSFER IS RETIRED". `board_reflash_only.py` (re-flash only). Protocol: `tests/rtl-smoke/socketio-api.md`,
  `tests/rtl-smoke/fpga_driver/`.
- Board URL token: **`FPGA_URL` environment variable, for the duration of ONE run.**
  The console URL embeds the access token in its path, so it is a credential: never
  commit it, never echo it into a capture, and **never persist it to disk** -- writing
  it to a dotfile (an older revision of this file suggested
  `~/.config/capstone/fpga-board-url`) is still a leak. Ask the user out-of-band each
  time; in committed text write `<FPGA-CONSOLE-URL>`.
- A local `.bit` is NOT needed — re-flash names the **server-side** bitstream
  `working-caplifive-captype-fixed.bit`.

## UART TRANSFER IS RETIRED — BAKE EVERY PROGRAM INTO THE IMAGE

**Standing rule (2026-08-03): do not ship a program to the board over the UART console.**
Put it in the buildroot image and let it ride the firmware over JTAG, which happens anyway.
The transfer tiers below this line are kept only as history so nobody rebuilds them.

**Why.** UART delivery moves 16 characters per socket.io emit, and each emit is an HTTPS
round trip — the network, not the UART, is the wall clock. Measured 2026-08-03: shipping a
~10 KB domain took **minutes**, while the same set BAKED INTO THE IMAGE ran **10 domains in
ONE boot in ~5 minutes**, because a 10 KB domain is free inside a JTAG upload that is already
happening. The board owner's own answer, when asked how to deliver a binary, was in substance
*"isn't it built into the buildroot image and loaded through JTAG? Why do we need UART?"*

### The one workflow

```bash
O=capstone/caplifive-system/sw/buildroot/overlay/test-domains
T=capstone/caplifive-system/sw/buildroot/build/target/test-domains
cp -f <artifact> "$O/" && cp -f <artifact> "$T/"      # BOTH dirs -- buildroot packs $T
cd capstone/caplifive-system/sw/buildroot
make build LINUX_PAYLOAD=1 A=linux-rebuild  CAPSTONE_CC_PATH="$(realpath ../../../capstone-c)"
make build LINUX_PAYLOAD=1 A=opensbi-rebuild CAPSTONE_CC_PATH="$(realpath ../../../capstone-c)"
```

`A=linux-rebuild` FIRST: buildroot does not track `overlay/` -> cpio, so an OpenSBI-only
relink silently ships the OLD initramfs. Then run with a driver that invokes the baked
artifact from the shell:

| What you are running | Driver | Invocation on the board |
|---|---|---|
| silicon-ladder rungs | `fpga_driver/run_baked_rungs_fpga.py` | `/test-domains/lpc <rung> /test-domains/<rung>.dom` |
| SQLite / staged probes | `fpga_driver/run_sqlite_stages_fpga.py` | `/test-domains/sqlite_host.user <dom> [selector]` |

```bash
export FPGA_URL="$(cat ~/.claude-c/secrets/fpga-console-url)"   # credential, never commit
export FPGA_FW=.../opensbi-custom/build/platform/fpga/ariane/firmware/fw_payload.bin
BAKED_RUNGS="clp1 clp8 r14sl" python3 -m fpga_driver.run_baked_rungs_fpga
```

### THE DRIVER DOES NOT REBOOT BETWEEN PROGRAMS

A wedged program takes the core, so **every program after the first failure is collateral,
not a result.** This produced a wrong verdict within an hour of the baked path existing:
`e3rd` failed at position 4 of a 6-program boot, and `e4wr` and `r14lp` were recorded as
failures; re-tested one-unknown-per-boot, **`e4wr` passes**. Rules:

* at most **ONE unknown per boot**, placed **last**, after a known-good control;
* read a run **no further than its first failure**;
* everything expected to return goes first, in ascending order.

### Retired: the UART transfer tiers (do not rebuild these)

`fast_xfer.py` (`fast_put`), the `burst(16) -> fast -> safe -> safest` escalation and the
`gzip+base64` + per-chunk-sha protocol were a real ~3x win over one-emit-per-character, and
the 16-char burst is the ns16550a FIFO depth so it was correctly derived. None of that matters
now: baking removes the transfer entirely. `run_ladder_perf_fpga.py`, `run_sqlite_fpga.py` and
`run_ladder_base_fpga.py` still transfer and are **deprecated** — each now carries a header
saying so. Keep `fast_xfer.py` on disk only because those legacy drivers import it.

One thing from that era that is NOT retired, because it is about the board and not the
transport: **a second domain reused at the same entry VA (`0x10000`) within one boot can
silently hang its `cscall`** (R-3, no icache invalidate on domain switch). Baking does not fix
it. Either link rungs at distinct `DOMAIN_BASE_VA`s, or accept one program per boot — and
never enable a one-boot sweep across same-VA domains without saying so in the write-up.

## Rebuilding the monitor — SOLVED 2026-07-28, but delete the stale objects first

The monitor **can** be rebuilt now. Every rebuild used to boot-hang with zero serial; the
cause was never the compiler. `fw_jump.o` in the shared build dir had been compiled for the
**FPGA** firmware, where embedding a device tree is mandatory, and `A=opensbi-rebuild` only
relinks — so the QEMU monitor inherited an FPGA DTB and `fw_base.S`'s
`#ifdef FW_FDT_PATH → lla a1, fw_fdt_bin` discarded the DTB QEMU passes. Wrong UART, no
console, no banner.

```bash
cd capstone/caplifive-buildroot
D=build/build/opensbi-custom/build/platform/generic/firmware
rm -f $D/fw_jump.o $D/fw_jump.elf $D/fw_jump.bin $D/fw_dynamic.o $D/fw_payload.o   # REQUIRED
make build A=opensbi-rebuild CAPSTONE_CC_PATH="$(realpath ../capstone-c)"
readelf -sW build/images/fw_jump.elf | grep -c fw_fdt_bin      # MUST be 0
readelf -SW build/images/fw_jump.elf | grep '\.rodata'         # MUST be 002de8, not 003a10
```

**This re-arms every time the FPGA firmware is built in this tree** — same build dir, and
the FPGA side genuinely needs `FW_FDT_PATH`. Treat the `rm` as part of the recipe.
Restore point if anything goes wrong: `~/capstone-b-artifacts/monitor-known-good/`
(`fw_jump.elf.good` md5 `6724bcb3`, plus the known-good `.c.S`). Trail: issue C-11.

## Superseded: "don't rebuild the monitor with our tree's `capstone-c`"

Regenerating `fw_jump.elf` (QEMU) or the FPGA firmware monitor from **our tree's**
`capstone-c` (`master`@`8cda52c`) **boot-hangs** (zero serial). **Known fix
(2026-07-25):** the working firmware is built by `caplifive-system`'s pinned
`capstone-c` = branch **`bugfix`@`508342a`** (divergent from our `master`; carries
a gct-alignment fix). So for any monitor rebuild, build with **that** compiler, not
our tree's. Recovering + pinning this is `plans/monitor-regen-audit-task-B.md` (fast
path); until it's pinned in-tree, **use the existing working prebuilt as-is** for
board runs. Memory `project_opensbi_monitor_rebuild_include_wrapper`.

## Large domains go in the BUILDROOT IMAGE over JTAG, not over UART (2026-07-28)

> **Superseded 2026-08-03 by §"UART TRANSFER IS RETIRED": this now applies to EVERY program,
> not just large ones.** Kept for the board owner's rationale below.

For anything bigger than a few tens of KB, build it into the FPGA buildroot image and
let it ride the firmware over JTAG. The board owner's answer when asked how to deliver
a large binary was, in substance, *"isn't it built into the buildroot image and loaded
through JTAG? Why do we need UART?"* -- and that is right.

    capstone/caplifive-system/sw/buildroot/overlay/test-domains/   <- put artifacts here
    cd capstone/caplifive-system/sw/buildroot
    make setup
    make build CAPSTONE_CC_PATH=$(realpath ../capstone-c)

The FPGA rootfs is a CPIO **initramfs compiled into the kernel**
(`fpga_defconfig`: `BR2_TARGET_ROOTFS_CPIO`, `BR2_TARGET_ROOTFS_INITRAMFS`;
`fpgakernel.config`: `CONFIG_INITRAMFS_SOURCE=.../rootfs.cpio`), so anything in the
overlay is present at boot with no transfer step at all. Note this is
**caplifive-system's** overlay -- caplifive-buildroot has a separate one that feeds the
QEMU image.

### Why UART is the wrong answer, and two mistakes made getting there

**Do not size a UART transfer from the raw file.** `fast_put` gzips first, so the
transfer is base64 of the COMPRESSED size. Sizing SQLite from its raw 2.27 MB gave
">= 63 min" and led to ruling UART out; the real figure was ~15 min (1.62 MB -> 529 KB
gzipped -> 705 K base64 chars at ~800 chars/s).

**But ~15 min was also wrong, in the other direction.** It assumes zero retries. At
703 K chars the transfer is 1,759 chunks, a dropped character is near-certain, and
`_put_once` **truncates and restarts the whole file** on any failure -- then falls back
to burst=1, which for that size is ~4 hours. Measured: the 529 KB domain wedged the
shell partway through and began exactly that fallback. So UART's limit at this scale is
**reliability, not throughput**, and no amount of pacing fixes a whole-file restart.

If UART ever has to carry something this big, make `fast_put` verify and retry
PER CHUNK instead of per file. Until then: use the image.

## Two things that LOOK like a hang and are not (2026-07-28)

Both cost time in one session; check them before debugging anything.

1. **The board powering off at the end of a run is CORRECT.** `powered off` / `unlocked`
   in the log is the runner's `finally`, and the process exits 0. It is not a crash.
2. **A reboot banner in the UART right after a rung's `BG<rung>` marker is usually the
   NEXT rung's power-cycle, not a wedge.** `Hello World! ... booting! ... OpenSBI v1.3`
   appearing after `domain ID = 0` reads like the domain took the board down; it is the
   runner power-cycling because that rung produced no END marker within 120 s. The actual
   failure is a hang (`cscall` never returns). To tell them apart, look at whether a
   `power-cycle + reload firmware` line precedes the banner.

**And one real hazard: an ad-hoc console script that never calls `disconnect()` stays
alive forever.** The socket.io client thread is non-daemon, so a script that connects,
does its job and falls off the end keeps running and holds a console session — it shows
up as an inflated `users connected:` count and looks like someone else is on the board.
One such script lived 49 minutes. Always `disconnect()`, or run it under `timeout`.

## The board-driver contract (2026-07-31) — read before waiting on ANY run

Three drivers follow it: `run_sqlite_baked_fpga.py`, `probe_sqlite_wedge.py`,
`probe_revnode.py`. All of it exists because "is the board still busy?" was answered
wrongly, repeatedly, and each wrong answer cost either board time or an idle session.

**Required environment.** Both, or the driver dies before touching the board:

| Var | What | Failure if unset |
|---|---|---|
| `FPGA_URL` | console URL **with token** — a credential | `FPGA_URL not set` |
| `FPGA_FW`  | absolute path to `fw_payload.bin` | `KeyError: 'FPGA_FW'` at import |

`FPGA_FW` has no default on purpose — an implicit one silently flashes whatever was built
last. The FPGA firmware is **not** in `build/images/`; it is
`caplifive-system/sw/buildroot/build/build/opensbi-custom/build/platform/fpga/ariane/firmware/fw_payload.bin`
(~17.4 MB). `build/images/` holds only `fw_jump.bin` (~569 KB), which is the QEMU monitor
and will not boot the board. Getting this wrong throws at *import*, i.e. before the run —
so a stale `sqlite-run-scoped.txt` from an earlier session is still sitting on disk and
reads exactly like a fresh result. **Always confirm the driver actually ran before reading
its output file.**

**Completion is signalled, never inferred.** Every driver prints, in this order:

```
RUN_DONE | PROBE_DONE     <- first statement in finally; survives a throwing teardown
BOARD_RELEASED            <- switches -> power off -> unlock -> disconnect, each time-boxed
```

Poll for those strings. **Never poll with `pgrep -f <pattern>`** — the polling loop's own
command line contains the pattern, so it matches itself and spins forever. Six such loops
ran here for up to 21 hours. Bound every wait with `for i in $(seq 1 N); do ...; done`.

**Drivers exit via `hard_exit()`, not `sys.exit()`.** `sys.exit` unwinds the main thread
and then *waits on every non-daemon thread*; socketio's survives `disconnect()` often
enough that it cannot be relied on. Measured 2026-07-31: a run printed `RUN_DONE` and
`BOARD_RELEASED` and then stayed alive emitting `user_count` events — the board was
genuinely free, but from outside it was indistinguishable from a session still in
progress, and it was reported as phantom board activity. `hard_exit` flushes the streams
and calls `os._exit`, preserving the exit code.

**Teardown is time-boxed and ordered least-important-first** (`safe_cleanup.py`): switches,
then power, then unlock, then disconnect. Each step waits on a board event and can block
forever while holding the flock — which strands every session queued behind it. A step
that will not complete is abandoned and logged rather than waited on, because releasing
the board matters more than tidying it. Precedent: `probe_revnode.py` once sat 16 minutes
inside `set_switches(console, 0)` with the board powered and the lock held, after all four
of its LED reads were already on disk.

## Running SQLite (and any staged probe) on the board — 2026-08-02

SQLite does not run as a one-shot binary. It runs as a **domain** loaded by a host
loader, and it is investigated through **staged probes**: builds that execute the first
N steps of `run_sqlite()` and RETURN a marker instead of running to the failure. Use
this path for SQLite and for anything else that needs more than a single pass/fail.

### The runner and its inputs

```bash
export FPGA_URL="$(cat ~/.claude-c/secrets/fpga-console-url)"   # credential, never commit
export FPGA_FW=.../opensbi-custom/build/platform/fpga/ariane/firmware/fw_payload.bin
export SQLITE_STAGE_DOMS="/test-domains/f10.dom:0,/test-domains/f10.dom:9,/test-domains/f10.dom:10"
export SQLITE_STAGE_TIMEOUT=200
export PROBE_SCOPED_OUT=/tmp/capstone/scoped-$(date +%H%M%S).txt   # UNIQUE per run, see below
python3 capstone/tests/rtl-smoke/fpga_driver/run_sqlite_stages_fpga.py
```

Domains run **in the listed order, and the runner stops at the first one that does not
return** — a wedged domain takes the core with it, so everything after it is lost. Put
ascending stages in order, controls first, and at most ONE expected-to-wedge domain last.

### `dom:selector` — one image, many probes

An entry may carry an optional `:selector`, passed to the host as `argv[2]`, published in
the shared region's `opcode` field, and read by the domain (magic-guarded `0x5A6E00nn`,
`sqlite_capstone_domain.c`) to choose its probe **at run time**:

    /test-domains/mech.dom:0,/test-domains/mech.dom:129,/test-domains/mech.dom:128

Without a suffix the domain uses its compile-time `CAPSTONE_SQLITE_STAGE`, so every
existing invocation is unchanged. Only stages compiled INTO that image can be selected —
the `#if` ranges still gate what exists, so a selector outside the compiled range silently
falls through into the real SQLite path and measures the wrong thing. Group probes that
share one `#if` block.

Why it matters: one binary per probe means every measurement is a fresh roll of the
build/boot dice. Run-time selection lets several questions ride one image and one boot.

### Every test names itself on the UART

Because one boot runs several domains back to back, the runner echoes a banner **on the
board** around each one, so it appears live in the console GUI and inside the transcript:

```
### TEST 3/5 START /test-domains/min.dom:146 ###
SQ: A/dom-ok ... SQ: obs=1516338
### TEST 3/5 END /test-domains/min.dom:146 rc=0 ###
```

and locally, `--> TEST 3/5  <label>` / `<-- TEST 3/5  <label>  returned in 12s`, or
`NO RETURN within 150s -- everything after this is lost`. Without this, a wedge mid-sequence
had to be attributed by counting `SQ:` markers by hand, and the run that stalled was
routinely mis-identified.

The prefix is `###` and **must not be `SQ: `**: the missing-domain guard tests
`"SQ: " not in text`, so banners carrying that string would make every run look like it
produced domain output and would silently disable the guard.

### Retrying: REDRAW the image, do not re-run it

An entry stall wedges the boot, so a losing draw costs the whole attempt — which makes a
retry loop look attractive. But R-16 is **per-image and deterministic**: `min.dom` stalled
2/2 under a correctly-calibrated watchdog, and `sb10` 3/3, `x101` 6/6, `r112` 3/3 before it.
Re-running the same binary is N identical losing tickets and pure board time.

So a retry loop must **rebuild or switch image between attempts**. Stage several distinct
images that carry the same probes and walk them; a bounded 2 attempts is right for genuine
*infra* flakes (JTAG/upload), which are transient, and more than that only pays if each
attempt draws a new binary. Cheap way to draw: vary `CAPSTONE_SQLITE_STAGE` within a single
`#if` block, so every image carries the whole ladder and the probe code under test stays
byte-identical across draws — the draw then cannot confound the result. Always `sha256sum`
the set and abort if any two match.

### ALWAYS run `:0` first. A verdict from an image whose control did not return is noise.

`run_sqlite_staged()` begins `if (stage <= 0) return 0;`, and that early return is **not
inside any `#if`** — so **selector `:0` is live in every staged image ever built**, at zero
build cost.

    :0 RETURNS  -> the image's entry, glue, reentry, marker write and return path all work,
                   so a later wedge in that image belongs to the CONSTRUCT under test.
    :0 WEDGES   -> the image is unsound; every other verdict from it is void.

Both outcomes were observed on 2026-08-02. On `f10`, `:0` and `:9` returned and `:10`
wedged — that is what makes "the blocker is `sqlite3RegisterBuiltinFunctions`" a result.
On `n112`, `:0` itself wedged, and five earlier "variant D wedges" readings from that image
had to be withdrawn.

### Classify before recording: three things are NOT results

Read the **last marker** of the domain's block, never just "did not return":

| Last marker | Meaning | What to do |
|---|---|---|
| `SQ: obs=<n>` | returned a value | record it |
| `SQ: G/enter` then silence | entered, wedged in its own code | record it — a real result |
| `SHA5:xxxx` (no `SHA6`) | **entry stall** — monitor handed off, domain never ran | retry; carries no information |
| no `Ok, good file.` / JTAG errors | **infra** — board never ran the image | retry |
| `__CAPSTONE_INFRA_FLAKE__` | QEMU/boot flake | retry |

`SHA5` = "about to leave M-mode for the domain", `SHA6` = "the domain returned"
(`sbi_capstone.c`). The runner's own "FIRST FAILURE" summary collapses the first three
rows into "did not return" — the distinction has to be made when reading the scoped log.
**A failure that happened before the thing under test began is not evidence about the
thing under test.**

### Supervise every board run with the watchdog, and wait with `wait-for.sh`

A wedged domain emits nothing and a dead runner writes nothing; from outside both look
exactly like healthy progress. Run these as a second process:

```bash
python3 .../run_sqlite_stages_fpga.py > "$LOG" 2>&1 &
R=$!
ABORT_ON_ENTRY_STALL=1 ENTRY_STALL_S=260 \
  bash capstone/tests/rtl-smoke/board-watchdog.sh "$LOG" 300 "$R" &
wait $R
```

**Calibrate `ENTRY_STALL_S` against the JTAG upload, not against a guess.** This is the
single most expensive mis-setting in the harness's history. Measured 2026-08-02: the JTAG
transfer runs at **~130 KiB/s**, so a 17.4 MB firmware is **~133 s** of complete UART
silence and larger images have taken **227 s** — all of it perfectly healthy. The default
was **45 s**. The watchdog therefore aborted essentially every run mid-upload, and the
resulting "the board will not boot" / "cyclic boot" / "firmware is broken" diagnoses were
all harness artefacts: the same firmware file was declared dead six times and then booted
five times in a row with no rebuild in between. **Use ≥ 260 s.** An earlier version of this
line said "healthy boots have gone 120 s quiet" — that figure predated the upload
measurement and is an underestimate; do not restore it.

`board-watchdog.sh <uart-log> [idle-limit-s] [runner-pid]` emits one line per interval:
`ALIVE +<bytes>` / `QUIET <idle>s` / `STALE` / `ENTRY-STALL` (aborts the runner) / `GONE`
(runner died) / `ENDED`. Liveness is `kill -0 <pid>` — **never `pgrep -f <pattern>`**,
which matches the watchdog's own command line and reports the runner alive forever.

`wait-for.sh <file> <sentinel> <pid> [timeout]` returns on the sentinel **or the producer
dying**: exit 0 = sentinel, 3 = producer died without it, 4 = timeout. Waiting on a
sentinel alone hangs forever if the producer crashes.

### Hazards that cost real board time on 2026-08-02 — check these first

1. **The console REPLAYS the previous boot's scrollback (~548 KB) on connect.** Any
   grep over a whole board log finds markers from an EARLIER run. Split at the run
   boundary — the last `load_image`, or the runner's `booted once`. A watchdog that
   grepped the whole log matched a stale `SHA5` and killed ~24 healthy runs over ~87
   minutes right after `load_image`, before the board had even booted, and produced the
   false conclusion "the board stopped accepting images". Scope the scan to **this run's
   own `load_image` emit**, not to a byte offset captured at startup: callers `rm -f "$LOG"`
   before launching, so any "remember the starting size" fix is inert — it measured ~1 byte
   every time and silently did nothing for an entire session while appearing to be the fix.
   The working form is

   ```bash
   scan=$(awk '/emit gdb_input .*monitor load_image/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' "$LOG")
   ```

   and `NO-BOOT` keys on `buildroot login` / `SQ: `, never on the bootrom banner (the banner
   appears in replayed scrollback: 45 copies before `load_image`, 0 after — which is what
   was misread as "cyclic boot").
2. **Never reuse `PROBE_SCOPED_OUT` between runs, and delete it first.** The runner writes
   the transcript only at the end, so a killed run leaves the PREVIOUS run's file in place
   and the classifier reports it as current. A 5-hour-old result about a different domain
   image was once reported as a fresh measurement. Stamp the filename and `rm -f` it.
   Cross-check the transcript's `===== <path> =====` header against the image you ran.
3. **Pruning the overlay does NOT shrink the initramfs.** Buildroot packs
   `build/target/test-domains/`, not `overlay/test-domains/`. Remove stale `.dom` files
   from the target dir too — **by explicit name**, never a glob: a prefix glob once
   deleted the package-installed `sbi.dom`. The image grew 15.4 MB -> 30.0 MB across one
   session because dead probe images accumulated there.
4. **Never edit a script while it is running.** Bash reads scripts incrementally, so the
   edit corrupts the tail it has not yet reached; a run died on a syntax error before
   printing its sentinel and stranded a waiter for ~9 minutes. Copy it, edit the copy.
5. **Verify perturbed builds are actually different.** Passing an unused `-DFOO=n` through
   `DOMAIN_EXTRA_DEFS` produces byte-identical binaries. Three "independent draws" were one
   binary counted three times. `sha256sum` the outputs and abort if any two match — note
   the "distinct hashes" line printed by `build-stage-probes.sh` counts every file in the
   destination dir, not the stages just built, and no caller checks it.
6. **Domains are big; keep only what the current experiment needs.** Every `.dom` left in
   the target dir rides the firmware over JTAG on every boot.
7. **Never `kill -9` a board runner.** It bypasses the `finally` that calls
   `release_board()`, which leaves the **server-side GDB/OpenOCD session orphaned in
   `error`**. `gdb_start()` no-ops unless the state is `idle|error` and only waits for
   `running`, so every subsequent run then dies with
   `ActionTimeout: timed out waiting for event 'gdb_state'` and never issues `load_image` —
   indistinguishable from "the board is broken", and it survives a power cycle *and* a lock
   release. Only `gdb_stop()` clears it (`/tmp/capstone/gdb-recover.py`). Stop runners with
   `TERM` and let teardown run; check `gdb_state` is `idle` before blaming the board.
8. **Firmware identity is the hash, never the size.** Buildroot pads images in **2 MiB**
   steps, so a rebuilt firmware routinely has a byte-identical *size* and completely
   different contents — two builds compared here matched at 17466376 bytes and differed in
   3.8 M bytes (`md5 e1a17f74…` vs `79084b88…`). "Same size, so it's the known-good image"
   is not a check. `md5sum`/`sha256sum` it.
9. **Overlay edits need `A=linux-rebuild`.** Buildroot does not track
   `overlay/test-domains/` → cpio as a dependency, so staging or pruning a `.dom` and then
   rebuilding only OpenSBI produces firmware whose initramfs is unchanged. Symptom: the
   image size does not move no matter what you delete, and the board keeps running the
   *old* domain while you attribute its results to the new one. Order is
   `A=linux-rebuild` → `A=opensbi-rebuild`.
10. **Before concluding the board or firmware is broken, rule out the harness.** Of the
    apparent board failures on 2026-08-02, the large majority were self-inflicted (items
    1, 4, 7 above). The discriminator that settled it was blunt and worth reusing: the
    *same firmware file* was "dead" six times and then booted five times with no rebuild
    between — so the variable was not the firmware. Diff what actually changed in the
    harness before spending a rebuild or a reflash.

### Full SQLite (not staged)

`run_sqlite_baked_fpga.py` runs the real domain and gates on the five success markers in
the run-scoped file. Give it room: silence between `SQ: G/enter` and the first row is
legitimate while it opens the database and runs CREATE/INSERT, and a run was once aborted
on exactly that stretch and read as a wedge. Set `SQLITE_RUN_IDLE` well above the default
75 s (600 s is reasonable) and `SQLITE_RUN_TIMEOUT` to match.

## Non-negotiables

Lock → power-cycle → run → **power off + unlock in `finally`** (never leave it
locked/on). Verify the resident bitstream is `working-caplifive-captype-fixed.bit`
before measuring. `C_PRINT` (`csrw 0x800`) goes to the **RTL trace**, not the UART —
don't use it as a UART probe. Signal of a live domain = the controller prints its
first line (that's AFTER `IOCTL_DOM_CREATE` returns).
