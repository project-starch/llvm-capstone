# FPGA borrow/revoke-cost reproduction runbook (for Agent A)

**Goal:** reproduce, end to end, the on-board CapliFive CVA6 run that gets the
Capstone domain benchmark past the old glibc/FP hang and up to the domain `cscall`,
so the domain-call fault can be diagnosed and (eventually) the cycle numbers captured.

This is the exact path Agent B walked (tasks 016–018 + the 2026-07-20 freestanding
controller). Read it top to bottom once before running anything. Companion docs:
- `history/19-07-2026_19-55-15_fpga-mode-build-run.md` (the `--mode fpga` UP image + the fsd diagnosis)
- `history/20-07-2026_*_fpga-freestanding-controller-domain-call-reached.md` (the freestanding fix + domain-call finding — the state report)
- `history/19-07-2026_09-30-00_captype-fixed-flash-loadfault-mcause.md` (bitstream flash + power-cycle rules)
- memory `fpga-benchmark-must-be-freestanding`, `fpga-bitstream-flash-and-pairing`, `fpga-up-image-vermagic`

---

## 0. Hard rules (do not skip)

- **The FPGA token is secret.** It arrives as a URL `https://fpga.corank.info/<token>/`.
  Never commit it, never write it into a file under the repo, never echo it into a
  captured log. Put it in an env var only (`export FPGA_URL=...`) for the duration of a run.
- **Non-persistent board use only.** Boot via JTAG/gdb (`load_image`), never rely on the
  board's resident firmware. **A bitstream flash is the ONLY persistent write** and is a
  HARD STOP-and-ask — volatile *or* non-volatile — because we cannot rebuild a bitstream
  here. The one exception already exercised is re-flashing Jason's named
  `working-caplifive-captype-fixed.bit` to undo another team overwriting the board (see §5).
- **Lock the board** before driving it, **release + power off** in a `finally` on every run
  (good citizen; the user authorized ignoring other users but not leaving it powered/locked).
- Commits go on `capstone-bootstrap-b` only; no submodule-source commits; no `Co-Authored-By:`.
- Manager/collaborator-facing notes under `/tmp/capstone/`, not the repo.

---

## 1. Why this is hard (the two blockers, in order)

1. **The stock benchmark `.user` controller is a glibc Linux program, and glibc emits
   `fsd` (double-precision FP store).** This `captype-fixed` bitstream's FPU **rejects
   `fsd` even with `mstatus.FS=Clean`** (JTAG-proven: mcause=2 illegal, mepc in userspace,
   insn=`fsd`, FS=Clean). The first `printf` traps → the monitor `while(1)`s → silent hang.
   **Fix = a freestanding soft-float controller (`borrow_cost_fpga_ctl.c`)** that links no
   glibc and emits zero FP. This is *proven working* — it boots, creates the domain, and
   maps both regions on real silicon.
2. **With (1) fixed, the domain `cscall` is finally reached — and the domain wedges at its
   own entry (vaddr `0x10044`, the `<test>` glue right after `_start`).** Sometimes the core
   resets to the bootrom (banner), sometimes it sits spinning at `0x10044`. **This is the
   open blocker** and the thing Stage-0 instrumentation (§7) is meant to diagnose.

You must clear (1) to even observe (2).

---

## 2. One-time environment

```bash
cd /home/alexey/dev/llvm-capstone-b        # (Agent A: your clone)
source capstone/tests/capstone-test-env.sh
```

- Host compiler must stay `/usr/bin/clang++` — never a capstone-built clang (memory
  `llvm-build-constraints`). Cap ninja at ~70–80% of cores.
- The board driver lives at `capstone/tests/rtl-smoke/fpga_driver/` and is already wired to
  the real (verified) hybrid HTTP+Socket.IO protocol. `run_rtl_smoke.py` is the entry point;
  the ad-hoc run scripts B used (keepalive capture, gdb-probe) are in the session scratchpad
  and are reproduced inline in §6–§7 because scratchpad is not committed.

---

## 3. Build the freestanding controller + the domain `.dom`

```bash
bash capstone/tests/rtl-smoke/build-borrow-cost-fpga.sh
# Produces in $CAPSTONE_TMP_ROOT/capstone-rtl-smoke/ :
#   borrow_cost_fpga_ctl   <- the freestanding soft-float controller (THE one that runs)
#   borrow_cost_fpga.dom   <- the Capstone-clang domain payload
#   borrow_cost_fpga.user  <- the OLD glibc controller (kept for QEMU / D-capable cores; HANGS on-board)
```

Sanity-check the controller emits **zero** FP and is static/no-PIE:

```bash
BR=$CAPSTONE_BUILDROOT_DIR
$BR/build/host/bin/riscv64-buildroot-linux-gnu-objdump -d \
  $CAPSTONE_TMP_ROOT/capstone-rtl-smoke/borrow_cost_fpga_ctl | grep -cE '\bf(sd|sw|ld|lw|add|mul|div)\b'
# expect: 0
file $CAPSTONE_TMP_ROOT/capstone-rtl-smoke/borrow_cost_fpga_ctl   # expect: statically linked, no interpreter
```

Key build flags (in the script; do not drop any): `-Os -static -no-pie -fno-pie -nostdlib
-ffreestanding -fno-stack-protector -march=rv64imac -mabi=lp64`. The `_start` must init
**both** `sp` and `gp` (`lla gp, __global_pointer$` under `.option norelax`); forgetting
`gp` makes every global store SIGSEGV (cause 0xf) — this bit us once.

---

## 4. Build the `--mode fpga` **UP (SMP=n)** image with the controller baked in

This is the caplifive-system "official" software build; it produces the OpenSBI FW_PAYLOAD
with the kernel + initramfs embedded and `caplifive.dtb` baked in (so boot needs only
`--image`). Must be **UP / `CONFIG_SMP=n`** — the SMP kernel floods the console with
`remote fence ... not available in SBI v1.0` (2000+ lines) and buries the login prompt.

Prereqs: `caplifive-system` software submodules initialised (`caplifive-system` →
`sw/buildroot`, `sw/capstone-c`, nested `buildroot`, `components/opensbi`, and
`capstone-sbi` @ the `99aaffa8` genesys-testing reference). Skip the heavy RTL/Vivado/anvil
submodules.

Container toolchain (Podman not installed here; Docker is — add a `podman`→`docker` shim on
PATH so the caplifive scripts run unmodified):

```bash
# one-time: build the container image
cd capstone/caplifive-system
scripts/build-image.sh          # -> caplifive-build:latest  (Ubuntu 22.04 + rust + opam/OCaml 5.2.0)
```

Build inside the container (bind-mount the already-checked-out tree; skip setup.sh's
recursive submodule pull). The three gotchas below are the whole reason this is a runbook:

```bash
docker run --rm -v $PWD:/workspace -w /workspace caplifive-build:latest bash -c '
  set -e
  # A) build once (kernel config must be SMP=n; if the defconfig is SMP=y, disable it
  #    in the kernel fragment and rebuild the kernel — see history 19-07 19:55 note)
  make build
  # B) GOTCHA 1: `make build LINUX_PAYLOAD=1` does NOT re-trigger the OpenSBI link,
  #    so you get a 2.1 MB payload with NO kernel. Force the relink explicitly:
  make -C build/build/opensbi-custom PLATFORM=fpga/ariane \
       CROSS_COMPILE=$(pwd)/build/build/host/bin/riscv64-buildroot-linux-gnu- LINUX_PAYLOAD=1
'
```

After changing the rootfs overlay (next step) you MUST force the initramfs to re-embed:

```bash
# C) GOTCHA 2: buildroot does not track the cpio dependency; force it:
docker run --rm -v $PWD:/workspace -w /workspace caplifive-build:latest bash -c '
  make build A=linux-rebuild        # re-embeds rootfs.cpio into the kernel Image
  make -C build/build/opensbi-custom PLATFORM=fpga/ariane \
       CROSS_COMPILE=$(pwd)/build/build/host/bin/riscv64-buildroot-linux-gnu- LINUX_PAYLOAD=1
'
```

A correct payload is ~15.3 MB (kernel at 0x200000 embedded). A 2.1 MB payload = you hit
gotcha 1.

### Stage the controller + domain into the rootfs overlay (before the re-embed above)

```bash
DST=capstone/caplifive-system/sw/buildroot/overlay/root/rtl-smoke
mkdir -p $DST
cp $CAPSTONE_TMP_ROOT/capstone-rtl-smoke/borrow_cost_fpga_ctl $DST/
cp $CAPSTONE_TMP_ROOT/capstone-rtl-smoke/borrow_cost_fpga.dom $DST/
# (also the revoke_cost_fpga_*.dom for the full sweep)
```

Copy the finished payload out to the artifacts dir (kept out of the repo):

```bash
cp .../build/.../fw_payload.bin ~/capstone-b-artifacts/fw_payload_fpga_up_ctl.bin
sha256sum ~/capstone-b-artifacts/fw_payload_fpga_up_ctl.bin   # B's good build: fe37ebdb...
```

`capstone.ko` must match the **UP** vermagic (rebuilt alongside the UP kernel) or `insmod`
fails; the overlay ships it at `/capstone.ko` (memory `fpga-up-image-vermagic`).

---

## 5. Put the correct bitstream on the board (flash + power-cycle)

Only if the board's resident NV bitstream is NOT `working-caplifive-captype-fixed.bit`
(the other team has overwritten it before, e.g. with stock `ariane_xilinx.bit` — which has
no capability unit and resets on any `cscall`; ALL evidence gathered on that is garbage).
Check first:

```bash
# the driver exposes GET /api/bitstreams + flash_state; nv_bitstream_name tells you what's resident
```

If it must be re-flashed, this is the one allowed persistent write (Jason's file, restoring
the intended config). Two rules learned the hard way:
- **Power on + settle BEFORE flashing** (a cold board's JTAG programmer isn't up → `flash_state=error`, no SPI write).
- **Power-cycle AFTER flashing** (`--power-cycle`); a non-volatile flash only writes SPI, the
  FPGA keeps running the old config until it reconfigures at power-on. Skip this and the DTM
  comes up degenerate (IDCODE 0x00000001, `load_image` fails "waiting for busy to go low").

---

## 6. Boot + run (the freestanding controller) with keepalive capture

The lab websocket idle-drops ~60 s into a passive output wait, so background the controller
to a file and send a keepalive during the wait. Skeleton (full version = scratchpad
`run_ctl_image7.py`; the driver primitives it uses are stable):

```python
# export FPGA_URL=https://fpga.corank.info/<token>/   (never commit this)
IMG = "~/capstone-b-artifacts/fw_payload_fpga_up_ctl.bin"
CTL = "/root/rtl-smoke/borrow_cost_fpga_ctl"
DOM = "/root/rtl-smoke/borrow_cost_fpga.dom"
# 1. lock; power(False); power(True); settle
# 2. gdb_start; monitor reset halt
# 3. load_image IMG @0x80000000 bin   (~2 min, 15 MB); DTB baked in, no separate load needed
#    (if using a non-baked DTB: load caplifive.dtb @0x82200000 and set $a1)
# 4. set $pc=0x80000000; $a0=0; continue
# 5. poll for "login:"  (keepalive: send '\r' every ~12 s so the socket stays alive)
# 6. root; quiet the console:  echo 1 > /proc/sys/kernel/printk
# 7. insmod /capstone.ko ; test -e /dev/capstone
# 8. run in BACKGROUND to a file:  ( CTL DOM; echo CTLEXIT=$? ) >/root/out.txt 2>&1 &
# 9. keepalive-poll out.txt for "measurement complete" / "CTLEXIT=" (up to ~240 s)
# 10. finally: gdb_stop; power(False); unlock; close
```

**Expected today:** boots → shell → `insmod` OK → controller prints `created domain ID = 0`
→ `create_region`/`map_region` OK for both regions → then **hangs at the domain `cscall`**
(bootrom banner, or a silent spin). You will NOT get `RESULT` cycle lines yet — that is the
open blocker (§7).

If it hangs, gdb-probe the parked core (non-destructive; scratchpad `run_ctl_image7.py`
tail does this): `monitor halt; p/x $pc; p/x $mcause; p/x $mepc; p/x $mtval`. B saw
`pc=0x819a0044` = domain vaddr `0x10044` (the `<test>` entry glue), `mcause=0` — i.e. the
switch transferred fetch into the domain and it wedged at the first entry instruction. The
CSRs read post-hoc are muddy (overwritten by bootrom execution); use §7 for a clean dump.

---

## 7. Diagnosing the domain-call wedge — CONFIRMED ROOT CAUSE

**The diagnosis is done (2026-07-20); this section records both the method that worked and a
dead end so you don't repeat it.**

**Dead end — the M-mode `mtvec` trap-dumper.** We built a dumper that repoints M-mode `mtvec`
(reset-default = bootrom) to a UART hex-dump handler, injected as raw asm into
`build/build/opensbi-custom/lib/sbi/sbi_capstone_dom.c.S` (the monitor C is pre-compiled to a
checked-in `.c.S`; no capstone clang on PATH to regenerate it; use the `lla` idiom the file
uses — `la` triggers a binutils `elfnn-riscv.c:2358` crash). On real `captype-fixed` the
dumper **stays silent**: an in-domain capability fault routes to the Capstone cap-trap vector
**`ctvec`**, NOT M-mode `mtvec`, so `$mcause` reads 0. (The dumper's `@@MT` output only ever
appeared on the *contaminated stock-Ariane* bitstream, which lacks the cap unit → faults go
to M-mode.) Jason also confirmed `cscall`/`csreturn` **implicitly flush the icache**, killing
the stale-icache / `fence.i` theory. Don't rebuild the mtvec dumper for this.

**Method that worked — gdb single-step + register read at the wedge** (no rebuild; boot the
plain `fw_payload_fpga_up_ctl.bin`, run the controller, `time.sleep(20)` to reach the domain
call, `monitor halt`, then `stepi`/`p/x $gp`; scratchpad `run_singlestep.py`, `run_gpprobe.py`):
- 40× `stepi` from `0x819a0044` never advances → the instruction at domain vaddr `0x10044`
  (`delin gp`) cannot retire; no trap fires.
- `gp = 0x0` (null/untagged) at the wedge; `sp = 0x819c0000` (valid).
- Set pc past the delin (`set $pc = 0x819a0048`) → the domain executes normally.

**Root cause:** `delin gp` with `gp=0` **stalls the CVA6 pipeline** (no retire, no trap).
`gp` is 0 because `start.S` `_start` inits only `sp` (from `cscratch`/cap-CSR `0x4`), never
`gp`, and the monitor's `create_domain` (`sbi_capstone.c:279`) zeroes the domain context's
`gp` slot (`dom_seal[0]=code,[2]=data,[3]=priv`, rest 0). QEMU's `helper_csdelin` asserts a
*tagged* operand and the same `.dom` passes under QEMU, so `gp` arrives valid under QEMU but
`0` on the FPGA. (The board evidence above is correct; the *cause* was mis-attributed —
see the corrected verdict next.)

**Fix = OUR domain runtime, NOT the RTL (corrected 2026-07-20 per Jason).** The
`gp = pc_cap(cursor 0)` line in QEMU's `helper_cscall` (`op_helper.c:1227-1231`) is **our
own non-canonical patch** — commit `7aca0540` ("riscv: unblock native domain capability
calls"), not canonical Capstone. So the RTL is correct to omit it, and the cursor-0
approach isn't representable on silicon anyway. **The canonical reference domains
(`capstone-test-domains`: fib/thread/smode, same capstone-cc compiler) never use `gp`:**
no `delin gp`, no `.capstone_cap_init`, no `cincoffset gp,<abs>` — they address code
pc-relative through the implicit PCC, get the stack cap from `cscratch`, and take data
caps as arguments; `gp` is merely preserved as an opaque caller register. Our
`my_first_domain/start.S` invented the gp machinery for capability-globals, and for
borrow_cost it is pure dead weight (`.capstone_cap_init` is **size 0** — zero capability
globals). **Fix (in-repo, no bitstream rebuild):** drop `delin gp` / the cap_init loop for
empty-cap_init domains and call `domain_main` within PCC, matching the canonical compiler
output → runs on silicon → cycle numbers. General fix = rework clang `CapstoneCapGlobalInit`
+ start.S off the cursor-0 gp model and retire QEMU hack `7aca0540` (A's lane). See the
state report for the full derivation.

Full ladder + RTL cross-refs: `/home/alexey/.claude-b/plans/curried-crunching-gizmo.md`.
State report: `history/20-07-2026_04-03-20_fpga-freestanding-controller-domain-call-reached.md`.

---

## 8. Once the domain call returns a RESULT (future)

Run the full sweep (borrow + three revoke `.dom`s) via `run_rtl_smoke.py`, harvest the
`RESULT` lines, and feed them to `run-revoke-cost-fpga-qemu.sh --parse-uart` to reproduce the
per-op cycle breakdown next to the QEMU baseline (borrow raw2/borrow6; revoke bump7 /
norevoke60 / revoke65, +5 O(1) revoke-at-free). That is the original deliverable.

---

## Artifacts / pointers

- Good UP+freestanding image: `~/capstone-b-artifacts/fw_payload_fpga_up_ctl.bin` (sha `fe37ebdb`).
- Controller source: `capstone/tests/rtl-smoke/borrow_cost_fpga_ctl.c`; builder
  `build-borrow-cost-fpga.sh`.
- Board driver: `capstone/tests/rtl-smoke/fpga_driver/` (`run_rtl_smoke.py`, `fpga_console.py`).
- Session run scripts (not committed): scratchpad `run_ctl_image{5,6,7}.py`, `rebuild_all.sh`.
