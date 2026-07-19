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

## 7. Stage-0: catch the domain-call trap cleanly (the diagnostic step)

Post-hoc gdb-probing is inconclusive because the reset runs the bootrom over the CSRs. Build
a monitor that turns the silent reset into a readable trap dump, then run it:

1. In the OpenSBI Capstone monitor, program M-mode **`mtvec`** (it is left at reset-default
   `ROMBase+0x40` = bootrom and is dormant in normal operation) to a tiny handler that writes
   `mcause/mepc/mtval` to the ariane uart8250 (@0x10000000, reg-shift 2: THR +0x00, LSR +0x14)
   then halts. **Build obstacle:** the monitor C (`sbi_capstone_dom.c`, which `#include`s
   `capstone-sbi/sbi_capstone.c`) is pre-compiled by the Capstone capability compiler into a
   **checked-in `sbi_capstone_dom.c.S`** and there is no rule to regenerate it here (no
   capstone clang on PATH). So inject the dumper as **raw asm directly into**
   `build/build/opensbi-custom/lib/sbi/sbi_capstone_dom.c.S`, using the `lla` idiom the file
   already uses (`la` triggers a binutils `elfnn-riscv.c:2358` link crash here).
2. Make the dumper **LSB-nibble-first** and **bound the THRE poll** — the board hardware-resets
   ~9 chars into the handler, and the earlier MSB-first / unbounded-poll dumper got truncated
   before the exception code and hung after ~2 chars on real UART timing.
3. This image must ALSO have the **freestanding controller in the overlay** (§3–§4) — the
   older `diag0` dumper image predates the freestanding fix and would hang at the fsd blocker
   before ever reaching the domain call. So it is a fresh combined build.
4. Boot it (§6), reach the `cscall`, capture the trap dump.

**Branch on the exception code (`mcause` low bits):**
- instruction-access-fault / illegal-instr at ~`0x10044` → **stale-icache fetch** of the
  freshly-placed domain code (CVA6 does no icache invalidate at the switch; QEMU models no
  icache, so it never bit there) → try a `fence.i` at the domcall boundary (Stage-1A).
- cap-violation causes **25–28** → compare the RTL guard that fired
  (`commit_stage.sv:205-229`, `capstone_dyn_unit.anvil:226-291`) against the QEMU golden
  model (`capstone-qemu op_helper.c helper_cscall`) to decide monitor-fix vs RTL-bug.
- if a `fence.i` is issued at the boundary but the fetch still faults at `0x10044` → this
  CVA6's `fence.i` does not flush the icache → **RTL**, write up for an out-of-tree bitstream
  rebuild; do not loop on fence variants.

Full staged ladder + RTL cross-refs: `/home/alexey/.claude-b/plans/curried-crunching-gizmo.md`.

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
