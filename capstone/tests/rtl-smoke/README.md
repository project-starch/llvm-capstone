# RTL/FPGA smoke test — cycle-accurate borrow cost (task-016)

Hardware port of the task-014 borrow-cost probe. Goal: the **cycle-accurate**
per-operation cost (raw / borrow / copy) that the QEMU instruction-count proxy
in `paper/evaluation.tex` (`tab:borrowcost`, `sec:eval-perf`) stands in for.

**Status: plumbing VALIDATED under QEMU (2026-07-14); UNTESTED on hardware.** The
two-region result hand-off, the `rdcycle` read inside a domain, and the
controller read-back all work on the functional model — see `RESULTS.md`. The
QEMU pass caught (and this port now fixes) two defects that would have wasted a
hardware slot: an unsound single-region read-back (the task-007 host-landmine)
and an `-O2` Capstone-backend ICE. The run on the board is still human-driven on
the FPGA web console; the remaining open items (below) are narrowed.

## What changed from the QEMU probe

The measured loops are **identical** to
`../runtime-qemu/borrow-cost-probe/borrow_cost.c` (byte-for-byte; that is the
point — measure the same code on silicon). Only the instrumentation differs,
because the QEMU probe uses two **emulator-only** ops that do not exist on the
CVA6/Capstone core:

| Concern | QEMU probe | This variant |
|---------|-----------|--------------|
| Cycle count | `csrdicount` (QEMU icount op) | **`mcycle` CSR** (`fpga_instrument.h`; the board gates unprivileged `cycle`, so `mcycle` is the default — `-DFPGA_CYCLE_USE_RDCYCLE` selects `rdcycle`) |
| Result output | `csdebugcount*` → QEMU serial | domain writes results into a **retained (`REV_SHARED`) results region**; controller `printf`s them → **UART** |

The result region is **separate** from the borrow arena and handed `REV_SHARED`,
not `REV_TRANSFERRED`: a transferred region cannot be read back by the host after
the domain revokes it (the read traps — task-007 host-landmine; see `RESULTS.md`).

Two ports live here, both hardware ports of a `../runtime-qemu/` cost probe:

**Borrow-cost** (paper `sec:eval-perf` — the per-op borrow vs copy cost):
- `borrow_cost_fpga.c` — domain payload (Capstone clang). Measured loops copied
  verbatim from the QEMU probe (**keep in sync**); reads the cycle counter; writes
  the 8 results into the retained results region.
- `borrow_cost_probe_guest_fpga.c` — controller (buildroot gcc). Same
  create-domain / share-region / call as the QEMU controller, then reads the 8
  results back and prints `RESULT` lines (per-op cycles + ×vs-raw) on the UART.
- `build-borrow-cost-fpga.sh` — builds `.user` + `.dom`.

**Revoke-cost** (paper `sec:eval-perf-compare` — the temporal-safety overhead vs
CHERI, the headline comparison):
- `revoke_cost_fpga.c` — domain payload. Hardware port of
  `../runtime-qemu/revoke-cost-probe/revoke_cost.c` (**keep in sync**): the
  malloc/touch/free loop under one allocator config per build
  (`-DROF_COST_MODE`: bump / norevoke / revoke); reads the cycle counter; writes
  the 4 counters into the retained results region.
- `revoke_cost_probe_guest_fpga.c` — controller. Shares the LINEAR arena
  (`REV_TRANSFERRED`) + a small results region (`REV_SHARED`, one page — two
  large regions starve the arena, see `RESULTS.md`), reads the counters back,
  prints the `RESULT` line.
- `build-revoke-cost-fpga.sh` — builds `.user` + the three `.dom` configs.
- `run-revoke-cost-fpga-qemu.sh` — QEMU plumbing-validation: builds, runs the
  three configs, and parses the `RESULT` lines into the temporal-safety
  breakdown. Its parser doubles as the **UART parser** for the board:
  `./run-revoke-cost-fpga-qemu.sh --parse-uart <pasted-terminal-capture>`.

Shared:
- `fpga_instrument.h` — the cycle read (`mcycle` default / `rdcycle` fallback) +
  the borrow-cost result-slot writeback.

## Build

```sh
source ../capstone-test-env.sh
./build-borrow-cost-fpga.sh            # → $CAPSTONE_TMP_ROOT/capstone-rtl-smoke/
#   borrow_cost_fpga.user   (Linux userspace controller)
#   borrow_cost_fpga.dom    (Capstone domain payload)
#   asm/borrow_cost_fpga.s  (static cross-check of the per-op counts)
```

## Assemble the FPGA boot image (`fw_payload.bin`)

The FPGA boots an OpenSBI **`fw_payload.bin`** (OpenSBI + kernel baked in), built
via the umbrella repo `github.com/project-starch/caplifive-system`. Our two
artifacts ride inside it via the buildroot **rootfs overlay** (embedded
initramfs) — there is no separate block device on the FPGA path.

```sh
# build both ports into the default out dir first:
./build-borrow-cost-fpga.sh          # → $CAPSTONE_TMP_ROOT/capstone-rtl-smoke/
./build-revoke-cost-fpga.sh          # (same out dir)

# in a caplifive-system checkout, stage all artifacts into the rootfs overlay:
mkdir -p overlay/root/rtl-smoke
cp $CAPSTONE_TMP_ROOT/capstone-rtl-smoke/borrow_cost_fpga.user   overlay/root/rtl-smoke/
cp $CAPSTONE_TMP_ROOT/capstone-rtl-smoke/borrow_cost_fpga.dom    overlay/root/rtl-smoke/
cp $CAPSTONE_TMP_ROOT/capstone-rtl-smoke/revoke_cost_fpga.user   overlay/root/rtl-smoke/
cp $CAPSTONE_TMP_ROOT/capstone-rtl-smoke/revoke_cost_fpga_*.dom  overlay/root/rtl-smoke/
# point buildroot at the overlay (BR2_ROOTFS_OVERLAY = our own dir, per the
# collaborator), then:
scripts/build-software.sh --mode fpga
#   → sw/buildroot/build/opensbi-custom/build/platform/generic/firmware/fw_payload.bin
```

The modcapstone kernel module + the Capstone monitor are already part of the
capstone buildroot (`caplifive-buildroot`, the same tree the QEMU flow uses), so
`/dev/capstone` and the domain-load path work on the FPGA exactly as under QEMU.

### Patched board image (`fw_payload_up_builtin_fence.bin`, sha256 `9c53ffd8...`)

The board image to run the sweep with is the **UP built-in image carrying the
domain-switch `fence.i` fix** (see `agent-handoff/history/*_fpga-domain-call.md`
and `*_fpga-domain-call-rebuild.md`). Two independent fixes are baked in:

- **capstone built into the kernel** (`obj-y`, not a module) — `/dev/capstone`
  exists at boot, no `insmod` (which hangs this CVA6). Boot with the driver's
  `--builtin` flag (skips the insmod step).
- **the 9 `fence.i` icache-flushes** restored to the OpenSBI Capstone monitor
  (`sbi_capstone.S`) — the CVA6 needs them across the M-mode↔domain switch; our
  lineage had dropped them, which is why the first domain CALL stalled on the
  board while QEMU (no icache model) was fine.

Rebuild recipe (from this repo; no external `caplifive-system` checkout needed,
reusing the existing UP built-in Linux Image):

```sh
BR=capstone/caplifive-buildroot
CROSS=$BR/build/host/bin/riscv64-buildroot-linux-gnu-
# 1. apply the monitor patch to the OpenSBI capstone-sbi submodule working tree
git -C $BR/components/opensbi/lib/sbi/capstone-sbi apply \
    ../../../../../agent-handoff/patches/opensbi-capstone-sbi-domain-switch-fence-i.patch
# (and mirror it into the rsynced build copy)
cp $BR/components/opensbi/lib/sbi/capstone-sbi/sbi_capstone.S \
   $BR/build/build/opensbi-custom/lib/sbi/capstone-sbi/sbi_capstone.S
# 2. extract the UP built-in Linux Image from the prior image (offset 0x200000)
dd if=fw_payload_up_builtin.bin of=Image_up_builtin bs=$((0x200000)) skip=1
# 3. rebuild the fpga/ariane OpenSBI, wrapping that Image
cd $BR/build/build/opensbi-custom && rm -rf build/platform/fpga/ariane/firmware
make PLATFORM=fpga/ariane CROSS_COMPILE=$CROSS FW_PAYLOAD_PATH=<abs>/Image_up_builtin
#   → build/platform/fpga/ariane/firmware/fw_payload.bin
# QEMU regression check (fence.i is a no-op under QEMU; this confirms no break):
#   cp the rebuilt generic fw_jump.elf into $BR/build/images/, then boot the
#   Image_up_builtin under virt-capstone → shell + /dev/capstone at boot +
#   borrow .dom RESULT raw=2/borrow=6; run-revoke-cost-fpga-qemu.sh → 7/60/65.
```

## Run (human-driven on the FPGA web console)

No scriptable API yet (browser GUI only). Sequence on `the FPGA web console`:

1. **Boot Images** → upload `fw_payload.bin` (JTAG to `0x80000000`, ~2 min).
2. **Reset**; wait for the Linux prompt on the **Terminal** (UART) tab.
3. Borrow-cost (one run):
   `/root/rtl-smoke/borrow_cost_fpga.user /root/rtl-smoke/borrow_cost_fpga.dom`
4. Revoke-cost (three runs, one per allocator config):
   ```
   /root/rtl-smoke/revoke_cost_fpga.user /root/rtl-smoke/revoke_cost_fpga_bump.dom
   /root/rtl-smoke/revoke_cost_fpga.user /root/rtl-smoke/revoke_cost_fpga_norevoke.dom
   /root/rtl-smoke/revoke_cost_fpga.user /root/rtl-smoke/revoke_cost_fpga_revoke.dom
   ```
5. Copy the `RESULT` lines off the Terminal tab into a file, then derive the
   paper numbers locally:
   `./run-revoke-cost-fpga-qemu.sh --parse-uart <that-file>` (revoke-cost
   temporal breakdown; the borrow-cost `RESULT` line is already the per-op cost).

(The hardware **tracer** — switches 0/1 + Trace Dump — is *not* needed for this
cycle-count run; it is for the T7/T10 event trace / security-fault demo.)

## Expected output shape

```
borrow-cost-fpga: RAW iters=1024 empty=… raw=… borrow=… copy256=… copy1024=…
borrow-cost-fpga: RESULT cycles/op  raw=…  borrow=…  copy@256B=…  copy@1024B=…
borrow-cost-fpga: RESULT vs-raw     borrow=…x  copy@256B=…x  copy@1024B=…x
```

The `RESULT cycles/op` line is the paper number: it should show the borrow as a
**small payload-independent constant** over raw, and the copy **growing with
payload** — the same *shape* as the QEMU proxy (raw 2 / borrow 6 / copy 34@256B /
130@1024B instr), now in real cycles. A cross-check: the QEMU instruction ratios
should roughly bound the hardware cycle ratios.

## Open items (verify on first boot; ask the collaborator while reachable)

1. **Cycle counter inside a Capstone domain — RESOLVED for QEMU, verify on board.**
   The collaborator confirmed the board **gates the unprivileged `cycle`**, so the
   probe reads **`mcycle`** (the default). Under our QEMU + OpenSBI Capstone
   monitor the domain reads `mcycle` (and `rdcycle`) with **no fault**
   (`RESULTS.md`) — so the domain-payload model works with `mcycle` without a
   bare-metal M-mode harness. **If the domain faults on `mcycle` on the board:**
   either (a) build `-DFPGA_CYCLE_USE_RDCYCLE` and have the monitor set
   `counteren.CY` for the domain context, or (b) run the measurement bare-metal in
   M-mode (`mcycle` is always M-mode readable — `caplifive-cva6 core/csr_regfile.sv:677`).
2. **Overlay path / defconfig.** The exact `BR2_ROOTFS_OVERLAY` wiring in
   `caplifive-system` (question is out with the collaborator). Until confirmed,
   the overlay dir above is a placeholder.
3. **RTL identity.** Confirm the on-board bitstream is built from the tracing RTL
   (`caplifive-cva6` vs the `capstone-ariane` submodule we read) so `rdcycle`/the
   tracer CSRs are present.

## Deliverable

Fill `RESULTS.md` with the `RESULT` lines + the QEMU↔RTL gap notes (boot format,
any config/port needed, `rdcycle` availability). That RESULTS + the cycle number
is what upgrades `evaluation.tex`'s perf subsection from proxy to cycle-accurate.
