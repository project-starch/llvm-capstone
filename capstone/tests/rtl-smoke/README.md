# RTL/FPGA smoke test — cycle-accurate borrow cost (task-016)

Hardware port of the task-014 borrow-cost probe. Goal: the **cycle-accurate**
per-operation cost (raw / borrow / copy) that the QEMU instruction-count proxy
in `paper/evaluation.tex` (`tab:borrowcost`, `sec:eval-perf`) stands in for.

**Status: prepared, UNTESTED on hardware.** Everything here is built and staged
by an agent; the build must be compiled on the caplifive toolchain and the run
is human-driven on the FPGA web console. Two open items (below) may need one
tweak each — flag them to the RTL collaborator while reachable.

## What changed from the QEMU probe

The measured loops are **identical** to
`../runtime-qemu/borrow-cost-probe/borrow_cost.c` (byte-for-byte; that is the
point — measure the same code on silicon). Only the instrumentation differs,
because the QEMU probe uses two **emulator-only** ops that do not exist on the
CVA6/Capstone core:

| Concern | QEMU probe | This variant |
|---------|-----------|--------------|
| Cycle count | `csrdicount` (QEMU icount op) | `rdcycle` CSR (`fpga_instrument.h`) |
| Result output | `csdebugcount*` → QEMU serial | domain writes 8 results into the shared region; controller `printf`s them → **UART** |

Files:
- `borrow_cost_fpga.c` — domain payload (Capstone clang). Measured loops copied
  verbatim from the QEMU probe (**keep in sync**); reads `rdcycle`; writes results
  through the reclaimed LINEAR handle into the region base.
- `borrow_cost_probe_guest_fpga.c` — controller (buildroot gcc). Same
  create-domain / share-region / call as the QEMU controller, then reads the 8
  results back and prints `RESULT` lines (per-op cycles + ×vs-raw) on the UART.
- `fpga_instrument.h` — the `rdcycle` read + result-slot writeback.
- `build-borrow-cost-fpga.sh` — builds `.user` + `.dom` (same two-toolchain split
  as `build-borrow-cost-probe.sh`).

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
# in a caplifive-system checkout:
mkdir -p overlay/root/rtl-smoke
cp $CAPSTONE_TMP_ROOT/capstone-rtl-smoke/borrow_cost_fpga.user overlay/root/rtl-smoke/
cp $CAPSTONE_TMP_ROOT/capstone-rtl-smoke/borrow_cost_fpga.dom  overlay/root/rtl-smoke/
# point buildroot at the overlay (BR2_ROOTFS_OVERLAY) — exact path TBC with
# the collaborator (see open item 2), then:
scripts/build-software.sh --mode fpga
#   → sw/buildroot/build/opensbi-custom/build/platform/generic/firmware/fw_payload.bin
```

The modcapstone kernel module + the Capstone monitor are already part of the
capstone buildroot (`caplifive-buildroot`, the same tree the QEMU flow uses), so
`/dev/capstone` and the domain-load path work on the FPGA exactly as under QEMU.

## Run (human-driven on the FPGA web console)

No scriptable API yet (browser GUI only). Sequence on `fpga.corank.info`:

1. **Boot Images** → upload `fw_payload.bin` (JTAG to `0x80000000`, ~2 min).
2. **Reset**; wait for the Linux prompt on the **Terminal** (UART) tab.
3. Run: `/root/rtl-smoke/borrow_cost_fpga.user /root/rtl-smoke/borrow_cost_fpga.dom`
4. Copy the `RESULT` lines off the Terminal tab into `RESULTS.md`.

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

1. **`rdcycle` inside a Capstone domain.** `rdcycle` reads the `cycle` CSR, which
   S/U mode may read only if the higher level sets `[m|s]counteren.CY`. Linux sets
   `scounteren.CY`, but whether the monitor exposes the counter to a *domain* is
   unconfirmed. **If the domain faults on `rdcycle`:** either (a) have the monitor
   set counteren.CY for the domain context, or (b) read `mcycle` bare-metal in
   M-mode (swap the one asm line in `fpga_instrument.h`; `mcycle` is always M-mode
   readable — `capstone-ariane core/csr_regfile.sv:677`).
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
