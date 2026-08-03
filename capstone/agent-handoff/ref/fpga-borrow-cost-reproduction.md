# Reproducing the Capstone RTL/FPGA cycle measurements

End-to-end manual to reproduce, on the CapliFive CVA6 "Capstone" FPGA, the
cycle-accurate borrow-cost numbers and the per-primitive breakdown. Read once top
to bottom before running anything. Results: `history/*_RESULTS-fpga-borrow-cost-*.md`.

## 0. Hard rules

- **The board token is secret.** It arrives as `<FPGA-CONSOLE-URL>`.
  Never commit it, never write it into a repo file, never echo it into a captured
  log. Keep it only in `~/.config/capstone/fpga-board-url` (read by the runners).
- **Non-persistent boot.** Boot the firmware via JTAG/gdb into DRAM every run; never
  rely on resident firmware. **A bitstream flash is the ONLY persistent write and is
  a STOP-and-ask** (we cannot rebuild a bitstream here). The one authorized flash is
  restoring the board owner's `working-caplifive-captype-fixed.bit` when another team
  has overwritten it (§6).
- **Good-citizen board use:** lock before driving; power off + unlock in a `finally`
  on every run.
- Commits: no submodule-source commits, no `Co-Authored-By:`. Collaborator-facing
  notes under `/tmp/capstone/`, not the repo.

## 1. What runs, and why it is "gp-free"

The measurement runs inside one Capstone domain entered on a single `REV_SHARED`
region (the domain both computes results and writes them into the region the host
reads back). The domain is built **gp-free / cjalr-free**: our LLVM Capstone backend
otherwise reaches globals via `cincoffset gp,<abs>` assuming `gp = PCC(cursor 0)`, a
form only our QEMU fork fabricates — on silicon `gp = 0`, so `delin gp` stalls. The
gp-free domain has no module statics (so no `gp` use), takes its scratch as a linear
cap carved off the stack, and returns via a plain `ret` (the build script retargets
clang's one capability-return to `jalr zero,0(ra)`), matching the reference monitor's
call/ret ABI. Root-cause trail: `history/20-07-2026_*plain-call-ret*.md`.

## 2. Environment

```bash
cd <your llvm-capstone clone>
source capstone/tests/capstone-test-env.sh
```
Host compiler stays `/usr/bin/clang++` (never a capstone-built clang). The board
driver is `capstone/tests/rtl-smoke/fpga_driver/`; a Python venv with `socketio`
drives it (`/tmp/capstone/fpga-venv` in this workspace).

## 3. Build the probes

```bash
bash capstone/tests/rtl-smoke/build-borrow-cost-fpga-nogp.sh       # super-ops
bash capstone/tests/rtl-smoke/build-borrow-breakdown-fpga-nogp.sh  # primitives
```
Each emits a soft-float freestanding controller (`*_ctl`) and a Capstone `.dom` into
`$CAPSTONE_TMP_ROOT/capstone-rtl-smoke/`. The controller links no glibc (the board's
FPU rejects glibc's hard-float `fsd`); it uses raw Linux syscalls, integer-only
output, and the `/dev/capstone` ioctl protocol. The build scripts assert the domain
is **gp-free and cjalr-free** (a new static or `memcpy` libcall would reintroduce
`gp`; the assert fails the build).

Knobs (breakdown): `BREAKDOWN_ITERS` (default 64), `BREAKDOWN_MREV_ITERS` (default
16 — the `mrev`-only loop never revokes, so keep it small).

## 4. QEMU functional pre-check

Always validate functionally before board time — it catches ABI/build bugs for free.
Needs the shared `rootfs.ext2` write-lock, so serialize with other QEMU suites.

```bash
SHARE=$CAPSTONE_TMP_ROOT/capstone-breakdown-share
bash capstone/tests/rtl-smoke/build-borrow-breakdown-fpga-nogp.sh "$SHARE"
python3 capstone/tests/runtime-qemu/run-domain-smoke.py \
  --share-dir "$SHARE" --qemu-extra-arg=-icount --qemu-extra-arg="shift=0,sleep=off" \
  --guest-command "cp /mnt/host/borrow_breakdown_fpga_nogp_ctl /tmp/bd && chmod 0755 /tmp/bd && /tmp/bd /mnt/host/borrow_breakdown_fpga_nogp.dom" \
  --success-marker "borrow-breakdown-fpga: measurement complete"
```
Under `-icount`, `mcycle` counts retired instructions, so the QEMU figures are
instruction counts (functional proxy), not cycles — use them only to confirm the op
sequence runs and the decomposition is self-consistent. Same recipe with the
borrow-cost controller/domain for the super-operations.

## 5. Board firmware image

Boot `fw_payload_fpga_up_ctl.bin` (a `--mode fpga`, **UP / `CONFIG_SMP=n`** OpenSBI
FW_PAYLOAD with kernel + initramfs + `caplifive.dtb` baked in). It ships `capstone`
built-in (`/dev/capstone` at boot) and the freestanding tools (`base64`, `gunzip`,
`sha256sum`). Build it in the caplifive-system container; the load-bearing gotchas:

- **Must be UP.** The SMP kernel floods the console with thousands of
  `remote fence ... not available in SBI v1.0` lines and buries the login prompt.
- **`make build LINUX_PAYLOAD=1` does not relink OpenSBI** → a 2.1 MB payload with no
  kernel. Force the relink: `make -C build/build/opensbi-custom PLATFORM=fpga/ariane
  CROSS_COMPILE=.../riscv64-buildroot-linux-gnu- LINUX_PAYLOAD=1`. A correct payload
  is ~15 MB.
- **Buildroot doesn't track the cpio dep** → after changing the rootfs overlay, force
  `make build A=linux-rebuild` then relink OpenSBI, or the initramfs is stale.
- `capstone.ko` must match the UP vermagic if loaded via `insmod`; the built-in path
  avoids this (and `insmod` can hang this CVA6's module loader).

The probe binaries **MUST be baked in** (2026-08-03). An earlier revision of this line said
they "do not need to be baked in — the runner UART-transfers them at run time"; UART delivery
is now retired, because each socket.io emit carries 16 chars over an HTTPS round trip and a
~10 KB domain costs minutes, while a baked one is free inside the JTAG upload that happens
anyway. Copy them into `overlay/test-domains/` AND `build/target/test-domains/`, then
`A=linux-rebuild` followed by `A=opensbi-rebuild`. Full container build recipe:
`history/19-07-2026_*fpga-mode-build-run*.md`.

## 6. Bitstream

Measure **only** on `working-caplifive-captype-fixed.bit`. Stock `ariane_xilinx.bit`
has no capability unit → every `cscall` resets → all data is garbage. The runner
verifies `nv_bitstream_name` before measuring. If it is wrong (another team
overwrote it), re-flash the board owner's file (the one authorized persistent write):

- **Power on + settle before flashing** (a cold JTAG programmer errors otherwise).
- **Power-cycle after flashing** — a non-volatile flash only writes SPI; the FPGA
  keeps running the old config until it reconfigures at power-on. Skip this and the
  DTM comes up degenerate.

## 7. Board run

`board_run_breakdown.py` (in `/tmp/capstone/`) does the whole flow and is the
template: connect → lock → **verify bitstream** (re-flash if wrong) → power-cycle →
`upload_boot_image` → gdb `load_image` @0x80000000 → `set $pc`, `$a0=0`, continue →
wait for `login` → quiet the console → confirm `/dev/capstone` → **invoke the BAKED
controller+domain from the shell** (`/test-domains/...`; the UART-transfer step it used to do
here — gzip+base64, per-chunk sha256, retry — is retired) → run bracketed by unique
BEGIN/END markers → harvest the `RESULT` line → **power off + unlock in `finally`**.
Run it under the venv; the resilient socket wrapper survives transient drops.

Harvest: trust only the freshly-printed `RESULT` line inside your BEGIN/END markers
(the console replays stale UART history, which can carry old runs' text).

## 8. The numbers

Super-operations (`mcycle`, cyc/op, 64 iters): **raw 8, borrow 182, copy@256 902,
copy@1024 3611**. Borrow is O(1) in payload; copy is O(payload).

Elementary primitives (cyc/op, small tree): **load 2, shrink 1, mrev 50,
delin+revoke 121** → mrev+delin+revoke 171, borrow ≈173. The cost is the revocation
machinery (`mrev` mint + `revoke` reclaim); `load`, `delin`, `shrink` are 1–2-cyc
register ops. In a tight single-lineage loop borrow(N) ≈ 75 + 3·N/2 (the revocation
tree is never pruned, so `revoke` walks a growing list).

## 9. Platform constraints (design around these)

- **No slot reclamation in the rev-node pool.** `drop`/csdrop *is* implemented — it is
  decoded in QEMU (`helper_csdrop`) and the RTL (`decoder.sv`, rev-node drop endpoint)
  and has an LLVM builtin (`__builtin_capstone_cap_drop`) — but it only *invalidates* a
  node (clears its valid bit); it does not free the node's slot. The rev-node pool is a
  fixed 1024-entry **bump allocator** with a monotonic head (`capstone_rev_node.anvil`:
  head is set once at init and only ever `+1`), and neither `drop` nor `revoke`
  decrements it. So the tree cannot be pruned in software; each `mrev` consumes a slot
  for the domain call's lifetime. A clean single-op `revoke` / pruned O(1) borrow needs
  the RTL allocator to free the slot on `drop`/`revoke`.
- **`delin` is load-bearing.** `mrev`+`revoke` without `delin` returns UNINIT (not a
  reusable linear cap), so it can't loop; `revoke` can't be timed apart from `delin`.
- **Revocation ceiling.** Keep total `mrev`s per domain call well under ~256 — nodes
  are never released; 1024 in one call breaks `domreturn` and wedges the debug
  module. `mrev`-only loops accumulate fastest; keep their count small.
- **`mcycle`, not `cycle`.** The board gates the unprivileged `cycle` counter; read
  `mcycle` in-domain (default in `fpga_instrument.h`).

## 10. Still open

The temporal-safety-vs-CHERI sweep (bump / norevoke / revoke) needs the same gp-free
treatment applied to `revoke_cost_fpga.*` (it uses the revoke-on-free allocator —
more moving parts). The three configs isolate the temporal cost: **bump** = plain
bump allocator, no safety (baseline); **norevoke** = the revoke-on-free allocator
with revocation disabled (alloc-side cost: split+mrev+delin per malloc, free is a
no-op); **revoke** = full temporal safety (revoke per free). Then
`norevoke − bump` = alloc-side overhead and `revoke − norevoke` = the revoke-at-free
op. Feed the board `RESULT` lines to `run-revoke-cost-fpga-qemu.sh --parse-uart`.
