# Task 017 — RFENCE boot fix: root-caused, UP-kernel fix built, but the board boots RESIDENT firmware (out of scope)

**Date:** 2026-07-17
**Branch:** capstone-bootstrap-b
**Scope:** additive test tooling + an our-image rebuild (kernel build flag), no repo/monitor/submodule source changes

## Outcome in one line

Traced the boot hang to a real SBI-RFENCE mismatch and **built the fix** (a UP —
`CONFIG_SMP=n` — kernel image with our six `.dom`s, sha256 `6991c0f7…`). But the
decisive discovery is that **`load-image` + `reset-board` does not boot our
uploaded image at all** — the board boots a **board-resident firmware** (the
collaborator's `root@reference-build` build); our `alexey@focs-server` kernel never
appears in any boot log. Fixing the boot therefore means **re-flashing the
shared board's resident firmware**, which is **out of scope** (Step-1 case 2 /
carve-out). Stop and report. The built UP image is ready to hand to the board
owner to flash.

## The decisive evidence (Step 1: where the running firmware comes from)

| | Kernel build identity |
|---|---|
| **Our uploaded images** (`fw_payload.bin` sha `aadd213f`, `fw_payload_up.bin` sha `6991c0f7`) | `alexey@focs-server` |
| **What the board actually boots** (UART ring buffer) | `root@reference-build`, `#3 SMP Sun May 24` and `#30 SMP Mon May 25` |

Our build id **never** appears on the board; only the collaborator's does. So
`load-image` (JTAG write to `0x80000000`) followed by `reset-board` does **not**
run our image — on reset the board's bootrom reloads the resident firmware from
SPI/SD (the `Hello World! … init SPI … could not initialize sd… exiting` zsbl we
kept seeing), clobbering our JTAG load. The console's `load-image` is effectively
a debug load, not a "boot this firmware" path. **Correction to the previous board-
run note (`17-07-2026_14-50-30`): the OpenSBI+Linux that booted was the RESIDENT
`the reference-build` firmware, not "our OpenSBI + our Linux" — that was a mis-attribution.**

## Root cause of the hang (in the resident firmware)

Confirmed against the genesys-testing sources and the caplifive-system build that
produced the on-disk images:

- The monitor **compiles RFENCE in** (`CONFIG_SBI_ECALL_RFENCE=y`, and
  `CONFIG_SBI_ECALL_LEGACY=y`) — the exact `fpga/ariane` `.config` for the
  byte-identical `aadd213f` build has both. The RFENCE handler in
  `sbi_ecall_rfence.c` is the full standard implementation, **not** stubbed, and
  nothing in `sbi_capstone.c` intercepts it.
- **Yet at runtime the kernel detects TIME and IPI but NOT RFENCE**
  (`riscv: providing IPIs using SBI IPI extension`, `SBI TIME extension
  detected`, `remote fence extension is not available in SBI v1.0`). So the
  Capstone monitor **does not advertise RFENCE to the S-mode domain** — a
  deliberate capability-model choice (case-(b): a domain shouldn't remote-fence
  other harts). Not a config we should force on.
- The kernel is `CONFIG_SMP=y`, `NR_CPUS=64`, **`CONFIG_RISCV_SBI_V01` not set**.
  With SMP on + SBI-IPI, `riscv_use_ipi_for_rfence()` stays false, so remote
  fences route through `sbi_remote_fence_i()` → the **no-op stub** (compiled
  because `SBI_V01=n`) → it just `pr_warn`s (×475) and returns 0. TLB/icache
  maintenance silently no-ops, `/init` runs stale translations, faults ~9 s in
  (t≈39.9 s), and the board resets in a loop.

## The fix that was built (Step 3.2, our-image / kernel side) — ready but unflashable by us

Rebuilt the kernel **UP (`CONFIG_SMP=n`)** so all fences are local (`fence.i` /
local sfence), independent of the monitor's unadvertised RFENCE. Reproducible
recipe (run in the caplifive-system build tree, `sw/buildroot`):

```
LX=build/build/linux-6.4.14 ; CROSS=build/host/bin/riscv64-buildroot-linux-gnu-
cd $LX
./scripts/config --file .config --disable SMP
./scripts/config --file .config --set-str INITRAMFS_SOURCE <abs>/build/images/rootfs.cpio
make ARCH=riscv CROSS_COMPILE=$CROSS olddefconfig
make ARCH=riscv CROSS_COMPILE=$CROSS -j Image
# then rebuild the OpenSBI fw_payload around the new Image:
cd build/build/opensbi-custom
rm -rf build/platform/fpga/ariane/firmware
make PLATFORM=fpga/ariane CROSS_COMPILE=$CROSS FW_PAYLOAD_PATH=<abs>/$LX/arch/riscv/boot/Image
```

Notes: `CONFIG_INITRAMFS_SOURCE` had to be pinned to the absolute `rootfs.cpio`
(the buildroot `${BR_BINARIES_DIR}` var doesn't expand in a direct kernel build,
else the initramfs would be empty). The DTB is **not** embedded in `fw_payload`
(no `d00dfeed` in either our image or the resident one) — the board supplies it
at `0x82200000`, so no FDT flag was needed.

**Verification:** the resulting `fw_payload_up.bin` (sha256
`6991c0f797182c5f59f292647c814927c46eaeb54bd26f34b1220d3deda61bfd`, 15,367,176 B)
embeds the initramfs (4,126,208 B) with **all six** `/root/rtl-smoke/*.dom`,`*.user`.

But because the board boots resident firmware, uploading + resetting this image
does **not** run it — confirmed by a full gated board run (Lock → upload → load
(retry cleared the post-power-on JTAG transient) → reset → 180 s wait → still the
`the reference-build` resident boot loop → Lock released). So the fix cannot be applied from
our side.

## What needs the board / monitor owner (hand-off)

Pick one; all are owner-side because they touch resident/shared infrastructure:

1. **Flash `fw_payload_up.bin` (UP kernel, sha `6991c0f7…`) as the board's
   resident firmware.** It already carries our `.dom`s and avoids RFENCE entirely.
   Simplest path to numbers.
2. **Re-flash a resident firmware whose monitor exposes RFENCE to the domain**
   (or a resident kernel built UP / with `CONFIG_RISCV_SBI_V01=y` to use the
   monitor's advertised legacy fence).
3. Provide a console path that actually **boots the JTAG-loaded image** instead of
   the resident SPI image, which would make this in-scope for us again.

## Deliverables status

- **Numbers:** none — the board cannot boot our (or any) image to a shell as
  currently flashed. QEMU reference unchanged: bump 7 / norevoke 60 / revoke 65 →
  revoke-at-free +5, O(1).
- **Fix:** built + verified (`fw_payload_up.bin`, sha `6991c0f7…`), staged at
  `/tmp/capstone-b/fpga-image/`, ready for the owner to flash. Recipe above.
- **Boot-to-shell fix documented:** yes (this note).

## Guardrails honoured

Token never entered the tree or any log (all board output redacted; 0 token
occurrences across every log/capture). Good-citizen: ran only with the operator's
explicit `--ignore-users` authorization for this perf run, Lock taken only while
running and released on exit, no power-off. No monitor/RTL/submodule-source or
`llvm/` changes; the kernel change is a build flag applied in the scratch build
tree, documented here for reproducibility, not committed into any submodule
history.
