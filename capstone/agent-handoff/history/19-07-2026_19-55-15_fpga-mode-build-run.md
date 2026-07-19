# Task 018 — build the FPGA-paired image (`--mode fpga`), boot on captype-fixed, run the domain benchmark

**Date:** 2026-07-19
**Branch:** capstone-bootstrap-b
**Scope:** rebuild the FPGA software the official way (`caplifive-system`,
`build-software.sh --mode fpga`, per the board owner), boot it on the already-flashed
`working-caplifive-captype-fixed.bit` (non-persistent gdb-boot, no reflash), run
borrow-cost. Board-driver + our-image build changes only; no submodule-source commits.

## Reframe that started this task

The board owner (Jason) resolved two things: (1) **use `working-caplifive-captype-fixed.bit`,
don't regenerate the bitstream**; (2) our earlier custom image load-faulted on the UART
because it **wasn't built for FPGA** — build the software with the FPGA mode via
`caplifive-system` (`scripts/build-software.sh --mode fpga`, his "`--fpga`"), which
regenerates the payload + FPGA device tree with the correct UART/PMP setup.

## What was done

1. **Synced** `capstone-bootstrap` into the `-b` lane (new submodule
   `capstone/caplifive-system` @ `50e9ca8d`). Initialised only the **software** submodules
   (skip the heavy RTL/Vivado/anvil pull): `caplifive-system` →
   `sw/buildroot` (`captainer-buildroot`), `sw/capstone-c`, and buildroot's nested
   `buildroot` (upstream), `components/opensbi` (`b8ec99d6`), and **`capstone-sbi`
   `99aaffa8`** (the board-validated *genesys-testing* reference monitor).

2. **Built the container toolchain.** Podman isn't installed here; **Docker is**. Added a
   `podman`→`docker` shim (scratchpad) so `scripts/build-image.sh` /
   `run-in-container.sh` work unmodified. Built `caplifive-build:latest` (Ubuntu 22.04 +
   rust + opam/OCaml 5.2.0 + buildroot host deps).

3. **Built `--mode fpga`** in the container (skipped `setup.sh`'s recursive submodule
   pull — the software submodules were already checked out on the host and bind-mounted).
   **Gotcha found + fixed:** the recipe's `make build` then `make build LINUX_PAYLOAD=1`
   did **not** re-trigger the OpenSBI build, so `fw_payload.bin` came out 2.1 MB (OpenSBI +
   a stub test payload, **no kernel**). Forced the OpenSBI firmware to relink with
   `LINUX_PAYLOAD=1` (`make -C build/build/opensbi-custom PLATFORM=fpga/ariane
   CROSS_COMPILE=... LINUX_PAYLOAD=1`) → a proper **15.37 MB `fw_payload.bin`** with the
   RISC-V kernel embedded at 0x200000 and **`caplifive.dtb` baked in**
   (`FW_PAYLOAD_FDT_PATH`, so boot needs only `--image`, no separate `--dtb`).
   sha256 `aa097a09…` staged at `~/capstone-b-artifacts/fw_payload_fpga_mode.bin`.
   The benchmark binaries were pre-built and dropped into the buildroot **rootfs overlay**
   (`sw/buildroot/overlay/root/rtl-smoke/`) so they bake into the initramfs; verified
   present in `rootfs.cpio` alongside the matching lp64d glibc interpreter.

## Headline result — the UART fault is GONE; the domain path is now reachable

**On `working-caplifive-captype-fixed.bit`, the `--mode fpga` image boots OpenSBI +
Linux to a shell — the S-mode UART load-access-fault is GONE.** So the earlier
`mcause=0x5 / mtval=0x1000000c` fault was **our custom image not being built through the
`fpga/ariane` platform path** (wrong DTB/PMP pairing), exactly as the owner said — NOT an
RTL fault, NOT the domain CALL. The prior "it's RTL / stock-Ariane" narrative is retired.

**`insmod /capstone.ko` succeeds → `/dev/capstone` appears.** On genuine Capstone silicon
the module loads fine; **insmod does NOT hang** here. The prior "insmod hangs this CVA6"
was on hardware of uncertain provenance and does not reproduce on the correct
bitstream + official image. (So the UP *built-in* driver patch is not needed for that
reason.)

## The real blocker to a reliable shell: SMP kernel RFENCE flood

The stock `--mode fpga` kernel is **`CONFIG_SMP=y`** but the platform DTS has **1 CPU**.
The SMP kernel routes TLB/icache flushes through the SBI **remote-fence** path, and this
OpenSBI is seen by the kernel as **SBI v0.1 (no RFENCE extension)**, so
`__sbi_rfence_v01` (`arch/riscv/kernel/sbi.c:213`) fires a **`pr_warn` on every fence** —
`remote fence extension is not available in SBI v1.0` — flooding the console continuously
(2000+ lines). That flood buries the login prompt, so the driver's `login_root` can't
confirm a shell (it timed out on 4 of 5 boots; one early boot got a lucky clean window
and reached the shell + insmod). On a single hart the remote fence is functionally a
no-op, so this is console noise — but it also means icache flushes are no-ops under
SMP+v01, which is bad for the domain CALL.

Editing the DTB `bootargs` (`quiet`) did **not** take effect (kernel command line stayed
`earlycon console=ttyS0,57600`; OpenSBI/kernel didn't honour the modified DTB) — a dead
end. **The correct fix is an UP kernel (`CONFIG_SMP=n`)**: no remote-fence calls at all
(no flood) and proper *local* `fence.i` icache flushes. This **re-validates the earlier
UP-image finding** ([[fpga-up-image-vermagic]]) — now demonstrated by the clean baseline,
not assumed. Rebuilt `--mode fpga` with `CONFIG_SMP=n` (kernel + `capstone.ko` rebuilt to
match the UP vermagic + rootfs + payload relink).

## Borrow-cost / domain CALL result — reached the domain path, blocked at a UART-TX stall

With the **UP image + the driver login-history fix** (below), a clean run gets all the
way through: boot → shell → `insmod` OK → the borrow controller runs and **`create_dom`
succeeds** — the kernel logs `Domain memory region vaddr=… paddr=819a0000` and
`code size = 3776`, and a valid `dom_id` is returned. **No bootrom reset** — the historical
"domain CALL resets the core" does NOT reproduce on the correct bitstream + UP image.

**Reproducible blocker (2 identical runs):** the very next userspace `printf`
(`borrow_cost_probe_guest_fpga.c:57`, "created domain ID") emits exactly **16 bytes**
(`borrow-cost-fpga`) — the 8250 TX FIFO depth — then **hangs for the full 240 s**.
Kernel `printk` (which *polls* THRE) worked right up through `create_dom`, but the first
userspace `write()` fills the FIFO and blocks forever waiting for a TX interrupt that
never comes. **Diagnosis: `create_dom` leaves S-mode interrupt delivery broken**, so
interrupt-driven UART TX stalls. This is *before* the actual `call_dom` (line 90), so the
domain CALL itself is still not exercised — the benchmark hangs at domain *creation*.

**Precise root cause (via a non-destructive GDB CSR probe of the hung core):** the hang is
a **synchronous illegal-instruction exception** (`mcause = 0x2`) from a **userspace** PC
(`mepc` in the `0x2a…` mmap region), and the core is parked in the monitor's exception
handler spinning in a **`while(1)`** (`pc = 0x8001716x`, disassembling to `li t0,1; bnez
t0,<self>`). `capstone-sbi/sbi_capstone.c` `handle_exception`/`handle_interrupt` only
service a few cases and `while(1)` on anything else, and `medeleg` does **not** delegate
illegal-instruction (bit 2) to S-mode — so any userspace instruction the monitor can't
handle traps to M-mode and **hangs the whole system** (silent freeze; that is why *all*
output stops dead). The 16-byte "FIFO" coincidence earlier was a red herring — it was the
first substantial glibc `printf` after `create_dom`, and the trap is at that instruction.

**Layer 1 — disabled FPU (found + fixed).** The first faulting instruction was **`fsd`
(FP store)** — an FP op traps illegal exactly when `mstatus.FS = Off`. **The monitor never
enables S-mode FP** (`MSTATUS_FS` is `#define`d but never written), so Linux userspace runs
with the FPU off; the first glibc FP register-save traps → monitor `while(1)`. Fix applied
(staged patch): set `mstatus.FS` before the `mret` to S-mode in `sbi_capstone.S`
`return_to_sumode` (`li t0,0x6000; csrs mstatus,t0`). Verified working — after the fix the
probe shows `mstatus.FS = 2` (Clean, FP enabled).

**Layer 2 — still hangs (open).** With FP enabled, the controller gets *further* but still
takes an `mcause=0x2` illegal instruction at a different userspace PC → same monitor
`while(1)`. So the glibc-linked Linux **controller** uses more instructions this
CVA6+monitor combo can't service. This is the ABI/ISA-fit issue the task flagged: the
benchmark's **`.user` controller is a full glibc program**, and glibc's optimized routines
hit instructions the monitor `while(1)`s on. The collaborator almost certainly runs
benchmarks as **bare-metal domains** (the `.dom`), not via a glibc Linux controller.

**Two ways forward to actually get the cycle number (both need more work/collaborator
context):**
1. **Controller side (preferred):** rebuild `borrow_cost_fpga.user` **freestanding /
   static, integer-only I/O** (no glibc — write(2)-based integer printer), so it never
   emits the offending instructions. Matches how the domain code is already built.
2. **Monitor side:** make `handle_exception` **delegate** unhandled illegal instructions to
   S-mode (or emulate them) instead of `while(1)` — but delegating illegal-instr risks the
   capability-instruction emulation path, so it needs the monitor author's sign-off.

Also worth doing regardless: change the monitor's `while(1)` traps into a readable trap
dump (so future hangs self-report instead of freezing silently), and confirm with the
collaborator whether the RFENCE-flood / no-S-mode-FP / `while(1)`-on-unhandled behaviours
are expected on this reference monitor for the FPGA Linux path.

Note: structural DTB edits DO take effect (a UART-interrupt-removed DTB booted with
`irq = 0`); the earlier `bootargs quiet` "not taking effect" was a history-replay artifact
in the capture. The polled-UART image is not needed given the illegal-instruction diagnosis
(polled also hangs — at the same illegal instruction, just fewer bytes out first).

## Driver improvements (additive; on capstone-bootstrap-b working tree)

- `run_command` now **throttles the keystrokes** char-by-char (the board's UART RX FIFO
  overruns on a bulk write and silently drops chars — the borrow command arrived as
  `row_cost_fpga`), with a leading Ctrl-U to clear a partial line. The bug the earlier
  sweeps hit.
- **Tolerant / shorter completion markers** (`RESULT vs.?raw`, `measurement.?complete`,
  and a tripled-sentinel `OKOK|NONO` insmod check) so a single dropped output char no
  longer fails the capture; `load_capstone_module` now tests `/dev/capstone` directly
  (idempotent — survives an already-loaded module).
- `--runs LABELS` filter to run just the decisive **borrow** pair first without churning
  the shared board through the whole sweep.

## Verdict on the old patches (task deliverable)

- **`fence.i` domain-boundary patch:** not applied to this image; verdict deferred to the
  UP borrow run (if borrow returns on the stock-monitor UP image, it was never needed on
  correct hardware).
- **UP built-in-capstone patch:** the *built-in* half is NOT needed (insmod works). The
  **UP (SMP=n) kernel** half IS needed — but for the RFENCE-flood reason above, not the
  insmod-hang reason originally cited.

## Artifacts

- `~/capstone-b-artifacts/fw_payload_fpga_mode.bin` (SMP, sha `aa097a09…`),
  `fw_payload_fpga_quiet.bin` (SMP+quiet DTB, ineffective), and the UP rebuild (below).
- UART captures `board-run-fpgamode-borrow*.uart.txt`, `board-run-fpgaquiet-borrow.uart.txt`.
- Container image `caplifive-build:latest`; build logs in scratchpad.
