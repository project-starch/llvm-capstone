# Task — Flash working-caplifive-captype-fixed.bit; read the boot-handoff mcause

**Date:** 2026-07-19
**Branch:** capstone-bootstrap-b
**Scope:** flash the Capstone bitstream the board owner named, boot our diagnostic image on it,
read the exception the dumper catches. Board driver only (no submodule-source commits).

## Headline results

1. **The board had been overwritten by the other team.** The console's own
   `GET /api/bitstreams` + `flash_state` report the flashed NV bitstream as
   **`ariane_xilinx.bit`, modified 2026-07-19 01:10** (today) — the newest entry in
   the store. That is **stock upstream CVA6/Ariane** (no Capstone capability unit), so
   it would reset/trap on any `cscall`. **All prior Stage-0 board evidence was gathered
   on this wrong bitstream** and must be treated as contaminated. (Confirms the board owner's
   "the other team might have flashed it with something different.")

2. **Restored the real Capstone RTL.** Flashed `working-caplifive-captype-fixed.bit`
   (the board owner's file), **non-volatile**. Two driver bugs found + fixed along the way:
   - *Flash race:* the driver flashed immediately after `power(True)` on a cold board —
     the FPGA/JTAG programmer isn't up yet, so `flash_state -> error` in ~1 s (no SPI
     write). Fix: power-on + settle **before** the flash (`run_rtl_smoke.py`).
   - *Reconfigure:* a non-volatile flash only writes SPI; the FPGA keeps running the old
     config until it **reconfigures at power-on**. Must **power-cycle** after the flash
     (`--power-cycle`) or the DTM comes up degenerate (IDCODE 0x00000001, abstract
     commands time out, `load_image` fails "waiting for busy to go low").
   With flash + power-cycle, the DTM is healthy and the 15 MB JTAG load succeeds.

3. **On captype-fixed, our image boots OpenSBI cleanly** (full "OpenSBI v1.3 for
   Capstone" banner, correct Domain0 regions, hands to S-mode at 0x80200000) — the
   furthest we have ever booted. Then it **traps at S-mode Linux** and goes silent
   (no bootrom banner — a trap, not a reset-to-bootrom).

4. **The exception (read via a non-destructive GDB CSR probe — new `--gdb-probe`):**
   - `mcause = 0x5` — **Load access fault** (synchronous; NOT cap-violation 25–28, NOT
     illegal-instruction 2).
   - `mepc = 0xffffffff8018ab12` — a **kernel virtual address**: the fault is in
     **S-mode Linux** (paging already up), not the monitor, not a domain call.
   - `mtval = 0x1000000c` — faulting load target = the **uart8250** (@0x10000000,
     offset 0xc = 16550 LCR, reg-shift 2).
   - `pc = 0x80023f0a` — core parked in the dumper's `putc` THRE poll.
   - `medeleg = 0xb109` has bit 5 clear → load-access-fault is not delegated to S-mode →
     traps to M-mode (mtvec = our dumper).

## Interpretation

This is **not** the domain-CALL reset bug. On genuine Capstone RTL, Linux boots far
enough to init its 8250 console, then **faults reading a UART register at phys
0x1000000c**. OpenSBI wrote THR (offset 0) fine; the Linux driver's access to
LCR (offset 0xc) load-access-faults. Two candidate causes, both = **image/DTB not
paired with this bitstream**:

- **DTB mismatch:** `caplifive.dtb` describes the UART (reg-width / reg-shift / address)
  differently than captype-fixed's actual peripheral → the 8250 probe access faults.
- **Cap-type mismatch:** the bitstream is literally *captype-**fixed***. Our monitor's
  `sbi_capstone_dom.c.S` was pre-compiled by an older Capstone capability compiler; if
  the S-mode DDC is built with a capability **type** encoding this RTL's fixed check now
  rejects for MMIO, the first S-mode MMIO load faults (surfaced as access fault).

Either way: to run the borrow/revoke benchmark (and finally test the domain CALL) on
captype-fixed, we need **the firmware image that pairs with this bitstream** (the board owner has
a working setup on it), or the captype-fixed resident image, rather than our bundle.

## Dumper limitation found

The mtvec dumper's `putc` (poll LSR@0x14 bit 0x20, then `sw` THR) **hangs after ~2
chars on real UART timing** (prints `\n@`, then spins in the THRE poll at 0x80023f0a) —
so it never printed the hex. QEMU-inert but not HW-robust. The `--gdb-probe` CSR read
sidesteps it (reads mcause/mepc/mtval straight from the halted core, no reflash/reload).
If the dumper is reused, bound the poll or drop it.

## Driver additions (uncommitted, on capstone-bootstrap-b working tree)

- `--flash-bitstream NAME` (config.py `flash_bitstream` Http + fpga_console method) —
  non-volatile flash from the console store; power/settle ordering fixed.
- `--gdb-probe` — attach, `monitor halt` (no reset), read `$pc/$mcause/$mepc/$mtval`.

## Next

- Get the captype-fixed-paired firmware/DTB from the board owner (or identify the resident image),
  boot it, run borrow-cost → finally test the domain CALL on real Capstone silicon.
- If cap-type mismatch: our monitor needs recompiling against the captype-fixed toolchain.

Artifacts: `~/capstone-b-artifacts/board-run-captypefixed-pc-{fence,diag0}.uart.txt`,
`board-run-flash-captypefixed-fence.uart.txt`; logs in scratchpad.
