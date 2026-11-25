# R-73's exposure: does the board's `monitor reset halt` reset the core? (board 2, 2026-11-25)

The read that R-73 (`docs/ref/ISSUES.md`, "Exposure") left open since 2026-11-14. The board lane ran it at the runtime
lane's request, in the gap before the first B5b-2 Linux boot. Board 2 was booked (hub booking 1499,
03:32-03:33 UTC) and nothing was flashed. The resident bitstream read back by name as
`caplifive_…-r75d-b5b2n_904cd9c6b.bit`. No image was loaded and the hart was never resumed.

## Why a CSR answers it

These hops are read at capstone-ariane 904cd9c6b:
- The core's reset is `ndmreset_n`. It is driven by `rstgen` from `pll_locked & ~ndmreset`
  (`corev_apu/fpga/src/ariane_xilinx.sv:322-327`), which feeds `i_ariane .rst_ni` (`:784`).
- In `core/cva6.sv`, that `rst_ni` reaches `csr_regfile` (`:2230`) and `wt_cache_subsystem` (`:2587`).
- `wt_cache_subsystem` passes it on to `wt_axi_adapter` (`core/cache_subsystem/wt_cache_subsystem.sv:240`).
- In the adapter, `rst_ni` sets `tbm_clr_q`: the page bitmap's 4096-cycle reset clear (`wt_axi_adapter.sv:357-360`).
  It also resets the TWC control (`p_twc_ctl`, `:519`).
- The same `rst_ni` zeroes `mscratch_q` and `mepc_q` (`core/csr_regfile.sv:3392-3418`).

So if `reset halt` takes a value written into mscratch and mepc back to 0, the core's `rst_ni` was asserted. Every
register above is reset by that same net. This is a source hop: the silicon reading is of the CSRs, not of the bitmap.

## Procedure (`run_sup_bare-r73.py`, sha256 prefix a5bddf6435e7d0aa)

The predictions are in the script's header, written before the run. It ran from the board-3-capable driver tree
(`<mnt-hdd>/capstone/wt-b5b2drv` at 5980866f9f65), under a wrapper that does four things in order:
- checks the board is free;
- starts the busy holder;
- books the board for 15 min;
- unbooks.

The script first does the bare runner's own power-cycle and gdb attach. Then:
1. It records the configuration: `version`, `targets`, `reset_config`, and the `reset-assert`, `reset-assert-pre` and
   `reset-deassert-post` events.
2. `reset halt`, then pc, mscratch and mepc, each read with `force`.
3. It writes mscratch = 0x5a5a5a5a12345678 and mepc = 0x80001234 and reads both back. This is the control.
4. `reset halt` again, then the same three reads.

Before the run, the script was dry-run on a fake console, and all three verdict branches fired: RESET, NO RESET, and
VOID when the control's write is dropped.

## Result lines

```
monitor version              Open On-Chip Debugger 0.12.0
monitor targets              0* riscv.cpu riscv little riscv.cpu halted
monitor reset_config         none separate
cget -event reset-assert / reset-assert-pre / reset-deassert-post: empty (no handler)
reset halt #1                pc 0x10000  mscratch 0x0  mepc 0x0
write + read back (control)  mscratch 0x5a5a5a5a12345678  mepc 0x80001234
reset halt #2                pc 0x10000  mscratch 0x0  mepc 0x0
R73 VERDICT RESET: reset halt resets the core (ndmreset asserted)
```

## Reading

`reset_config none separate` means no SRST is configured, and no reset event overrides the default. OpenOCD's
riscv-013 target therefore resets through `dmcontrol.ndmreset`. On silicon, that reset takes the control's values
back to 0 and leaves the hart halted at the boot ROM (0x10000).

Every board boot path runs `reset halt` before `load_image` and resumes only after the last load. That covers:
- run_sup_bare.py;
- run_base_bare_fpga.py;
- run_ladder_base_fpga.py;
- run_ladder_perf_fpga.py;
- run_rtl_smoke.py's `boot_via_gdb`.

Every path but `boot_via_gdb` also power-cycles first (`boot_via_gdb` itself was not checked for a power-cycle), and
the `reset halt` alone is enough. So when the JTAG load writes DRAM, every page is unmarked in the bitmap
and the TWC holds nothing. As R-73's Exposure section argues, that makes the exposure nil for this flow, whether the
load goes through the system bus or the program buffer.

What this does not cover:
- a `load_image` issued after the core has run without an intervening reset;
- any write from the debug module's system-bus master to a running system (a gdb memory write into a live domain,
  for instance).

The defect itself stays as R-73 records it.
