# Task 017 — Stage-0 diagnostic (domain-CALL reset): monitor trap-dumper + findings

**Date:** 2026-07-19
**Branch:** capstone-bootstrap-b
**Scope:** build a monitor-side trap dumper to localize the domain-CALL reset;
non-persistent gdb-boot runs. No SPI/flash, no submodule-source/RTL commits.

## What was built (Stage-0 instrumentation)

An M-mode `mtvec` dumper in the OpenSBI Capstone monitor. `cap_env_init` sets only
the cap trap vector `ctvec`; the standard M-mode `mtvec` is left at its reset default
`ROMBase+0x40` (the bootrom), and is **dormant during normal operation** (the system
boots to a shell fine, so no trap uses it). The dumper repoints `mtvec` to a small
handler that prints `mcause/mepc/mtval` over the ariane uart8250
(@0x10000000, reg-shift 2/width 4: THR +0x00, LSR +0x14) then halts — turning the
silent reset into a readable trap dump.

**Build obstacle solved:** the monitor C (`sbi_capstone_dom.c`, which `#include`s
`capstone-sbi/sbi_capstone.c`) is pre-compiled by the Capstone capability compiler into
a **checked-in** `sbi_capstone_dom.c.S` that this OpenSBI build assembles directly —
there is **no rule to regenerate it here** (`No rule to make target …c.S`) and no
capstone clang on PATH. So *every* monitor C change must be injected into the `.c.S`
directly. The dumper + the `mtvec` write were injected as raw asm into
`build/build/opensbi-custom/lib/sbi/sbi_capstone_dom.c.S` (using the `lla` idiom the
file already uses — `la` triggers a binutils `elfnn-riscv.c:2358` link crash here).
Verified in the ELF (`_diag_mtrap` @0x80023ec0, `csrw mtvec,t0`) and QEMU-regression
clean (boots to shell, `/dev/capstone` at boot, borrow `RESULT raw=2 borrow=6` — the
dumper is inert under QEMU, as designed). Image: `fw_payload_up_builtin_diag0.bin`.

## Findings (partial but load-bearing)

1. **The dumper fires on the board** — captured `@@MT 0000…` (marker + start of the
   MSB-first `mcause`).
2. **The trap is a SYNCHRONOUS EXCEPTION, not an interrupt** — `mcause` top 16 bits are
   `0000` (interrupt bit clear).
3. **The board GENUINELY HARDWARE-RESETS ~9 chars into the handler** — the dump is
   immediately followed by the bootrom banner (`@@MT 0000 Hello World!`). Since `mtvec`
   now points at the dumper (not the bootrom), reaching the bootrom is a **real reset**
   (frontend `npc` reset-load / `rst_ni`), fired a fixed short time after the M-mode
   exception — **not** merely the mtvec-default vector the RTL survey assumed. On stock
   RISC-V a synchronous exception traps and is handled; here it resets the core. This
   leans toward an RTL trap-delivery / reset fault around the domain switch.
4. **The same exception fires at the boot→S-mode handoff** (intermittently), which is
   the root of the recurring "reset-loop during login" that has plagued every board run
   (in the stock image it goes to `mtvec`=bootrom → reset; here the dumper catches it).

## Blockers (why the exception code is not yet read)

- The ~9-char reset truncates the dump before the exception code (low `mcause` bits).
  A **minimized, LSB-nibble-first** dumper (`@@<mcause_le>.<mepc_le>.<mtval_le>`,
  staged in the same image) is built to beat this — it would surface the exception code
  in the first nibbles.
- The board's JTAG DTM **wedged** after the repeated reset-loops/power-cycles
  (≥4 consecutive `gdb_start` attach timeouts). It needs recovery (auto-shutdown/idle)
  before the LSB-first image can be booted to capture the code. Lock released each run
  (good-citizen); board will auto-shutdown.

## Next

- When the board recovers: boot the LSB-first `diag0` image, capture
  `@@<mcause_le>.<mepc_le>` → read the exception code + faulting PC.
- In parallel (offline, no board): Stage-2 RTL compare — map the exception code to the
  RTL cap-violation causes 25-28 (`commit_stage.sv:205-229`) or an instruction-access
  fault, against the QEMU golden model (`op_helper.c helper_cscall`), to decide
  monitor-fix vs RTL. Finding #3 (real reset on a sync exception) is itself an
  RTL-leaning signal.

Artifacts: `~/capstone-b-artifacts/fw_payload_up_builtin_diag0.bin`,
`board-run-diag0{,b,c,d,e,f}.uart.txt`. Plan: `/home/alexey/.claude-b/plans/`.
