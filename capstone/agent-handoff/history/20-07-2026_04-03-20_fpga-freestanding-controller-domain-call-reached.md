# Freestanding controller clears the FP blocker — domain `cscall` reached, wedges at domain entry `0x10044`

**Date:** 2026-07-20
**Branch:** capstone-bootstrap-b
**Scope:** on `working-caplifive-captype-fixed.bit` (non-persistent gdb-boot), replace the
glibc benchmark controller with a freestanding soft-float one, get past the FP hang, and
observe what the domain `cscall` actually does on real Capstone silicon. Controller source +
board-driver work only; no submodule-source/RTL commits.

## Headline

**Two blockers, first cleared, second now cleanly localized.**

1. **FP/glibc blocker — FIXED and PROVEN on-board.** The benchmark's `.user` controller was
   a glibc Linux program, and glibc emits `fsd` (double-precision FP store). This
   `captype-fixed` bitstream's FPU **rejects `fsd` even with `mstatus.FS=Clean`** (JTAG:
   mcause=2 illegal, mepc in userspace, insn=`fsd`, FS=Clean; `libc.so.6` has 65 `fsd`
   incl. `_IO_vfprintf`, so the first `printf` traps → monitor `while(1)` → silent hang).
   Jason's steer ("use soft float") = **a freestanding controller that links no glibc**:
   `capstone/tests/rtl-smoke/borrow_cost_fpga_ctl.c`, built `-nostdlib -static -no-pie
   -march=rv64imac -mabi=lp64` → **zero FP instructions**. Own `_start` (inits sp AND gp),
   raw Linux syscalls via `ecall`, integer-only output, libcapstone's exact ioctl protocol.
   On-board it runs past **all** prior failure points: boots → shell → `insmod` OK →
   `created domain ID = 0` (the printf that used to trap now works) → `create_region` +
   `map_region` succeed for arena + results regions.

2. **Domain `cscall` reached for the first time — and it wedges at the domain's own entry.**
   Fixing (1) unmasked the original project blocker. Two faces observed:
   - **run 6** (foreground, user's web console): after the region setup the core resets to
     the **bootrom** (`Hello World! / Hit any key to enter update mode / init SPI / SD`).
   - **run 7** (keepalive held the socket, no drop; non-destructive gdb-probe of the parked
     core): `pc = 0x819a0044`, `mcause = mepc = mtval = 0` — the core is **executing in the
     domain**, not in the monitor trap handler.

## The decisive offline finding (no board needed)

The domain `.dom` links its `.text` at vaddr `0x10000` and the monitor places it at phys
`0x819a0000` (kernel log: `Domain memory region … paddr=819a0000`, `entry_offset=0`). So
run 7's `pc = 0x819a0044` = domain vaddr **`0x10044`**. Disassembling
`borrow_cost_fpga.dom`:

```
0000000000010000 <_start>:
   10000: ccsrrw sp, 0x4, zero      # read cap CSR 0x4 (stack cap)
   10004: lcc    t1, sp, 0x4
   10008: scc    sp, sp, t1
   1000c: delin  sp
   10010: j      0x10044 <test>
0000000000010044 <test>:            # <-- run7 pc = 0x819a0044 lands EXACTLY here
   10044: delin  gp
   10048: cincoffsetimm sp, sp, -0x60
   ...  save ra/a0/a1, then the .capstone_cap_init loop (cjalr to cap-global initializers)
   10170 <domain_main>              # the actual benchmark, further down
```

So the `cscall` **physically transfers fetch into the domain** (control lands at the domain
entry, the `j 0x10044` target from `_start`) and the core **wedges at the first entry
instruction** — before `domain_main`, before even the capability-global init loop. This is
the earliest possible domain execution. The switch datapath works; the domain's first
instructions do not make progress. `mcause=0` in the probe means the core is not *currently*
in an M-mode trap handler — it is sitting in the domain — so the post-hoc CSRs cannot tell
us whether executing `0x10044` raised (then reset from) a fault; the bootrom run overwrites
the CSRs. **A live trap dump is required.**

## Interpretation (unchanged verdict direction, now sharper)

This matches the RTL survey: the CVA6 domain switch does **no icache invalidate, no TLB
flush, no PMP reprogram**, and the monitor issues **no `fence.i`** at the domcall boundary.
The freshly-placed domain code at `0x10044` is fetched against a possibly-stale icache. Two
live hypotheses, distinguished only by the exception code:
- **stale-icache fetch fault** at `0x10044` (instruction-access / illegal) → a domain-boundary
  `fence.i` is the candidate fix (Stage-1A). QEMU models no icache, so this would only ever
  bite on silicon — consistent with the monitor passing the identical `cscall` under QEMU.
- **capability violation** (causes 25–28) at/after the switch → compare the RTL guard that
  fired against the QEMU golden model to decide monitor-fix vs RTL-bug.

## Current state vs the goal

- ✅ FP/glibc blocker fixed and proven on silicon (freestanding controller).
- ✅ Domain `cscall` reached and confirmed to **enter the domain** for the first time.
- ✅ Wedge localized to domain entry vaddr `0x10044` (`<test>` glue), offline, unambiguously.
- ❌ Exact fault mechanism at `0x10044` not yet read → needs a clean Stage-0 trap dump.
- ❌ No `RESULT` cycle numbers yet (the original deliverable), blocked on the above.

## Next step

Build a **combined** image = freestanding controller in the overlay **+** the Stage-0 M-mode
`mtvec` trap-dumper in the monitor (LSB-nibble-first, bounded THRE poll — the earlier
MSB-first/unbounded dumper truncated before the exception code and hung after ~2 chars). Boot
it, reach the `cscall`, capture `mcause/mepc/mtval` at the `0x10044` wedge, and branch per the
reproduction runbook §7 / the plan. The `diag0` dumper image predates the freestanding fix
and would hang at the fsd blocker before the domain call, so this is a fresh build.

## Artifacts / pointers

- Image: `~/capstone-b-artifacts/fw_payload_fpga_up_ctl.bin` (UP SMP=n, `--mode fpga`,
  freestanding controller baked into `overlay/root/rtl-smoke/`), sha `fe37ebdb`.
- Controller: `capstone/tests/rtl-smoke/borrow_cost_fpga_ctl.c`; builder
  `build-borrow-cost-fpga.sh`.
- Reproduction runbook: `ref/fpga-borrow-cost-reproduction.md`.
- Plan / staged ladder: `/home/alexey/.claude-b/plans/curried-crunching-gizmo.md`.
- Prior trail: `history/19-07-2026_19-55-15_fpga-mode-build-run.md` (fsd diagnosis),
  `history/19-07-2026_09-30-00_captype-fixed-flash-loadfault-mcause.md` (flash rules),
  `history/19-07-2026_02-54-00_fpga-domain-call-stage0-diagnostic.md` (dumper build, on the
  since-identified contaminated bitstream).
- Session run scripts (scratchpad, not committed): `run_ctl_image{5,6,7}.py`.
