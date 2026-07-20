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
- ✅ **Exact fault mechanism CONFIRMED on silicon** (see ROOT CAUSE below): `delin gp` with
  `gp=0` stalls the CVA6 pipeline; `gp` is never delivered valid at the FPGA `cscall`.
- ❌ No `RESULT` cycle numbers yet (the original deliverable) — blocked on the `gp`-delivery
  fix, which is monitor/RTL-owner (Jason) territory.

## Next step

Decide the `gp`-delivery fix with Jason (monitor context vs domain `start.S` vs RTL `delin`);
options (a)/(b)/(c) in ROOT CAUSE. Fastest local test: option (b) — have the domain `start.S`
`test:` derive a valid `gp` from the domain's data cap before `delin gp`, rebuild the `.dom`,
and re-run; if it reaches `domain_main`, the diagnosis is proven and the borrow/revoke sweep
can finally produce cycle numbers. (start.S is a shared file / A's lane — coordinate before
editing.) Jason also suggested board **Trace Dump** + reproducing a **reference example
domain** as cross-checks, and bare-metal launch since borrow-cost is self-contained.

## ROOT CAUSE CONFIRMED (2026-07-20, later same session): `delin gp` with `gp=0` stalls the CVA6

Follow-on board probes (single-step + register read at the wedge, on freshly re-flashed
`captype-fixed`) pin the mechanism exactly:

1. **The `mtvec` dumper stayed silent; `$mcause=$mepc=$mtval=0`.** No M-mode trap. So the
   wedge is NOT an M-mode exception — on real Capstone silicon an in-domain fault routes to
   the capability trap vector `ctvec`, not `mtvec`. (The earlier `@@MT` dump only ever fired
   on the *contaminated stock-Ariane* bitstream, which has no cap unit → faults go to M-mode.)
   Jason confirmed `cscall`/`csreturn` **implicitly flush the icache**, so the stale-icache /
   `fence.i` hypothesis is dead.
2. **Single-stepping does not advance:** 40× `stepi` from `0x819a0044` leaves pc pinned there,
   no trap. The instruction at `0x10044` = **`delin gp`** cannot retire.
3. **Register read at the wedge:** `gp = 0x0` (null/untagged), `sp = 0x819c0000` (valid).
4. **Skipping the `delin` (set pc to `0x10048`) lets the domain run:** pc advances cleanly
   `0x48→0x4c→0x50→0x54→0x58→0x60`. So `delin gp` is *the* stalling instruction; everything
   after it executes.
5. **QEMU cross-check:** `helper_csdelin` (`capstone-qemu op_helper.c:871`) does
   `assert(rd_v->tag)` — it requires a **tagged** operand. Since this exact `borrow_cost_fpga.dom`
   passes under QEMU (`RESULT raw=2 borrow=6`), `gp` must arrive **tagged/valid** at
   `delin gp` under QEMU. On the FPGA it arrives `0`.

**Why `gp` is 0:** the domain `start.S` (`my_first_domain/start.S`) sets up **`sp`** in
`_start` (reads it from `cscratch`/cap-CSR `0x4`) but **never initializes `gp`** before
`test:`'s `delin gp`. `gp` is whatever `cscall` leaves in x3, i.e. the domain's saved-context
`gp` slot. The monitor's `create_domain` (`sbi_capstone.c:279`) builds the sealed context as
`dom_seal[0]=dom_code` (PCC), `dom_seal[2]=dom_data`, `dom_seal[3]=priv`, and **zeroes all
other slots** — so the `gp` slot is 0. Under QEMU's `cscall` the domain still gets a valid
`gp` (examples pass); on the FPGA `cscall` it lands 0.

**Verdict:** an RTL/QEMU divergence at the domain-entry `gp` delivery. `gp` is not delivered
as a valid capability to the domain at `cscall` on the FPGA (arrives 0), and this CVA6's
`delin` **stalls the pipeline on a null/untagged operand** (no retire, no trap) where QEMU
asserts/handles it. **This is the domain-CALL blocker, fully localized.**

**Fix options (for the monitor/RTL owner — Jason):**
- (a) **Monitor:** deliver a valid `gp` capability in the domain's initial sealed context
  (e.g. `dom_data`) so `cscall` restores a tagged `gp` before `delin gp`.
- (b) **Domain `start.S`:** initialize `gp` from the domain's data/code cap (as `sp` is
  initialized from `cscratch`) *before* `delin gp`, so it never delins a null.
- (c) **RTL:** make `delin` of an untagged/null operand a no-op or a clean trap instead of a
  pipeline stall (the "correct" fix; needs an out-of-tree bitstream rebuild).

The exact `dom_seal[i]`→register mapping (whether `gp` is *meant* to come from `dom_seal[2]`
and the FPGA `cscall` simply doesn't restore x3) is the one open detail — a question for
Jason, since it decides monitor-fix (a) vs RTL-fix (c).

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
