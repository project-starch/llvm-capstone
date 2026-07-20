# Freestanding controller clears the FP blocker — domain `cscall` reached, wedges at domain entry `0x10044`

**Date:** 2026-07-20
**Branch:** capstone-bootstrap-b
**Scope:** on `working-caplifive-captype-fixed.bit` (non-persistent gdb-boot), replace the
glibc benchmark controller with a freestanding soft-float one, get past the FP hang, and
observe what the domain `cscall` actually does on real Capstone silicon. Controller source +
board-driver work only; no submodule-source/RTL commits.

---

## LATEST (2026-07-20 PM) — gp-seed experiment run on-board; HANDOFF TO AGENT A

**Agent-B is handing the FPGA experimental lane to Agent A (token refresh).** Full state below;
the gp-delivery design + code is captured so A can iterate without re-deriving it.

**What we built + ran (2 board runs, captype-fixed, no reflash — it was still resident):**
- **Monitor** (`create_domain`, patch in `agent-handoff/patches/fpga-gpseed-monitor-createdomain.patch`):
  mint a non-linear alias of the domain image, park an image-covering data cap in the
  **ctvec slot `dom_seal[1]`** (which the reference monitor leaves 0), to seed `gp`.
- **Domain** (`my_first_domain/start-fpga-gpseed.S`, tracked): `_start` loads it via
  `ccsrrw(gp, ctvec, x0)` before `delin gp`. Built with `START_SRC=` that variant.
- Ran with the seeded cap's cursor = **0** and cursor = **base** (representable).

**Result — the delivery does NOT work on this RTL (both cursor values identical):**
- pc pinned at `0x819a0044` (`delin gp`), **`gp = 0x0`**, 19/20 single-steps pinned — same wedge.
- A cursor=base cap would read `p/x $gp = 0x819a0000`; it read **0** → **the cap never reached
  `gp`**. `sp` (via cscratch) and PCC arrive fine; **`ctvec` arrives 0**. So the fast
  first-entry (c-effective) domain-call path on this RTL **does not restore `ctvec`** for the
  entered domain — a **delivery** failure, not a representability result.

**IMPORTANT — this walks back the earlier "cursor-0 non-representable, confirmed" verdict**
(in the CORRECTION/Fix sections below and in commit `ef521aef`). Representability is **still
UNTESTED**: we never got a tagged `gp` into the domain to test it. What we newly learned is a
**delivery constraint**: the monitor can't hand a first-entry domain a capability via `ctvec`.

**Open question → the collaborator** (kept out of the repo, in `/tmp/capstone/gp-representability-question.md`):
which context slot (if any) the first-entry domain call restores for seeding `gp`, and what the
intended mechanism is for a domain with globals to obtain a cap covering them.

**Next options for A (see the runbook §7 + the AskUserQuestion trail):**
1. **Read the RTL** (`caplifive-cva6`) first-entry context layout to find a slot that IS
   restored (no board time), then re-target the delivery.
2. **Stack-memory delivery**: monitor writes `gp_cap` into the domain's stack region, `start.S`
   loads it via `sp`/`ldc` (the proven cscratch path) instead of ctvec.
3. **Compiler/canonical fix** (robust end state, A's lane): stop needing a delivered `gp` —
   address globals via a representable scheme (cursor=base + link-relative offsets), retiring
   the QEMU hack `7aca0540`.

**Artifacts:** `~/capstone-b-artifacts/fw_payload_fpga_up_gpseed.bin` (cursor=base build, sha
`3cc33379`); build recipe `scratchpad/rebuild_gpseed.sh`; board run `scratchpad/run_gpseed.py`
(does a resident-bitstream double-check + gdb single-step probe). QEMU-first verification of the
delivery mechanism (boot the patched monitor with the hack disabled, read `gp`) was proposed but
not yet run — a cheap way to confirm mechanism-vs-RTL before more board time.

---

## Headline

**Two blockers, first cleared, second now cleanly localized.**

1. **FP/glibc blocker — FIXED and PROVEN on-board.** The benchmark's `.user` controller was
   a glibc Linux program, and glibc emits `fsd` (double-precision FP store). This
   `captype-fixed` bitstream's FPU **rejects `fsd` even with `mstatus.FS=Clean`** (JTAG:
   mcause=2 illegal, mepc in userspace, insn=`fsd`, FS=Clean; `libc.so.6` has 65 `fsd`
   incl. `_IO_vfprintf`, so the first `printf` traps → monitor `while(1)` → silent hang).
   The collaborator's steer ("use soft float") = **a freestanding controller that links no glibc**:
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
- ✅ **Fix localized to OUR domain runtime** (`my_first_domain/start.S` + clang codegen),
  **NOT the RTL** — corrected 2026-07-20 after the collaborator's answer (see CORRECTION below).
- ❌ No `RESULT` cycle numbers yet — but the fix is in-repo (no bitstream rebuild needed).

## CORRECTION (2026-07-20, the collaborator's answer) — the fix is OURS, not the RTL

**My first verdict below ("fix is necessarily RTL") was WRONG.** the collaborator confirmed the
`gp = PCC(cursor 0)` line in QEMU's `helper_cscall` is **our own non-canonical patch**,
not canonical Capstone: commit `7aca05403dc52644072df84ad53b32cf17b9810f`
("riscv: unblock native domain capability calls", Alexey Paznikov, 2026-05-19). So:

- The **RTL is correct** to not set `gp` at `cscall` — canonical Capstone never does.
- The collaborator also notes the approach isn't representable anyway: "it's unlikely going to be
  representable when you keep the bound but set the cursor to 0."
- Therefore **the gap is in OUR domain runtime**, which was written to depend on that
  non-canonical, non-representable `gp` — and our QEMU patch masked it.

**What the canonical reference domains actually do** (verified against the buildroot
example domains `capstone-test-domains`: `fib.dom.S`, `thread.dom.S`, `smode.dom.S`,
all emitted by the same capstone-cc compiler):

- They **never establish or use a `gp` data/code capability.** There is **no `delin gp`,
  no `.capstone_cap_init` loop, no `cincoffset gp,<abs>`** anywhere in them.
- Code is addressed **pc-relative** (`lla`), fetched through the **implicit PCC** which
  covers the domain's code.
- Data capabilities come from (1) the **stack cap** read out of `cscratch`
  (`ccsrrw sp, cscratch, x0`) and (2) **passed-in capability arguments**
  (e.g. `stc a1,…` then `ldc … ; sd through it`).
- `gp` is treated as an **opaque caller register**: saved on `domreturn`
  (`stc gp, sp, -16`), zeroed (`li gp, 0`), restored on reentry (`ldc gp, sp, -16`).
  It holds the *host's* gp and is never used inside the domain.

Our `my_first_domain/start.S` invented `delin gp` + the `.capstone_cap_init` loop +
`cincoffset gp,<absolute>` to support clang-compiled C with capability globals via a
domain-wide cursor-0 `gp` cap — a model that only worked under our QEMU patch and is not
representable on silicon.

**And for the borrow-cost domain specifically it is pure dead weight:**
`readelf` of `borrow_cost_fpga.dom` shows `.capstone_cap_init` is **size 0**
(`__capstone_cap_init_start == __capstone_cap_init_end == 0x10690`) — it has **zero
capability globals**, so the entire gp machinery does nothing useful, yet `delin gp`
runs unconditionally at the top of `test:` and stalls the core.

## Fix (in-repo, no bitstream rebuild)

- **Minimal / benchmark-unblocking:** for domains with an empty `.capstone_cap_init`
  (borrow_cost qualifies), don't emit `delin gp` / the cap_init loop, and call
  `domain_main` within PCC (pc-relative) instead of via a `gp`-constructed code cap —
  i.e. make our entry match the canonical compiler output. This should run on real
  silicon and finally yield cycle numbers.
- **General / correct:** rework the clang `CapstoneCapGlobalInit` codegen + `start.S` so
  capability-global domains address code/globals **without** a cursor-0 domain-wide `gp`
  (pc-relative code within PCC; a *representable* scheme for capability globals), and
  **retire the QEMU hack `7aca0540`** so QEMU and silicon agree. This lands in A's lane
  (shared `start.S` + compiler); scope + coordinate.

--- The ROOT CAUSE section below is retained for the audit trail. Its board evidence
(single-step pins at `delin gp`, `gp=0`, skip-delin runs) is STILL VALID and correct.
Only its closing verdict ("fix is necessarily RTL") is WRONG and is superseded by the
CORRECTION + Fix sections above. ---

## ROOT CAUSE CONFIRMED (2026-07-20, later same session): `delin gp` with `gp=0` stalls the CVA6

Follow-on board probes (single-step + register read at the wedge, on freshly re-flashed
`captype-fixed`) pin the mechanism exactly:

1. **The `mtvec` dumper stayed silent; `$mcause=$mepc=$mtval=0`.** No M-mode trap. So the
   wedge is NOT an M-mode exception — on real Capstone silicon an in-domain fault routes to
   the capability trap vector `ctvec`, not `mtvec`. (The earlier `@@MT` dump only ever fired
   on the *contaminated stock-Ariane* bitstream, which has no cap unit → faults go to M-mode.)
   The collaborator confirmed `cscall`/`csreturn` **implicitly flush the icache**, so the stale-icache /
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

**Where `gp` is supposed to come from — traced to the exact QEMU line.** QEMU's
`helper_cscall` (`op_helper.c:1227-1231`) sets, on domain entry:
```c
if (env->priv == PRV_C) {
    capfat_t gp_cap = env->pc_cap;   // gp := the domain PCC
    gp_cap.bounds.cursor = 0;        //   with cursor forced to 0
    capregval_set_cap(&env->gpr[3], &gp_cap);
}
```
i.e. **`gp` is established by the `cscall` instruction itself**, *after* the context restore
(`swap_c_effective_regs`, 1225) — not from any saved slot. The FPGA `cscall`
(RTL `capstone_dom_switcher`) omits this step → `gp=0`.

**Both software workarounds are INFEASIBLE (investigated + ruled out 2026-07-20):**
- (a) **Seed the monitor context** — NOT possible: the first-entry sealed context uses the
  **c-effective layout** (`swap_c_effective_regs`, `capstone_helper.c:190`): PCC, `ctvec`,
  `cscratch`, `mstatus`, `mideleg/medeleg/mip/mie` — **no GPRs**. `create_domain`'s
  `dom_seal[0]=PCC`, `dom_seal[2]=cscratch(=dom_data→sp)`, `dom_seal[3]=mstatus`. There is
  **no `gp` slot** to seed; `gp` is not restored from context on first entry, it is set by
  `cscall` itself (the QEMU code above).
- (b) **Fix it in the domain `start.S`** — NOT possible: the domain has **no way to obtain a
  cursor-0 code capability**. There is no PCC-read instruction (cap-CSRs are only
  `ctvec/cih/cepc/cscratch`), and the caps the domain *does* hold (`sp`=`dom_data` bounded to
  the data region; `a1`=region cap) don't cover the code at cursor 0. `sp.base ≈ 0x819c0000`,
  so moving its cursor to 0 is far out of bounds → not representable → untagged → `delin`
  stalls again.

**Therefore the fix is necessarily RTL (option c):** the FPGA `cscall` must initialize
`gp = PCC(cursor 0)` on domain entry, exactly as `op_helper.c:1227-1231`. Needs an
out-of-tree bitstream rebuild (Vivado + anvil), so it is an escalation to the collaborator, not fixable
here. The FPGA cycle-accurate borrow/revoke numbers are **blocked on this RTL fix.**

**One thing for the collaborator to confirm (representability):** QEMU keeps exact bounds in a side
table, so `gp=PCC(base 0x819a0000, cursor 0)` round-trips losslessly there; on real hardware,
a cursor 2 GB below `base` for a small code region may not be representable (tag cleared). So
either the domain PCC is actually wide/base-0 on real HW, or the domain addressing model
(`cincoffset gp,<absolute>` with `gp.cursor=0`) needs confirming for silicon. This is the
same class as "what is `gp` supposed to be at first `cscall`."

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
