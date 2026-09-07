# Plan: CHERI-CVA6 on our Genesys2 — bring-up, bitstream, and a like-for-like comparison with Capstone-CVA6

**Status: PROPOSED 2026-09-07 — for the project lead's review before any build or board time.**
Owner: the `cheri` lane (this plan). Synthesis: the synth lane (Vivado is not on the dev host).
Board: the board lane, serialised with the SLT corpus campaign; **flashing is ask-first, always.**

## 1. Goal

Take `zero-day-labs/cheri-cva6` — a CHERI-RISC-V extension of the same CVA6 core our
Capstone-CVA6 is built on — and answer, in this order:

1. **Does it build for and run on our board** (Digilent Genesys2, Kintex-7 `xc7k325t`, the board
   the resident `caplifive_s12fix_5097eb166.bit` occupies)?
2. **If yes, at what hardware cost**, on the same device, same period, same flow: LUTs/FFs/BRAM,
   post-route WNS and failing-endpoint census — set beside vanilla CVA6 and beside Capstone-CVA6.
3. **Then everything else that can be compared on one board**: the ISA and security model,
   the software stack and toolchain, and the same benchmarks (CoreMark, BEEBS, RV8) on both
   cores, with the plain-RISC-V build of each benchmark as the common baseline.

This is a *comparison* plan, not an adoption plan. Nothing here changes Capstone. The paper
already carries a CHERI security table and a QEMU-to-QEMU performance plan
(`perf-cheri-vs-capstone-qemu.md`, direction set 2026-07-14: *"do not compare CHERI-QEMU
against Capstone-RTL — that's incomparable"*). What silicon adds is the axis that QEMU cannot
give: **real hardware cost and real cycles on the same FPGA**, which is exactly the comparison the
QEMU plan declined to make because the vehicles differed. Same board, same period, same flow
removes that objection.

## 2. What the reconnaissance established (2026-09-07, all read from the sources named)

Repository: `~/capstone-artifacts/ext/cheri-cva6` (shallow clone; default branch `cheri-cam`
at `9a95da4`, 2025-04-02; branches `main`, `cheri-cam`, `vcu118`; upstream-CVA6 fork, CHANGELOG
still upstream's).

| fact | evidence |
|---|---|
| **ISA revision = CTSRD CHERI-RISC-V (the "v9" encoding family)**, not the 2024+ standard `Zcheri*` encodings despite the config flag names | `core/include/cva6_cheri_pkg.sv`: "Adapted from the CHERI Capability Library (CTSRD cheri-cap-lib)"; `CAP_OTYPE_WIDTH = 18` (RV64), `CAP_HPERMS_WIDTH = 12`, `CAP_E_WIDTH = 6`, `CLEN = 2*XLEN` + 1 tag bit; decoder handles `OpcodeCheri = 7'b1011011` (`riscv_pkg.sv:263`, `decoder.sv:1532`) with `CSEAL`/`CUNSEAL`/`CINVOKE`/`CJALR` |
| **Our existing CHERI-LLVM SDK targets exactly that ISA** | `~/cheri/output/sdk/bin/clang` = clang 17 from CTSRD `llvm-project` `7e122876ee`; `-target riscv64-unknown-elf -march=rv64imafdcxcheri -mabi=l64pc128` compiles purecap (emits `cincoffset ca0, ca0, a1`), `-mabi=lp64d` compiles hybrid/plain. Built for the QEMU CHERI baseline (`tests/cheri-baseline/`), with a CheriBSD purecap image and rootfs beside it |
| **Config**: `cv64a6_imafdchzcheri_sv39` — purecap + hybrid both on, 1-bit tag carried as `DataUserWidth`; the fork's `Makefile` defaults `target` to it | `core/include/cv64a6_imafdchzcheri_sv39_config_pkg.sv:30-42`; `Makefile:99-100` |
| **Tags need storage behind the caches.** The core carries the tag on the AXI user bits; a DDR MIG does not store user bits, so an FPGA build needs the tag controller (a tag cache backed by a reserved DRAM region) in the SoC top | `corev_apu/cheri_tag_mem/cheri_tag_mem.sv` (tag cache: `DRAM_TAGS_BASE`, `DRAM_LENGTH`); controller `axi_tagctrl_reg_wrap` from the submodule checked out at `vendor/zero-day/axi_tagcontroller` (`.gitmodules:54`, reachable) |
| **On `cheri-cam` the controller is only in the testbench**, not in the FPGA top | `corev_apu/tb/ariane_testharness.sv:522` (`gen_cheri_tag_controller`, tag region at `0xA0000000`); `corev_apu/fpga/src/ariane_xilinx.sv` has only the boot capability (`:768-783`) and ties user bits to 0 |
| **The `vcu118` branch is the one with FPGA tag support**, and it keeps Genesys2 in the flow | 3 commits on top of `9a95da4`: "add vcu118 fpga support", "add second uart", "minor fixes"; 23 files, incl. `ariane_xilinx.sv`, `run.tcl`, `genesys-2.xdc`, `vcu118.xdc`, an AXI 512→64 dwidth converter and a DDR4 MIG; it adds `TagCacheMemBase = 0xBFF00000` to the FPGA top (top 1 MiB of a 1 GiB DRAM window → tags for the first 128 MiB); `run.tcl` still selects `genesys-2.xdc` and `src/genesysii.svh` for `BOARD=genesys2`, and the top keeps the `GENESYSII` DDR3 port list. **Confirmed the same day:** in that branch's `ariane_xilinx.sv` the `axi_tagctrl_reg_wrap` instantiation (`:1076-1163`) sits on the common DRAM path between the DRAM AXI slice (`.slv(master[ariane_soc::DRAM])`, `:1036-1040`) and the MIG-side bus, outside every board `ifdef` (the only conditional nearby is `PROTOCOL_CHECKER`, `:1110`), with `DRAMMemBase = 0x8000_0000`, `CapSize = 128`, an 8-way/128-line/4-block tag cache (`:213-222`). So Genesys2 gets the tag controller with no RTL edit; the tags live as ordinary data in DRAM, so the DDR3 MIG needs no user-bit path. |
| **Genesys2 DRAM window matches**: 1 GiB at `0x8000_0000` | fork `corev_apu/tb/ariane_soc_pkg.sv:51,67` (1 GiB variant), `TagCacheMemBase = DRAMBase + DRAMLength - TagCacheMemLength` (`:71`) |
| **Verification the fork ships**: a TestRIG/RVFI-DII harness against the Sail model (the CHERI reference method), Verilator-based | `corev_apu/tb/tb_testRig_cheri/` (Makefile, `ariane_testharness_dii.sv`, `cva6_dii_toplevel.cpp`); the CHERI target has no directed tests under `verif/tests/custom` |
| **Our synthesis runs elsewhere**: Vivado is not installed on the dev host and no Vivado image exists in its Docker; synthesis happens on the synth machine inside its container via `synth-guard.sh` (40 GB ceiling; a healthy run 2 h 23 m, 13 GB PSS) | `capstone-ariane/synth-guard.sh:14-60`; `docs/state/BRANCH-INVENTORY.md:31` |
| **Our baseline numbers to compare against**: Capstone-CVA6 places at 169–171 k of 203.8 k LUTs (83–84 %) and has never met timing at the 40 ns period (25 MHz): WNS −10.6 … −14.1 ns, 93–104 k failing endpoints; the census, not the slack, is what makes a bitstream usable | `docs/ref/bitstream-usability-is-the-census-not-the-slack.md`; `docs/history/20-08-2026_12-30-00_…md:310`; `docs/history/26-08-2026_03-17-36_…md:25` |
| **Bare-metal purecap runtime**: not built yet; cheribuild has produced `bbl-baremetal-riscv64-purecap`, `cheribsd-riscv64-purecap`, `qemu`, `llvm-project` under `~/cheri/build` — the newlib bare-metal purecap target is the missing piece for CoreMark/BEEBS/RV8 | `ls ~/cheri/build` |

Not established, and to be read rather than assumed: the fork's own published utilisation/timing
(the group's paper on CHERI-CVA6 — cite once found; nothing in the tree carries numbers); the
Vivado version the fork's IP scripts assume versus the synth machine's; whether the Genesys2 DDR3
MIG path carries the AXI user bits into the tag controller or needs the same dwidth-converter
treatment the VCU118 path got.

## 3. Phases, each with its gate and its predicted reading

### Phase 0 — a synth-ready tree (dev host, no board, no lock; ~half a day)

- Full clone (not shallow) of `cheri-cva6` at `vcu118`, submodules included
  (`axi_cheri_tagcontroller`, `hpdcache`, `cvfpu`, the APU peripherals), into
  `~/capstone-artifacts/ext/` — outside the repo, like every external tree.
- ~~Answer the tag-controller question~~ — answered (§2): the controller is on the common DRAM
  path, no RTL edit. What remains to read end to end is the Genesys2 clock/reset and the DDR3
  MIG instantiation on that branch, since the branch also regenerated the VCU118 IP scripts.
- Produce the exact `make fpga XLEN=64 BOARD=genesys2` recipe, the file list, and the Vivado
  version assumption, in a README next to the tree, so the synth lane can run it blind.

Gate: the Verilator model builds (`target=cv64a6_imafdchzcheri_sv39`) on the dev host —
containers per the resource rules (`--cgroup-parent=docker.slice --memory=16g`, at most three,
`NUM_JOBS=4`). Note the fork's `cva6.py` still hard-gates the Verilator version; use the fork's
own `verif/regress/install-verilator.sh` inside the container, not ours.

### Phase 1 — functional gate in simulation (dev host; ~1 day)

Before a single Vivado hour: prove the CHERI core executes capability code and enforces it.

1. Upstream `riscv-tests` under the CHERI config (hybrid mode runs plain RV64 code): the
   ISA sanity floor.
2. A purecap smoke program from our CHERI-LLVM 17 (`-mabi=l64pc128`): tagged store/load
   round trip through the cache into the testbench's tag memory, `csetbounds`, one
   **out-of-bounds load that must trap** (the positive control — a run in which nothing
   traps proves nothing), and `cinvoke` on a sealed pair.
3. Optional, only if 1–2 are green and the harness builds: the fork's TestRIG run against Sail
   for a few thousand instructions — it is the fork's own acceptance method.

Predicted reading: 1 and 2 pass on the testbench, since that is what the fork's authors used.
A failure here is a fork-vs-toolchain revision mismatch (the CTSRD ISA has moved several times;
our SDK is a July-2025 tree, the fork's cap-lib is January-2025) and is a finding, not a blocker
to phase 2 — the hardware cost can be measured on RTL that runs plain RISC-V.

### Phase 2 — synthesis and the hardware-cost comparison (synth machine; ~3 runs × 2–3 h)

Three builds, **same device, same 40 ns period, same `run.tcl` settings (retiming as in our flow,
untouched)**, each under `synth-guard.sh`, each with `collect-synth-artifacts.sh` so the census
exists whether or not it meets timing:

| build | source | what it is for |
|---|---|---|
| A | vanilla CVA6 at the fork's own base (`cv64a6_imafdc_sv39`) | the zero point; also checks the fork's flow against the synth machine's Vivado |
| B | CHERI-CVA6 `vcu118` @ `BOARD=genesys2`, `cv64a6_imafdchzcheri_sv39` | the subject |
| C | Capstone-CVA6, resident `5097eb166` | already measured; re-collect only if A/B use a different Vivado than it did |

Readings written down now, so the runs decide something:

- **A**: the numbers upstream reports for Genesys2 are ~45–55 % LUTs and timing met at 50 MHz;
  at our 25 MHz it should be comfortable. If A fails timing at 25 MHz, the flow (or the Vivado
  version) is the variable, not the design, and B/C cannot be read until that is fixed.
- **B**: a 128-bit capability datapath, the capability ALU/bounds logic in issue/execute/LSU,
  wider caches (129-bit lines) and the tag cache. Expected **+25–40 % LUTs over A** (i.e. roughly
  60–75 % of the device) and **timing met at 25 MHz**, plausibly at 50 MHz.
- **The comparison that matters**: Capstone-CVA6 sits at 83–84 % of the device and fails timing on
  ~100 k endpoints at 25 MHz; if B lands where predicted, the headline is that CHERI's
  spatial-safety hardware costs a third of the core and closes timing, while Capstone's linear
  capabilities plus revocation cost the whole device and do not — and the honest follow-up is
  *why*: how much of Capstone's cost is the design and how much is the S-07/S-10/S-12 fixes and
  the observability tree (§7a measured that tree alone at +750 LUTs and −1.82 ns). If B instead
  fails timing like ours, the comparison reads the other way and is equally worth having.

Deliverable: one table in `docs/ref/fpga-silicon-measurements-for-paper.md` (LUT/FF/BRAM/DSP,
WNS, failing endpoints, the census verdict) for A/B/C, plus the routed reports archived under
`~/capstone-artifacts/`.

### Phase 3 — the board (board lane; ASK-FIRST; ~1 board session)

- **Program the CHERI bitstream volatile-only** (JTAG, no SPI-flash write): the console's
  `flash-bitstream` supports `volatile`, so a power cycle restores `caplifive_s12fix_5097eb166`
  and nothing of the Capstone campaign is put at risk. Only if the lead later wants CHERI resident
  does a non-volatile flash happen, and that is a second ask.
- Boot path: the fork's bootrom then an ELF loaded over the debug module exactly as our driver
  does (`load_image` at `0x80000000`); the board driver's console/GDB plumbing is core-agnostic.
  UART on the same pins; the second UART the branch adds is optional.
- Programs, in the batching-and-ordering discipline of the `board-run` skill: a plain-RV64 control
  first (its host oracle known), then hybrid, then purecap variants; one expected-to-fail arm last.

Gate: the plain-RV64 control returns its oracle on CHERI-CVA6 silicon. Then purecap. Then the
security demonstrators (out-of-bounds trap, tag-clear on integer write, sealed-call), each with its
predicted reading written before the boot.

### Phase 4 — benchmarks on both cores (dev host + board; ~2 board sessions)

Same sources, same optimisation levels, three builds per benchmark on the CHERI core (plain
RV64 / hybrid / purecap) and two on ours (plain RV64 / Capstone domain), `mcycle` read the same
way on both:

- CoreMark; the BEEBS subset our silicon ladder already uses; RV8 (aes, primes, …).
- Reported as **cycles relative to the plain-RV64 run on the same core**, so each core's
  capability overhead is a ratio against its own baseline, and the two ratios are comparable
  without comparing absolute cycles across different microarchitectural states.

Needs the bare-metal purecap runtime: cheribuild `newlib-baremetal-riscv64-purecap` (+ the
`freestanding` SDK), built once on the dev host under a `systemd-run` memory scope.

**Explicitly out of scope for this plan**: SQLite / the SQLLogicTest corpus on CHERI silicon.
Ours runs inside Linux under the Capstone monitor; the CHERI equivalent is CheriBSD purecap on
the fork's SoC, which needs SD/Ethernet bring-up and a CheriBSD kernel port to that SoC — weeks,
not days. The QEMU-side CHERI baseline already covers the SQLite security story; silicon adds
nothing there that phase 4's benchmarks do not add more cheaply. Revisit only if phases 2–4 land
and the lead wants it.

### Phase 5 — the comparison write-up

One section in the measurements doc, one table per axis:

| axis | Capstone-CVA6 | CHERI-CVA6 | how measured |
|---|---|---|---|
| capability model | linear capabilities, revocation nodes, domains, monitor in M-mode | monotonic capabilities, sealing, tags, hybrid+purecap ABIs, no revocation in hardware | spec vs `cva6_cheri_pkg.sv` |
| temporal safety | revoke-at-free as an O(1) hardware op | none in hardware (Cornucopia-style sweeps are software; the QEMU plan's "eager" config) | existing security table |
| hardware cost | 83–84 % LUTs, timing not met | phase 2 | same device, period, flow |
| software stack | our LLVM fork + monitor + Linux + Capstone-C | CTSRD CHERI-LLVM 17, bare-metal purecap (CheriBSD only under QEMU) | what phase 3–4 ran |
| per-benchmark overhead | domain vs plain, our ladder | purecap/hybrid vs plain, same sources | phase 4, `mcycle` |
| verification method | our sweep + directed tests + board rungs | TestRIG vs Sail | phase 1 |

## 4. Resources, rules, and who does what

- **Dev host** (this lane): clones, Verilator, toolchain work. Containers: `≤3`, each
  `--cgroup-parent=docker.slice --memory=<cap>`; anything heavy outside a container under
  `systemd-run --user --scope -p MemoryMax=…` and the machine-wide memory lock
  (`~/bin/logs/AGREED-RESOURCE-RULES.md`). No QEMU lock needed by this plan.
- **Synth machine** (synth lane): phase 2, three runs, under `synth-guard.sh`. Handed as a
  branch/tree + a README with the exact command; the receiving lane refuses a tree without its
  lint numbers.
- **Board** (board lane + the lead): phases 3–4, serialised after the SLT corpus campaign;
  volatile programming only unless the lead says otherwise; every boot with a control first.
- **Toolchain**: `~/cheri/output/sdk` as is; newlib purecap built once. Nothing in
  `llvm/` changes.
- **Repo**: this plan and the phase-2/4 results in `docs/`; the external tree, build outputs and
  routed reports stay under `~/capstone-artifacts/ext/` and `~/capstone-artifacts/cheri-cva6/`.
  The branch `cheri-cva6-eval` needs a push-allowlist line from the lead before it can leave the
  machine.

## 5. Risks, and what each would cost

| risk | likelihood | cost if it bites | mitigation |
|---|---|---|---|
| Tag controller wired only for VCU118 | medium | one RTL edit + lint + audit before synthesis | phase 0 reads the top before anything else |
| Genesys2 DDR3 MIG path needs the user-bit/dwidth treatment the DDR4 path got | medium | a day of SoC plumbing | copy the VCU118 pattern; phase 1's tag round-trip is the test |
| Vivado version mismatch between the fork's IP scripts and the synth machine | medium | IP regeneration, a wasted run | build A first — it is the flow check |
| Fork–SDK ISA revision drift (CTSRD moved encodings during 2025) | medium | purecap programs mis-assemble or trap; hybrid/plain still runs | phase 1 item 2 is the detector; fall back to the fork's own toolchain pin if it names one |
| CHERI-CVA6 also fails timing at 25 MHz | low–medium | the comparison changes meaning, not value | the census (not the slack) decides usability, as for ours |
| Board time collides with the SLT campaign and other lanes | certain | scheduling only | board lane serialises; volatile programming makes every CHERI session reversible by a power cycle |
| The paper's framing | n/a | none — this plan reports into the measurements doc; the paper is the lead's | ask before touching `paper/` |

## 6. What was considered and rejected

- **Synthesising first, simulating later** — a bitstream costs 2–3 h of a shared machine; a
  Verilator run costs minutes and catches the toolchain/ISA mismatch that would otherwise be
  read as a silicon result.
- **Non-volatile flash of the CHERI bitstream** — it would displace the Capstone bitstream every
  other lane depends on; volatile programming gives the same measurements.
- **CheriBSD on the FPGA** — out of scope (above).
- **Using the fork's `cheri-cam` default branch** — no FPGA tag support; `vcu118` is the one
  with it and it keeps Genesys2.
- **Comparing against CHERI numbers from the literature instead of our own runs** — different
  devices, periods and flows; the whole point is one board, one flow.

## 7. First concrete step, if approved

Phase 0 in full (a day), ending in: a full checkout at `vcu118` with submodules, the answer to
the Genesys2 tag-controller question with `file:line`, a Verilator model that builds, and the
synth README for build B — then phase 1's two smoke programs.

## 8. Status log

### 2026-09-07 — phases 0 and 1 complete; phase 2 waits for a synth-machine slot

**Phase 0 — done, with three fork-tip defects found and worked around.** Full clone at `vcu118`
`a97cf097` under `~/capstone-artifacts/ext/cheri-cva6-full`; the Verilator 5.008 model
(`work-ver/Variane_testharness`, target `cv64a6_imafdchzcheri_sv39`) builds with `make verilate`.
The tip does not build as cloned:

| defect | where | effect | fix applied here | affects synthesis? |
|---|---|---|---|---|
| cvfpu gitlink stale: the tip's Flist lists the T-Head divsqrt sources (`core/Flist.cva6`, 25 lines added by `a97cf097`) but `core/cvfpu` still points at v0.7.0, which has none of them | `core/Flist.cva6:43-67` vs the recorded submodule commit `3116391b` | Verilator (and Vivado, which reads the same Flist: `Makefile:616,741`) stops with 25 missing files | cvfpu checked out at `2c794772` (2025-02-12) — the pin upstream OpenHW cva6 carried when it introduced exactly these Flist lines (`b3ece7c5`, 2025-10-13); the next upstream bump (`bb5e4f39`, 2025-11-24) adds an `early_valid_o` port the fork's `core/fpu_wrap.sv:535` never connects, so anything newer fails `PINMISSING` | **yes — the synth lane must apply it** (recipe in the tree's `SYNTH-README-genesys2.md`) |
| testbench: the "add second uart" commit (`b7957bef`) declares the `uart2_*` APB block twice and reuses the instance name `i_axi2apb_64_32_uart` for the second bridge; the testharness never wires the new `uart2` slave, its pins or its address rule | `corev_apu/tb/ariane_peripherals.sv:311-336`, `corev_apu/tb/ariane_testharness.sv:604-711` | 9 elaboration errors, then a missing interface pin | 6+/9− testbench-only diff mirroring the fork's own FPGA top (`InclUART2=0`, `master[ariane_soc::UART2]`); `corev_apu/tb/` is not synthesised | no |
| Verilator 5.008 emits `!=` on nested `VlUnpacked` arrays for the tag controller's bloom lock box, and 5.008's header has no such operator | `work-ver/…DepSet…cpp` on `i_lock_box_bloom`; `verilated_types.h` `struct VlUnpacked` | model C++ does not compile | element-wise `operator==`/`!=` added to the tree's private copy of the header (hardlink to our `capstone-ariane/tools` copy broken first); the fork's own `verilator-v5.patch` is byte-identical to ours and does not cover it | no |

Reading of the three together: the `vcu118` tip does not build from a clean checkout. Two further harness facts, both needed to get any verdict at all: the
fork's `cva6.py` sanity test needs a libc the container's bare-metal gcc lacks (`syscalls.c:4`
`string.h`), so the model is built with `make verilate` directly; and the harness only ends a run
on a write to `tohost` if `+tohost_addr=<hex>` is passed (`corev_apu/tb/rvfi_tracer.sv:47`) —
without it every run "SUCCEEDS" at the timeout with `tohost = 2147483647`, which is not a result.

**Phase 1 — done; the CHERI core executes and enforces capabilities in simulation, hybrid and
purecap, through the DRAM tag path.** Every probe writes each reading with a plain `sd` into a
results array (visible as `mem <addr> <data>` in the RVFI trace) and exits with a bitmask, so a
trapping step still leaves data. Predictions were written in the source headers before each run.
Sources: `~/capstone-artifacts/ext/cheri-tests/cheri_probe*.S` (our CHERI-LLVM 17, `-mabi=lp64d`
plus `.option capmode` for the purecap block); runner `ext/cheri-run-one.sh`, decoder
`ext/cheri-results.py` (exits non-zero when it finds no result store).

| item | what ran | readings | verdict |
|---|---|---|---|
| ISA floor | upstream `riscv-tests` env/p, compiled with the container's gcc, hybrid mode (plain RV64) | 107/107 pass — the full rv64{ui,um,ua,uc,uf,ud} set, the same 107 names as the fork's own `testlist_riscv-tests-cv64a6_imafdc_sv39-p.yaml` (51/13/19/1/11/12); 1 400–47 000 cycles each. One instrument fault on the way: the sweep passed a fixed `+tohost_addr=80001000`, and `rv64uc-rvc` (the only test whose `.text.init` exceeds 4 KiB, so env/p's `link.ld` puts its `tohost` at `0x80003000`) was reported TIMEOUT; its trace ended in the `<pass>` spin after `sw gp, tohost` with `gp = 1`, and the rerun with the address read from its symbol table passes in 2 035 cycles. The runners now read `tohost` per ELF with `nm` | pass |
| hybrid probe (`cheri_probe`) | DDC root cap; `csetbounds` 16 B; `sc.cap`/`lc.cap` round trip (L1-resident); integer `sd` over the slot; OOB `lw.cap` via `cincoffset +16`; in-bounds `lw.cap +12` | 25/25 as predicted: DDC tag 1, len 2⁶⁴−1, base 0, perms `0x78fff`; bounded cap tag 1, len 16, base = addr = buf, metadata `0xffff00000405a004` identical before and after the round trip and identical to the stored word; tag 0 after the integer overwrite; OOB load trapped with `mcause = 28`, `mtval = 0x2a1` = cause 1 (length violation) at cap index 21 (`cs5`), `cap_tval_t` layout in `cva6_cheri_pkg.sv`; in-bounds load no trap, value `0x0f0e0d0c`; 1617 cycles | pass; **positive control fired** |
| DRAM tag path (`cheri_probe_dram`) | write-through, no-write-allocate dcache (`DCacheType = WT`; `wt_dcache_wbuffer.sv:44`): the first load of a line that was only ever stored misses L1 and crosses AXI into `axi_tagctrl` (`fence` does **not** flush the WT dcache — `controller.sv:121-125` flushes only for WB — so the probe's own header overstates it; its A/C/refill readings are DRAM-path by L1 miss, its B reading is L1-consistent); then 3072 tagged stores one per 4 KiB (12 MiB, 3× the 8-way × 128-line × 32 B tag cache) to force eviction to the in-DRAM table, a 96 KiB scan of the table region (which also cycles the 32 KiB L1), and a reload of a slot whose table word had been written | 22/22: tag 1 / len 16 / metadata unchanged after a store and a first load (L1 miss, AXI); tag 0 after an integer overwrite (L1-resident line, consistent); a never-written slot reads untagged (AXI); 2058 table words written during the sweep, first at `0xA00200A0` = `0x4` (bit 2 = capability index 2 of the 1 KiB block at `0x81005000`, exactly `axi_tagctrl_ax.sv:101-107`); the reload of `0x81005020` — whose tag line was evicted — came back tag 1, len 16, i.e. **refilled from the in-DRAM table** (that line had never been loaded, and the 96 KiB scan had cycled L1, so neither L1 nor the tag cache could have served it); no traps; 280 552 cycles | pass |
| purecap mode (`cheri_probe_purecap`) | hybrid stub builds the caps, `CJALR` (raw `.insn`; our assembler accepts the mnemonic only under `.option capmode`) into a capmode block: `csc`/`clc`, OOB `clw` through a +16 cap, in-bounds `clw`, `CJALR` back through a flag-0 cap | 10/10: `cgetflags(pcc)` = 1 inside the block, tag 1, len 16, the OOB `clw` trapped (`mcause 28`, `mtval 0x2a1`), `mret` returned in capability mode (flag read 1 right after), in-bounds `clw` = `0x0f0e0d0c` with no second trap, flag 0 again after the return; 1533 cycles | pass; **positive control fired** |

Not run, stated so it is not read as done: `cinvoke` on a sealed pair (plan item 2's last clause)
and the fork's TestRIG-vs-Sail harness (item 3). Neither gates phase 2; the ISA-drift risk they
were meant to detect is already answered — every encoding our SDK emitted (`cspecialr`,
`csetbounds`, `cgethigh`, `sc.cap`/`lc.cap`, capmode `csc`/`clc`, `CJALR`) decoded as intended
(`core/decoder.sv:1557-1567,1748-1750`). Sealing goes into phase 3's demonstrators.

Two runs before these were **void**, and the lesson is worth the line: CHERI-RISC-V has a merged
register file, so `t1` and `ct1` are one register; a probe that uses `tN` as an integer scratch
while `ctN` holds the capability under test destroys the capability and reports a core defect that
is not there ("`cgetlen` = 0 after a round trip"). Probes now keep capabilities and integer scratch
in disjoint x-registers and say so in their header. A second instrument fault: the harness opens
`trace_rvfi_hart_00.dasm` in its working directory with truncation, so two simulations in one tree
clobber each other's trace (verdict lines on stdout are unaffected); each run now gets its own
directory.

**A finding for phases 2–3, decide before synthesis (bitstream constant).** On the FPGA the tag
table sits at `TagCacheMemBase = 0xBFF00000` (`ariane_xilinx.sv:217`) with tags tracked for
`0x8000_0000..0xA000_0000` (`:1106`); `TagCacheMemLength = 0x10000` (`:218`) is declared but not passed to the controller. The controller stores 8 bytes of table per 1 KiB of data
(`axi_tagctrl_ax.sv:101-107`, layout confirmed above), so the table for the 512 MiB window is 4 MiB
and ends at `0xC030_0000`, 3 MiB past the 1 GiB Genesys2 DRAM window. Tags for data below
`0x8800_0000` (the first 128 MiB) are inside DRAM; capabilities stored above that address have their
tag traffic sent to addresses beyond the DRAM window; what the MIG path does with those (an error, or a
wrap into the first megabytes of DRAM, where programs live) is unread — a wrap would corrupt memory
silently. §2's row already noted "tags for the first 128 MiB"; the
consequence is that any purecap program above 128 MiB silently loses tags. Options, the lead's call:
(a) synthesise as shipped and keep phases 3–4 below `0x8800_0000` (bare-metal does); (b) the
one-constant change `TagCacheMemBase = 0xBFC00000` (DRAM top − 4 MiB) so the whole window is covered
— a fork defect fix, not a flow change, but it must be in the tree before the run. Written into the
tree's `SYNTH-README-genesys2.md` for the synth lane either way.

**Phase 2 — ready, blocked on a synth-machine slot.** Recipe, the cvfpu pin, and the table-placement
decision are in `SYNTH-README-genesys2.md` in the tree. Nothing under `core/` was edited; the fork
is synthesised as it ships plus the pin. Our `rtl-lint-gate.sh` and its baseline are for
`capstone-ariane` and do not apply to this tree; the phase-2 gate for build B is therefore "the fork's
own flow runs to a routed design", with build A as the flow check, as §3 already says.

**Phase 5 rows that need no synthesis** (the rest of the table waits for phases 2 and 4):

| axis | Capstone-CVA6 | CHERI-CVA6 (`vcu118`, as read and as simulated) |
|---|---|---|
| capability format | see `docs/design/capability-bounds-model.md`; the revocation-node pool (65 536 nodes) is an RTL constant | 128-bit + 1 tag bit (`CLEN = 2·XLEN`), CHERI-Concentrate bounds (`CAP_E_WIDTH = 6`), 12 hardware + 4 user permissions, 18-bit otype, a flag bit selects capability mode; root and NULL share the all-zero bounds encoding (`cgethigh(ddc) = 0xffff000000000000`) |
| what a violation looks like | Capstone exception classes per our spec; `tval` conventions in `docs/ref` | `mcause 28`, `mtval = cap_idx[10:5] ‖ cause[4:0]` (`cap_tval_t`); length violation = cause 1, tag violation = cause 2 (both observed) |
| tag / metadata storage | metadata travels inside the capability word; no separate tag store | one bit per 16 B kept in DRAM by `axi_tagctrl`: 8-way × 128 lines × 32 B write-back tag cache (4 MiB of data covered per fill), table of 8 B per 1 KiB; on Genesys2 the table as shipped covers only the first 128 MiB (above) |
| temporal safety | revoke-at-free as a hardware operation on the node tree | none in hardware; monotonic capabilities + tags; sweeps are software |
| execution modes | one model: domains entered by `capenter`, monitor in M-mode | integer (hybrid) and capability (purecap) modes per PCC flag, switched by `CJALR`; traps run in the mode of `mtcc` and `mret` restores the interrupted mode (observed) |
| toolchain used here | our LLVM fork, Capstone-C, monitor, Linux | CTSRD CHERI-LLVM 17 (`~/cheri/output/sdk`), hybrid `lp64d` with `.option capmode` blocks; the hybrid compiler crashes on a tail call through a capability function pointer at `-O1` (`RISCVISD::TAIL` not selectable) — noted, not investigated |
| verification the tree ships | our sweep, directed tests, board rungs | TestRIG/RVFI-DII vs Sail harness present, not run; no directed CHERI tests under `verif/tests/custom`; the tip was pushed unbuilt (phase 0) |
