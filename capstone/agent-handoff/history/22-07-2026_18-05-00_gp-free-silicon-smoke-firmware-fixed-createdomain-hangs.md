# KB: gp-free domain on silicon — full investigation, root cause, reproduction

**Read this first if resuming the gp-free-on-silicon (Experiment A) work.** Branch
`capstone-gp-free`. Self-contained knowledge base: what was tried, why, what broke,
the definitive root cause, and how to reproduce every step. Companion runbook:
`ref/gp-free-silicon-smoke-runbook.md`. QEMU proof:
`22-07-2026_16-09-12_gp-free-domain-bringup-qemu-proof.md`.

## UPDATE 2026-07-22 (latest-2): ra-spill fix done (necessary) but plain call/ret STILL crashes on RTL — needs board-owner input

Added the ra-spill fix (`llc` change, committed): with `-capstone-gp-free`, callee-saved
`ra` (X1) now spills with integer `sd`/`ld` instead of `stc`/`ldc`
(`CapstoneInstrInfo.cpp storeRegToStackSlot`/`loadRegFromStackSlot`), since gp-free
`ra` is a plain integer return address. Verified in asm (`sd ra`/`ld ra`) and QEMU
still passes (retval 554745961). **But the silicon crash persists, essentially
unchanged:** gdb-halt on the ra-fixed domain gives
`pc=0x2 mcause=0x2(illegal) mtval=0 ra=0x819a0064 gp=0x819a1000 mstatus.MPP=M`
(vs `pc=0x0` before). Same signature: gp correct (domain entered), `ra=base+0x64`
(glue's post-`call domain_main` return), control flow lands at ~0 and traps to M-mode.

**Conclusion:** the blocker is NOT the ra spill (fixed) but the **plain `jal/jalr`
call/return convention itself inside a SEALED DOMAIN on this RTL.** A return
(`jalr x0, 0(ra)`) with an integer `ra` lands at ~address 0 (illegal-instr trap) on
hardware while running fine on QEMU. This is fundamental to the gp-free ABI premise
("plain jal/jalr within PCC"): it holds for the M-mode monitor but appears NOT to
hold for intra-domain control flow on silicon (the domain may require `cjalr` /a
capability return address). **This needs the board owner** — see the question below.
gp delivery + firmware are DONE; this call/ret ABI question is the last blocker.

### Board-owner question (draft)
"Inside a sealed pure-capability domain on the CVA6/Capstone RTL, do plain `jal/jalr`
calls and returns (integer `ra`) work for intra-domain control flow, or must the
domain use `cjalr` / a capability return address? Our gp-free domain enters correctly
and the monitor-delivered `gp` is validated (gp=base+0x1000 via SPLIT), but returns
crash to ~address 0 (mcause=2 illegal-instr, mtval=0) on the RTL while working on
QEMU. If cjalr is required in-domain, we'll keep cjalr for calls/returns and use the
gp-free change only for global addressing (`scc gp`), not for call/ret."

## UPDATE 2026-07-22 (earlier): gp DELIVERY WORKS on silicon; remaining bug is the plain-jal/jalr call/ret ABI

gdb-halt on the hung core (driver `/tmp/capstone/board_run_gpfree_gdb.py`) gave:
`pc=0x0 mcause=0x2(illegal-instr) mtval=0x0 ra=0x819a0064 sp=0x819bff50
gp=0x819a1000` (domain base ~0x819a0000). Readout:
- **`gp = base+0x1000` = the split-derived gp.** The domain ENTERED (share worked),
  gp was delivered via cscratch, and it EXECUTED. **The C_GEN_CAP→SPLIT fix is
  validated end-to-end on hardware.** (The controller's `region=`/MARK prints were
  just lost/buffered on the flaky UART; the domain really did run.)
- `ra = base+0x64` = the return point right after the glue's `call domain_main`
  (`0x10060 jalr <domain_main>; 0x10064 ...`). So the crash is INSIDE the call graph.
- **`pc=0, mcause=2, mtval=0` = the domain jumped to address 0 and fetched a zero
  (illegal) instruction.** A return/`jalr` landed at 0. This is the OTHER half of the
  gp-free ABI: plain `jal/jalr` calls + `jalr x0,0(ra)` returns (the domain uses
  `auipc;jalr` calls and plain rets, NOT cjalr). Independent of gp; silicon-only
  (QEMU runs the identical call graph fine).
- **Next:** investigate the plain-call/ret path in a domain on silicon — how the
  integer `ra` is saved/restored on the capability stack (`sd/ld ra, off(sp)` where
  sp is the cscratch cap) and whether plain `jalr` sets/consumes `ra` as expected
  inside a pure-cap domain. Single-step domain_main's prologue/helper-call/epilogue,
  or compare against how the reference monitor does intra-domain call/ret. gp is NOT
  implicated.

## UPDATE 2026-07-22 (earlier): create_domain FIXED via SPLIT-derived gp; hang moved to domain-entry

The `C_GEN_CAP` fix landed. Replaced the fabricate-based gp mint with a real
derivation (`plans/gp-cap-derive-on-silicon-proposal.md`): the gp-free domain is
now linked with globals forced to a fixed offset
(`tests/runtime-qemu/gp-free-domain/link-gpfree.ld`, `GPFREE_GLOBALS_OFFSET=0x1000`),
and the monitor `create_domain` does `dom_gp = __split(dom_code, base+0x1000)` →
PCC=[base,base+0x1000) execute, gp=[base+0x1000,base+code_size) R/W, delivered via
the cscratch top-16 slot (same store target as before). No `C_GEN_CAP`.

- **QEMU: PASS** with `CAPSTONE_GP_FABRICATE=0` — `retval = 554745961`. Logic
  validated (split direction, cscratch store, cursor). Verified by rebuilding the
  **caplifive-buildroot** `fw_jump.elf` (the QEMU tree) with the same edit and
  running `run-domain-smoke.py`.
- **Silicon: create_domain now RETURNS** — the controller printed
  `gpfree-fpga: created domain ID = 0` (previously it hung here with zero output).
  The split-derived gp is accepted by the RTL. Firmware = caplifive-system
  fpga/ariane rebuilt with the split edit (`fw_payload_fpga_up_gpfree.bin`).
- **New blocker:** silent hang (no reboot banner) at the **domain-entry / region
  share** — after `create_region`/`map_region`/`memset` succeed (controller reached
  the `region =` print), the `REGION_SHARE_ANNOTATED` that enters the domain hangs.
  QEMU runs the same entry fine, so it's a silicon-specific difference in the domain
  executing with the split gp (candidates: glue `ldc gp; delin` on the split cap,
  `scc gp` bounds/perms, or the `domreturn` reset flagged in the runbook). Next:
  add a UART marker right before the share to bracket share-vs-entry, and check the
  `domreturn` path (Experiment A allocates no rev-nodes, so a clean exit was
  expected). create_region/share themselves work for the borrow-cost domain on
  silicon, so suspicion is the domain body/exit, not the region plumbing.

## TL;DR

- **Goal:** run a real globals-using integer domain (gpfree_app: data globals +
  non-inlined call graph) on the CVA6/Capstone RTL, confirming the monitor delivers
  `gp` via the cscratch slot. Expected retval `554745961` (`0x2110C069`).
- **Achieved:** the whole toolchain + firmware + boot chain now works on silicon —
  board boots Linux to a root shell with our gp-delivery monitor, `/dev/capstone`
  present, controller+domain transfer + sha-verified.
- **DEFINITIVE ROOT-CAUSE BLOCKER:** the monitor's `gp` mint uses **`C_GEN_CAP`
  (`.insn r 0x5b,0x1,0x40`), which is a QEMU-only DEBUG instruction
  (`helper_csdebuggencap`) and is NOT implemented on the RTL.** It fabricates a
  capability from `(base,end)` — impossible under hardware capability monotonicity.
  On the RTL funct7 `0x40` hits the decoder `default: ;` (does nothing) → the two
  `C_GEN_CAP` results are garbage → the following `stc` stores through a garbage cap
  → **M-mode faults into `capstone_error` = `while(1);` → the core hangs silently.**
- **Fix direction (NOT yet done):** derive the `gp` data cap from an existing
  authority the monitor holds, using RTL-**implemented** ops (CAPCREATE / CAPPERM /
  CAPBOUND on opcode custom-3, or split/scc/movc on custom-2), following the
  capstone-c cscratch cap-table reference the board owner pointed at. Likely needs
  board-owner alignment.
- **The compiler side is NOT implicated** (gp-free, cjalr-free, `scc gp` codegen is
  fine; the domain never ran — the hang is in M-mode create_domain).

## How the root cause was proven (all static — no board time)

1. Board symptom: after `# /tmp/gpfree_ctl /tmp/gpfree_app.dom` the UART shows only
   `BEGIN1\n` (7 chars) then goes dead; shell dead for retries; **no `gpfree-fpga:`
   line ever printed** (that first `puts_` is AFTER `IOCTL_DOM_CREATE` returns).
   Machine-level (not a user crash), silent (no reboot banner) → M-mode hang.
2. `IOCTL_DOM_CREATE` → kernel → M-mode `create_domain`, whose gp block is the only
   new code. caplifive-system `capstone_error(e)` is a bare `while(1);` → a failed
   check there spins M-mode with no output. Exact match.
3. RTL decoder `caplifive-system/hw/rtl/core/decoder.sv:1181` (OpcodeCustom2 = 0x5b,
   funct3=001) decodes funct7 **0x00–0x0d, 0x20, 0x21 only** (REVOKE, SHRINK,
   TIGHTEN, DELIN, LCC, SCC, SPLIT, SEAL, MREV, INIT, MOVC, DROP, CINCOFFSET, CALL,
   RETURN, CAPENTER). **funct7 0x40 → `default: ;` (unimplemented).**
4. QEMU `capstone-qemu/target/riscv/op_helper.c:1398 helper_csdebuggencap` +
   comment at :604 confirm C_GEN_CAP is QEMU-supplied ("debug gencap"). So it works
   on QEMU and no-ops on the RTL → garbage cap → `stc` fault → hang.
5. RTL DOES implement CAPCREATE/CAPPERM/CAPBOUND (decoder.sv custom-3) — the
   legitimate derive-from-existing-authority path a real fix must use.

The gp block (sbi_capstone.c ~L306-314) that must be reworked:
```c
unsigned d_end = base_addr + tot_size;
__linear void *dom_gp;
C_GEN_CAP(dom_gp, base_addr, base_addr + code_size);                 // (1) QEMU-only
__linear void *gp_slot;
C_GEN_CAP(gp_slot, base_addr + code_size + DOMAIN_DATA_SIZE, d_end); // (2) QEMU-only
C_SET_CURSOR(gp_slot, gp_slot, d_end - 16);                         // scc  (OK on RTL)
*(__linear void **)gp_slot = dom_gp;                               // stc through garbage -> hang
```

## Firmware/boot chain — fixed (this is now reliable; see the recipe memory)

Two silent-boot traps, both found by static diff vs known-good `fw_payload_fpga_up_ctl.bin`:
1. **Missing embedded FDT** (`d00dfeed`) → kernel gets garbage `a1`, hangs after the
   OpenSBI banner. Fix: `FW_FDT_PATH=<caplifive.dtb>`.
2. **Wrong OpenSBI platform** — first relink was *generic* (head diff ~96%); the
   board needs **`CAPLIFIVE-ARIANE`** from **caplifive-system** (the other tree,
   caplifive-buildroot, has "ARIANE RISC-V" + a DIFFERENT monitor version, 605-line
   `sbi_capstone.c` diff, WITHOUT our validated gp-delivery). Generic even reset the
   core into the board ROM (`Hello World! … initializing SD…`) at the kernel jump.

**Correct fast recipe** (minutes; see `project_fpga_fw_payload_build_recipe` memory):
- Kernel = bytes `0x200000..EOF` of the known-good fw (bootable, `FW_PAYLOAD_OFFSET
  0x200000`). DTB = the `d00dfeed` blob (totalsize at +4 BE; 3111 B). Both extracted
  into `/tmp/capstone/{kernel_payload.bin,caplifive_extracted.dtb}`.
- Regen `.c.S` from caplifive-system `sbi_capstone.c`:
  `capstone-c/target/debug/capstone-c --abi capstone <lib/sbi/sbi_capstone_dom.c>
  -- -I<lib/sbi/capstone-sbi> -D__riscv_xlen=64 > sbi_capstone_dom.c.S` (rm stale
  `.c.S` first — pattern rule watches the wrapper, not the `#include`d source; same
  for `capstone_int_handler`).
- `make -C caplifive-system/sw/buildroot/components/opensbi O=<out>
  PLATFORM=fpga/ariane
  CROSS_COMPILE=<caplifive-buildroot>/build/host/bin/riscv64-buildroot-linux-gnu-
  FW_PAYLOAD=y FW_PAYLOAD_PATH=<kernel> FW_FDT_PATH=<dtb> -j8`.
- Verify: 15367192 B, `CAPLIFIVE-ARIANE RISC-V` string, kernel@0x200000
  byte-identical to good, `d00dfeed` present, OpenSBI-head diff ~15%. Stage to
  `~/capstone-b-artifacts/fw_payload_fpga_up_gpfree.bin`.

## Reproduce the board run

Artifacts (all in `/tmp/capstone/`, staged):
- Domain: build with `capstone/tests/runtime-qemu/gp-free-domain/build-and-run.sh`
  path (gate: cjalr=0, cincoffset-gp=0, scc-gp>=1) → `/tmp/gpfree_app.dom`.
- Controller: `capstone/tests/rtl-smoke/gpfree_fpga_ctl.c`, built with the buildroot
  cross-gcc `-Os -static -nostdlib -ffreestanding -march=rv64imac -mabi=lp64`
  (soft-float; verify 0 FP insns). Single REV_SHARED region (PERM_INOUT=0x1,
  REV_SHARED=0x2, size 4096); the share IS the domain entry (region cap arrives as
  `domain_main(res,func)`'s `res`); reads `region[0]` retval. **Untested E2E**
  (create_domain hangs before the domain runs).
- Drivers: `/tmp/capstone/board_run_gpfree.py` (boot fw + transfer + run + harvest),
  `board_reflash_only.py` (step-A re-flash). Venv `fpga-venv/`, token
  `~/.config/capstone/fpga-board-url`. Re-flash uses the **server-side** named
  bitstream `working-caplifive-captype-fixed.bit` (no local `.bit`).
- Board etiquette (all sessions were clean): lock → power-cycle → run → **power off
  + unlock in finally**. Bitstream re-flash is the only persistent write (was
  authorized). ~2 min just to JTAG-load the 15 MB fw at ~130 KiB/s.

## Next steps

1. **Rework the gp mint** to derive from existing authority using RTL-implemented
   ops (study capstone-c's cscratch cap-table; candidates: CAPCREATE/CAPPERM/
   CAPBOUND, or split a data view + scc). The domain image [base,base+code_size)
   holds the globals; gp needs R/W bounds over the writable ones — consider whether
   writable globals must live in dom_data (which the monitor already has a cap for)
   rather than the executable image.
2. Rebuild fw via the recipe, one board run to confirm `create_domain` returns
   (controller prints `gpfree-fpga: created domain ID`), then the domain retval
   `554745961`.
3. Likely check with the board owner (they anticipated this: cursor-0 unrepresentable
   + capstone-c cap-table is the reference). See
   `project_silicon_gp_delivery_boardowner_guidance` memory.

## Constraints honored

Monitor edit stays a LOCAL experiment (no submodule-source commit). No real-person
names. Bug-fix/investigation → history/ dated (this file). See also memories:
[[project_fpga_fw_payload_build_recipe]], [[project_silicon_gp_delivery_boardowner_guidance]],
[[project_opensbi_monitor_rebuild_include_wrapper]].
