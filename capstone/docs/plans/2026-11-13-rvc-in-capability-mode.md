# Standard RVC in Capstone capability mode: what the silicon allows (2026-11-13)

*RTL lane, for the lead's compressed-instruction track (stage 1: the compiler emits standard RVC for non-capability
instructions in Capstone code). Read on capstone-ariane 5f41c486e, the board's base revision under bkit60/61. The
relevant files are byte-identical on 6d0e754d5: commit_stage, branch_unit, frontend, csr_regfile, controller,
compressed_decoder, decoder, id_stage and instr_realign. The source reading is an rtl-oracle report; its key lines
were re-read here.*

## Verdict for stage 1

The hardware allows standard RVC for non-capability instructions in capability mode. Nothing found gates or breaks
it. The caveats and open items are below.

## RTL facts (5f41c486e)

- **Decode.** The compressed decoder's only gate is `if (CVA6Cfg.RVC)` (`id_stage.sv:159`), with CVA6ConfigCExtEn = 1
  and ZcbExtEn = 1.
  - `compressed_decoder.sv` and `instr_realign.sv` contain no Capstone logic (grep 0, against a positive control on
    `decoder.sv`).
  - The capmode, supervision and virtual-context gates in `decoder.sv` act on the expanded 32-bit word.
  - No 16-bit encoding is reused: the Capstone ops are in custom-2/3, with bits [1:0] = 11.
  - F and D are on, so c.fld, c.fsd, c.fldsp and c.fsdsp are live.
- **Links and targets.**
  - `branch_unit.sv:74`: next_pc = pc + 2 for a compressed instruction. JALR links it, and so does CJALR
    (`ex_stage.sv:527`).
  - The instruction-address-misaligned raise is inside `if (!CVA6Cfg.RVC)` (`branch_unit.sv:149`).
- **PCC.** `pc_cap_check` (`commit_stage.sv:388-417`, M + capmode) tests only the START pc against
  [bound_start, bound_end), with no instruction length.
  - In M-mode, a 32-bit instruction that starts at bound_end - 2 commits with its upper half unchecked. That is
    reachable today with an unaligned bound, and RVC makes it reachable with 4-aligned bounds.
  - At S/U the CPMP checks each 4-byte fetch word (`pmp_data_if.sv:322-337`).
- **epc.**
  - A trap saves the whole pc (`csr_regfile.sv:2380`). A software csrw mepc clears only bit 0 (`:1760`).
  - CCSRRW CEPC is unmasked (`:2680`). mret returns to mepc unmasked.
  - The switcher restores the pc exactly for entry and resume (modes 1 and 3, `restore_io`). Mode 4, the service
    reply, adds 4 past an ECALL (`capstone_dom_switcher.anvil:321-325`).
- **Every hardcoded +4 follows an instruction with no 16-bit form:**

  | site | after |
  |---|---|
  | `frontend.sv:449` | CSR, fence, sfence, AMO |
  | `commit_stage.sv:883-893` | REVOKE, DROP, CREVOKE, CSRETIRE |
  | `commit_stage.sv:685` | CSRUNV |
  | `ex_stage.sv:2129` | CALL |
  | switcher mode 4 | the service ECALL |
- **Trap values.** tval on an illegal-instruction trap is the RAW 16-bit encoding, zero-extended
  (`decoder.sv:1967`). tinst is 0 (no H extension).
- **Instruments.** The commit-pc apertures read the full pc. RVFI reports the raw 16-bit word.
- **Monitor and OpenSBI epc +4 sites:**
  - `sbi_capstone.S:89` and `:210` (ECALL);
  - `:126` (the CSR_TIME emulation arm only);
  - `sbi_ecall.c:140`;
  - `sbi_illegal_insn.c:43` and `:100`.

  None is reachable from a compressed integer instruction. `sbi_misaligned_ldst.c` derives the instruction length
  correctly.

## Simulation

capstone-ariane perf/rvc-capmode **f454ba87f**, on 6d0e754d5's RTL, M-mode capmode. The same source is built with RVC
on (32 compressed instructions in the text) and off.
- Every program reading is identical between the two builds: arithmetic 0x3e4, the straddling lui/addiw 0x12345678,
  the loop 0x141e, jump and call 0x3, mepc - label 0, trap count 1.
- The layout checks read as predicted: on 2/e/2/2/2, off 0/c/0/0/0.
- Covered: a 32-bit instruction at 2 mod 4; one straddling a 16-byte fetch block; a compressed loop; a taken jump and
  a jalr call to 2-mod-4 targets, with ret; a trapping load at 2 mod 4, with mepc exact.
- One prediction was refuted on both builds alike: the integer-base load's cause reads 24, not 5. The 5 came from a
  vp2-era note.

**Inside a CSRUNV context** (va-runv-rvc.S, capstone-ariane perf/rvc-capmode 929799815 + its reading commit,
6d0e754d5's RTL; the same source with RVC on and off): every reading is identical except the layout checks and the
2-mod-4 resume count, which differ by design.
- A 3000-cycle quantum preempted a 2000-iteration mixed 16/32-bit loop 5 times, and S resumed each one. In the on
  build 2 of the 5 resumes were at a 2-mod-4 pc, and the loop's sums are identical (0x1e9fd8, 0x9f6ba470).
- The ECALL at 2 mod 4 saved its exact pc, and the reply (switcher mode 4, pc + 4) resumed at 2 mod 4.
- A capability fault at 2 mod 4 escaped with cause 28 and its exact pc.
- The first run's 2-mod-4 count was VOID: S read the frame through an unmapped address (cause 13, skipped by the
  handler). The count was moved into M and re-run.

**QEMU's PCC check covers the whole instruction** (qemu55 `translate.c:1530/1538`, `helper_capstone_check_pcc(pc,
len)`; `capstone_supervisor.c:1215-1219`). It is stricter than silicon's start-only check, so a twin pass cannot hide
the silicon leniency.

**The .dom loader does no relocation or patching of text** (compiler lane: `binfmt-dom.h`, `capstone_vm.c`), so the
data-side hazard does not apply to domain images.

**Straddle fault attribution (read, not run).** The stock frontend attaches a fetch exception to every instruction of
its fetch packet (`instr_queue.sv:225-226`). A straddling instruction issues with the packet that holds its second
half, so a fault on that half is taken with mepc = the instruction's start and mtval = the faulting fetch address,
which RISC-V permits.

## Silicon evidence

The board firmware is already compiled with RVC. The build-fpga `fw_jump.elf` (2026-10-17) disassembles to 18,951
16-bit and 17,027 32-bit instructions, and 57 of its 59 Capstone-named monitor functions contain 16-bit instructions,
`handle_trap_ecall` among them. So M-mode capability-mode RVC fetch has very likely run on every boot.
- The counter was controlled with the arm's two ELFs: on 32, off 0 (the off build's single 2-byte hit is padding).
- Not yet checked against the image that is currently flashed.

## Caveats and open items

- **Data-side hazard.** A misaligned DATA access inside a domain is not emulated (`fault_return_from_domain`). Code
  that reads or patches instructions with 32-bit loads breaks on compressed text. Relocation did exactly this: an RVC
  kernel module once hung insmod (`docs/history/07-10-2026_16-11-23_virtual-capstone-stack-audit-and-landing.md`).
- **Open:**
  - whether CPMP bounds can end at 2 mod 4;
  - the M-mode PCC-end leniency itself (a 32-bit instruction starting at bound_end - 2 commits), to become an R-entry
    once characterised by an arm;
  - the straddle attribution by an arm, not only by source;
  - silicon: the compiler lane's RVC images on board 1.

## Stage 2: free compressed encoding space on this core (Zcb on, Zcmp/Zcmt off, F/D on)

- Quadrant 0, funct3 = 100 (the Zcb group): instr[12:10] in {100, 101, 110, 111}, four slots, in CL/CS-like format with
  3-bit registers.
- Quadrant 1, funct3 = 100, the zext/sext/not group: instr[4:2] in {110, 111}, two one-register slots.
- Reserved forms: c.addi4spn with imm = 0; c.lui or c.addi16sp with imm = 0; c.lwsp or c.ldsp with rd = x0; c.jr with
  rs1 = x0.
- Quadrant 2 has no free sp-relative slot unless c.fldsp and c.fsdsp are given up in capability mode, the CHERI-style
  trade, which needs a mode-dependent decode.
- **Cost.** `compressed_decoder.sv` is a combinational 16-to-32 expander. A new op that expands into an existing
  custom-2/3 encoding costs one case arm and nothing downstream, provided its operands fit.
