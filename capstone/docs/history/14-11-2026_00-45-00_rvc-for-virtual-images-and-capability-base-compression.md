# RVC for virtual-profile images, and compressing capability-base accesses (2026-11-14)

The lead asked, through the runtime lane, for compressed instructions in Capstone code. On silicon, L1I misses are
549 M on gcc train01 and 44.8 M on perl makerand, and Capstone text is about 1.75x riscv-base's. This records:

- stage 1: standard RVC (`+c`) in virtual-profile images;
- a register-allocation fix it needed;
- a compiler feature (`+cap-base-rvc`) that compresses loads and stores off a capability base;
- the per-exec kernel cost found on the way;
- the stage-2 census for a compressed-capability encoding set.

## Why `+c` was off

- **Profile fixity only, as with `+relax`.** `capstone-cc` accepts any feature the SDK's own `sdk.json` codegen names,
  and arm C's `capstone-cc` compiles with `+c` added there.
- **No encoding conflict.** Capstone ops are in custom-2/custom-3 (bits [1:0] = 11), and no 16-bit encoding is reused.
- **The hardware (rtl lane, board revision 5f41c486e, read with `git show`).**
  - The compressed decoder is gated only by `CVA6Cfg.RVC` (id_stage.sv:159) and carries no capability logic.
  - JALR links pc+2 after a 16-bit instruction (branch_unit.sv:74).
  - pc_cap_check reads only the start pc and makes no alignment check.
  - Every hardcoded +4 follows an instruction with no 16-bit form.
  - A simulation arm (capstone-ariane perf/rvc-capmode) ran RVC on and off with identical readings. It covered 2-mod-4
    targets, traps and CSRUNV resumes.
- **The native route.**
  - binfmt-dom and capstone_vm apply no relocation or patch to `.text` and never load an instruction word.
  - The runtime and musl have no code-reading path.
  - QEMU's virtual-context PCC check covers [pc, pc + len).
  - The rtl lane found that QEMU's plain-M path has no PCC fetch check. It is filed against R-72 and does not concern
    these images.

## Arms

Every arm uses arm C's SDK (3e70344d8c3f): libc and the runtime are unchanged and uncompressed. Each differs from its
reference only as named:

| Arm | = | Compiler |
|---|---|---|
| CC | C + `+c` | llvm-c4 |
| CCD | CC + `-mllvm -capstone-disable-cost-per-use` | llvm-c4 |
| CCDK | CCD + `+cap-base-rvc` | llvm-c7 |
| FRC, FRCD, FRCDK | the same steps on FR (fold + relax) | llvm-c6, llvm-c6, llvm-c7 |
| FRCDx | FRCD built by llvm-c7 with the feature off (control) | llvm-c7 |

- **The spill fix.** With `+c`, the allocator's CostPerUse penalises registers outside x8-x15. That cost perl +626
  LDC and +271 STC sites, and +0.40 % user instructions (gcc +0.16 %). `-capstone-disable-cost-per-use` (an existing
  option) removes the whole cost, and CCD's text is slightly smaller than CC's. Every later arm carries it.
- **`+cap-base-rvc`** is compiler-capbase-rvc 74080cce29e5. Every access on this target takes a capability base
  (`Load_ri` uses `GPCRMem`), so the upstream CompressPats, which name GPR bases, never fire. With `+c` alone,
  `ld a4, 0(a1)` stays 32-bit.
  - The feature adds codegen-only 16-bit forms with the base in c8-c15 or c2, using standard encodings, and
    compress-only patterns.
  - It is not an ISA change: the hardware expands C.LD to the same LD encoding, and that encoding already accesses
    memory through the base capability.

## Results (native twin, ids55, natio k3, capmix3)

**Outputs.** Byte-identical to the reference in every cell: perl 13 files, gcc 5 files. Validation is unchanged.

**User instructions.** Shift 24, every command window.

| Arm | perl test-sw | gcc train01 |
|---|---|---|
| CC vs C | +0.40 % | +0.16 % |
| CCD vs C | -0.00 % | -0.00 % |
| CCDK vs C | -0.00 % | -0.00 % |
| FRCDK vs FR | -0.02 % | -0.02 % |
| FRCDK vs C | -5.63 % | -1.22 % |
| FRCDx vs FRCD | +994 | +26 k |

- FRCDx and FRCD build byte-identical perl.
- In gcc, 386 of 387 objects are identical without `.comment`; toplev.o embeds the compiler version.
- As a control, FRCD against FRCDK differs in 353 objects.

**Executable bytes (static).** riscv-base for reference: gcc 5,554,356, perl 1,318,856.

| | gcc | perl |
|---|---|---|
| C | 9,646,644 | 2,316,272 |
| CCD | 8,706,844 (-9.7 %) | 2,079,216 (-10.2 %) |
| CCDK | 8,496,664 (-11.9 %) | 2,020,376 (-12.8 %) |
| FR | 9,340,768 | 2,214,848 |
| FRCD | 8,385,004 (-10.2 % vs FR) | 1,972,496 (-10.9 % vs FR) |
| FRCDK | 8,184,808 (-12.4 % vs FR, -15.2 % vs C) | 1,915,004 (-13.5 % vs FR, -17.3 % vs C) |

**Instruction bytes executed** (`perf/cclever/fetchbytes.py`). Shift-1 per-PC, window 0 (perl = makerand), image
only; instruction lengths are read from the ELF. The C and FR controls read exactly 4.000 B/instruction.

| | gcc | perl makerand |
|---|---|---|
| C | 50.35 GB | 13.30 GB |
| CCD | 45.18 GB (-10.3 %) | 12.07 GB (-9.2 %) |
| CCDK | 43.45 GB (-13.7 %) | 11.43 GB (-14.0 %) |
| FR | 49.75 GB | 12.38 GB |
| FRCD | 44.57 GB (-10.4 % vs FR) | 11.15 GB (-10.0 % vs FR) |
| FRCDK | 42.86 GB (-13.9 % vs FR, -14.9 % vs C) | 10.62 GB (-14.2 % vs FR, -20.1 % vs C) |

**QEMU gate** (c1-gate/gt2 gate-rvc and gate-rvck: the runtime/virtual contracts built with `+c`, and with
`+c +cap-base-rvc`):
- hosted 93/93 and fp-s2 pass in every arm;
- malloc, nodes, w55-malloc and w55-nodes fail with failing-check lists identical to gate-fold's c6/c6fr controls
  (the known ABI-7 environment);
- the gate-rvck contracts contain 265 capability-base 16-bit accesses.

lit, CodeGen/Capstone and MC/Capstone: 169 tests, 168 pass. The one failure is the known shared-patches-present.test.

## A per-exec kernel cost that follows the text segment's size, not RVC

**What was seen.** Perl's test.pl window (about 4,800 native execs) shows the cost:
- CC: kernel +2.7 % and pages +9.3 % against C, about 3.8 pages and 18 k kernel instructions per exec;
- FRCDK: +3.8 % and +11.1 % against FR;
- gcc, with few execs: +0.2-0.4 %.

The cost grows with the text pages removed: F about 0.4 pages per exec, CC 3.8, FRC 5.5. In a per-symbol kernel
profile (CC against C, window 1), `memset`, GUP, page allocation and free all rise, while file-fault work
(`filemap_map_pages`, `do_set_pte`) falls.

**Two layout arms** relink CC's objects (measurement-only scripts in perf/cclever/ld):
- **CCP:** CC with the text segment padded back to C's 0x2c0000 with zero bytes, which puts the data where C has it.
  Pages -0.15 % and kernel -0.78 % against C. The cost is gone.
- **CCB:** CC with .bss padded by the same 0x3a000, which leaves the data where CC has it and equalises the slack to
  the 4 MiB region end. The cost is unchanged from CC: pages +9.2 %, kernel +2.7 %.

So the cost depends on the text segment's size, or equivalently the data segment's position (the arms cannot tell
those two apart). It does not depend on the slack: a slack mechanism was considered in-session and CCB refutes it.

**Not settled.** The mechanism inside the native module (lazy admission, pin-ahead or resident scan) has not been read
to a cause.

**Consequence.** Smaller text costs per-exec kernel work on this module. The RVC images are neutral or better on
gcc and on makerand-only perl, but should not go to an exec-heavy row until the module side is understood or the text
segment is padded.

## Stage 2: what stays 4 bytes

`perf/cclever/c2census.py` classifies the 4-byte instructions left in FRC by RVC's own shapes applied to capability
ops. Results are dynamic, from shift-1 window 0, as a share of image instructions.

| Shape | gcc | perl makerand |
|---|---|---|
| ldc / stc sp-relative (C.LQSP-like, uimm 0..1008 / 16) | 7.82 % / 7.73 % | 6.24 % / 6.30 % |
| movc rd, rs (C.MV-like) | 4.56 % | 1.30 % |
| ldc rc, off(rc) (C.LQ-like, uimm 0..496 / 16) | 3.16 % | 5.32 % |
| cincoffsetimm sp, sp (C.ADDI16SP-like) | 2.32 % | 3.55 % |
| scc rd, gp, rs | 1.79 % | 7.64 % |
| cincoffset rd, rd, rs (C.ADD-like) | 0.89 % | 0.73 % |
| cincoffsetimm rd, rd, imm6 (C.ADDI-like) | 0.72 % | 0.18 % |
| standard shapes with a capability base (now `+cap-base-rvc`) | 8.75 % | 7.6 % |

**Zcb.** The board's CVA6 configuration also enables Zcb (c.lbu/c.lhu/c.sb/c.sh/c.zext.*/c.mul). The twin's QEMU CPU
leaves `zcb` off, so a Zcb arm needs `zcb=true` on the twin before it can be measured.

## Left

- **The per-exec cost above:** the module-side mechanism, or a text-segment padding rule.
- **Zcb:** the twin CPU flag, then an arm.
- **A compressed-capability encoding set:** the lead's and the rtl lane's choice, from the stage-2 table.
