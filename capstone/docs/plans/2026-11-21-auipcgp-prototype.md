# auipcgp: the PC-relative capability derive, prototype (2026-11-21)

Candidate (a) of `history/21-11-2026_09-30-00_pcrel-capability-derive-census.md`. It passed the 0.5 % gate on perl and
gcc; candidate (b) did not and was dropped. Definition, encoding and authority rule: `design/virtual-capstone/isa.md`,
"Prototype: auipcgp". **QEMU-only:** P1's decoder has no custom-1 arm, so no board image may carry it.

## Parts

- **Compiler, lane branch `compiler/auipcgp`.** Feature `+cap-auipcgp`, default off.
  - MC: the U-type in custom-1, decoded only under the feature; `%pcrel_hi` and `%pcrel_lo` as for auipc.
  - Codegen: CapstoneMergeBaseOffset forms it last, under the image-gp ABI only, from
    `auipc [+ addi %pcrel_lo] + PseudoCapGlobalBase` where each value has a single use and the symbol is not extern
    weak. The low part stays as a `cincoffsetimm`, or folded into the accesses.
  - lld: refuses it against an undefined symbol, since C-56's lui rewrite keys on auipc.
  - Tests: `MC/Capstone/auipcgp.s`, `CodeGen/Capstone/cap-auipcgp.ll`, `lld/test/ELF/capstone-auipcgp.s`. The Capstone
    lit scope (CodeGen, MC, disassembler, lld capstone*) passes 191 of 191.
- **QEMU, lane branch `compiler/auipcgp-qemu`, 72241b6ec6 on a0501ebc39.** Property `x-capstone-auipcgp`, default
  off; an illegal instruction without it.
  - scc's body is shared: `capstone_scc_cursor` serves `helper_csscc` and `helper_csauipcgp`, with rs1 = 3.
  - Binary sha256 ccec0b03a2504864.
- **Twin decode flag.** The property is static. auipcgp takes a free opcode, so unlike CRVC it needs no context mode
  bit, image magic or ELF flag.

## Arms and identity

AGP0 and AGP1 are CRVC0 (its SDK, flags and libraries) on the compiler llvm-agp, without and with `+cap-auipcgp`.
- The source is f8615464911c. The embedded version string reads 96c1612583eb: the build preceded the commit that
  carries only the getter rename it was built with, plus tests and docs.
- Libraries: libLLVMCapstoneCodeGen f91cc0ce806c64ba, libLLVMCapstoneDesc 9d164853b1992cd9, liblldELF
  5793b7a76fc6f0f3.

**Gate 1, feature off: section-identical to CRVC0, `.comment` excluded.**
- Ten of eleven images are identical.
- gcc differs, and only through the compiler's own version string: `toplev.o` embeds `__VERSION__` in
  `.rodata.str1.1`, and lld's merged-string layout moves every string address after it.
- Relinked by AGP0's lld with CRVC0's `toplev.o` in place of AGP0's, gcc is section-identical to CRVC0. All other
  386 objects are identical.

**Static reading, already taken (program code; `__capstone_cap_init` counted apart, generated per image).**

| row | auipcgp sites | registered band (computed from CRVC0 at 09:01, before the AGP1 build at 09:15) |
|---|---:|---:|
| perl | 15,221 (+2,041 in cap_init) | 13,071..14,826 |
| gcc | 59,758 | 52,304..60,342 |
| x264 | 1,696 | 1,302..1,621 |
| mcf | 51 | 37..48 |
| lbm | 30 | 24..29 |
| imagick | 12,640 | 5,861..12,058 |
| nab | 1,021 | 921..1,026 |
| xz | 485 | 444..503 |

- perl, x264, mcf, lbm and imagick come out above the band's upper edge, perl by 2.7 %. The band is a census
  heuristic over the off image's register allocation; the transform keys on SSA single use.
- Conservation holds: per image the auipc count falls by the auipcgp count plus calls newly relaxed to jal (perl
  18,038 = 17,262 + 776).

## Dynamic prediction, pre-registered before any AGP run

From CRVC0's shift-1 profiles, program functions. The band runs from the census rule (a) to (a) plus the sites the
census excluded only for distance, an intervening branch or conservative liveness (`s2gap.py`). Each execution
removes one 4-byte instruction.

| row | CRVC0 insns | predicted insns removed | predicted fetch bytes removed |
|---|---:|---:|---:|
| 500.perlbench_r-test-sw | 84,271,470,274 | 1.707 %..2.096 % | 1.949 %..2.393 % |
| 502.gcc_r-train01 | 12,221,841,750 | 0.955 %..1.284 % | 1.113 %..1.496 % |
| 525.x264_r-test | 508,109,995,900 | 0.036 %..0.054 % | 0.039 %..0.059 % |
| 505.mcf_r-test | 26,643,806,738 | 0.001 % | 0.001 % |
| 519.lbm_r-test | 6,092,487,936 | 0.000 % | 0.000 % |
| 538.imagick_r-test-ref | 95,383,602 | 0.005 %..0.008 % | 0.005 %..0.008 % |
| 544.nab_r-test | 7,154,822,856 | 0.022 %..0.039 % | 0.025 %..0.044 % |
| 557.xz_r-test | 49,630,300,219 | 0.049 %..0.050 % | 0.056 %..0.057 % |

- Second-order effects move the reading beyond the band in either direction: register allocation (fewer live
  integers), call relaxation, and gcc's string layout. Static code size already shows them: perl's instruction
  count falls 17,464, against 18,038 removed by the derive and relaxation.
- Validation: every row must validate as CRVC0 does (perl 1/2, x264 0/3, every other row all).

**Controls:**
- AGP0 xz on the auipcgp QEMU, property off, must read CRVC0's xz instruction count exactly: the images are
  identical, so this tests the QEMU change.
- AGP1 mcf without the property must stop on an illegal instruction: the feature-on image uses the opcode, and the
  check can fail.

## Readings (2026-11-21, after the pre-registration at d9fa11a409d2)

**Controls.**
- **AGP1 mcf without the property: fired.** The domain stopped with cause 2 (illegal instruction) at pc
  0x3ff7e0f242. Load base 0x3ff7e00000 plus link base 0x10000 maps that to 0x1f242, `auipcgp a0, 0x15`. The program
  exited 139.
- **AGP0 xz on the auipcgp QEMU, property off: FAILED its criterion as written ("exactly").** It reads
  49,630,299,499 instructions against CRVC0's 49,630,300,219, a difference of 720 (1.5×10⁻⁸), on identical images.
  - The criterion was wrong for this instrument. The same stage reads 49,630,299,500 on FRCDKMSa3 and
    49,630,308,775 on ZCB9 (`plans/2026-11-21-capability-rvc-stage2-prototype.md`), so twin instruction counts move
    by up to ~10⁴ from run to run.
  - The reading is consistent with the QEMU change having no effect off. It does not prove it to better than ~10⁴
    instructions.

**Rows.** Every row validates as CRVC0's does: perl 1/2, gcc 1/1, mcf 2/2, lbm 1/1, nab 1/1, imagick 1/1, xz 12/12.

| row | insns removed | registered band | fetch bytes removed | registered band |
|---|---:|---:|---:|---:|
| perl | **1.850 %** | 1.707..2.096 % | **2.129 %** | 1.949..2.393 % |
| gcc | **1.131 %** | 0.955..1.284 % | **1.326 %** | 1.113..1.496 % |
| nab | 0.028 % | 0.022..0.039 % | 0.032 % | 0.025..0.044 % |
| mcf | 0.001 % | 0.001 % | 0.000 % | 0.001 % |
| lbm | 0.000 % | 0.000 % | 0.000 % | 0.000 % |
| imagick | 0.009 % | 0.005..0.008 % | 0.017 % | 0.005..0.008 % |
| xz | 0.101 % | 0.049..0.050 % | 0.015 % | 0.056..0.057 % |

- perl and gcc, the two rows the census put through the gate, land inside their bands on both counts.
- **xz exposed a gap in the transform: jump tables.**
  - The derive formed on 14.9 M executions against the 24.5 M the census counted. rc_encode's
    `auipc; addi; scc gp` (9.26 M) and a second lzma_decode site (0.77 M) stayed, because both bases are jump
    tables (`AUIPC %jump-table.0`), and formAUIPCGP accepted globals, constant-pool entries, block addresses and
    symbols, but not jump tables.
  - A jump table lies in .rodata inside gp's bounds like a global, so this was an omission, not a safety condition.
    It is fixed on the lane branch, with a lit case; AGP2 below is built with the fix.
- **xz's remaining difference is register allocation in one function**, lzma_lzma_optimum_normal, which itself
  barely uses the derive (0.29 M):
  - 30.0 M fewer 2-byte `ld` and 14.5 M fewer `mv`;
  - but 27.2 M `and` moved from 2 to 4 bytes and 25.3 M more `movc`;
  - net −35.6 M instructions and +32.2 M bytes.
- Every converted site changed exactly as predicted: in lzma_decode, sha_compress, process and the rest,
  `auipc` + `scc` became `auipcgp` and `addi` became `cincoffsetimm`, one for one.
- Summed over every function: −50.3 M instructions and −26.6 M bytes, which closes the totals exactly.
- imagick is above its band by a few thousand instructions in 95 M; it was not localised.
- These are the readings of the FIRST build, AGP1 (libLLVMCapstoneCodeGen f91cc0ce806c64ba). x264 is still running on it.
- Tool: `s2fn.py` (per-function executions and bytes, two runs matched by name).
