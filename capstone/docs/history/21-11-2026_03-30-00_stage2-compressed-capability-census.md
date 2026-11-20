# Stage 2 of the compressed capability ISA: the per-benchmark census (2026-11-21)

The runtime lane asked for this census before a QEMU and compiler prototype. Its go/no-go: fetch bytes saved of at
least 3 % on two of perl, gcc and x264, the rows with the most code. For each candidate 16-bit encoding of the sketch's
option A (`plans/2026-11-14-capability-rvc-stage2-decoder-sketch.md`), per benchmark, it gives:
- static sites;
- dynamic executions;
- fetch bytes saved, as a share of the image's executed fetch bytes;
- the RVC slot the encoding takes, and what that displaces.

**Result: GO.**
- The full set saves 9.77 % of perl's executed fetch bytes, 15.35 % of gcc train01's and 2.22 % of x264's. Two of
  the three rows clear 3 %.
- x264 misses, and stays below 3 % on the set alone: its fetch is plain integer code (lbu, sub, xor) and
  `cincoffsetimm rd, rd, imm6`, for which option A has no slot.
- A compiler-only addition, Zcb's byte and halfword forms with a capability base, would add 1.26 % on x264 and take
  it to 3.48 %. It was not built.

## Instrument

- **Arm FRCDKMSa3**, the paper's Full arm: clang 29027126f39b (llvm-c9), the vp2 virtual SDK's ABI
  (`-capstone-gp-free -capstone-image-gp`), `+c` with `+cap-base-rvc` (stage 1), and
  `-capstone-fold-global-lo -capstone-disable-cost-per-use -capstone-memcpy-align-dispatch`. Its libc and runtime
  are uncompressed.
- **Profiles: native twin, shift-1 capmix3** (every user instruction counted), every command window.
  - QEMU qemu55 (binary e44716616ce352dd), plugin capmix3.so 28cd91878951c61c, `x-capstone-ids=55`, 936 MiB,
    `node_flat_records=524288`.
  - perl test-sw and gcc train01 were taken 2026-11-14, gcc test 2026-11-16, and the other six 2026-11-21.
  - specdiff: perl 2/2, gcc 1/1 (both inputs), mcf 2/2, lbm 1/1, imagick 1/1, nab 1/1, xz 12/12.
  - x264 reads 0/3, as every x264 twin run has: the twin plan does not run imagevalidate, so there is nothing to
    compare. Both of its commands returned 0.
- **Mapping.** Each window's PCs are grouped by image arena (binfmt-dom aligns the image to
  `roundup_pow_of_two(text + data)`; `pc - base = vaddr - text p_vaddr`).
  - An arena is accepted only when at least 99 % of its executions land on instruction starts of the stage's own
    binary. The rest is reported as not mapped, never folded in.
  - The mapping is per window. imagick and imagevalidate run in different windows and share an arena-sized block,
    and a summed profile mixed them (caught by the 99 % check, then fixed). perl and mcf read the same before and
    after that change.
  - Not mapped: perl 3.70 % (in its second window), x264 4.52 %, and imagick 38.4 %, which is imagevalidate (its own
    window). gcc test maps 99.58 %, the rest 100.00 %.
- **Decode.** llvm-objdump of the binary. Every executed image PC must be an instruction objdump lists, and one that
  is not is an error.
- **Candidates** (operand rules as the named RVC instruction, `rd'` = x8-x15):

| candidate | form | 16-bit slot it takes (RTL a948e4134 `compressed_decoder.sv`) | displaces |
|---|---|---|---|
| c.ldcsp | `ldc rd, uimm(sp)`, 0..1008, x16 | Q2 funct3 001 | C.FLDSP |
| c.stcsp | `stc rs2, uimm(sp)` | Q2 funct3 101 | C.FSDSP |
| c.ldc | `ldc rd', uimm(rs1')`, 0..496, x16 | Q0 funct3 001 | C.FLD |
| c.stc | `stc rs2', uimm(rs1')` | Q0 funct3 101 | C.FSD |
| c.cincoffset16sp | `cincoffsetimm sp, sp, nzimm`, -512..496, x16 | Q1 funct3 011 with rd = x0 | C.LUI x0, a HINT |
| c.movc | `movc rd', rs'` | Q0 funct3 100, [12:10] = 100 | nothing (reserved) |
| c.scc | `scc rd', gp, rs'` | Q0 funct3 100, [12:10] = 101 | nothing (reserved) |
| c.cincoffset | `cincoffset rd', rd', rs'` | Q0 funct3 100, [12:10] = 110 | nothing (reserved) |

Slot facts, read in the RTL at a948e4134:
- Zcb's Q0 100 group decodes only [12:10] = 000..011 (c.lbu, c.lh/c.lhu, c.sb, c.sh); 100..111 are
  `illegal_instr_o`.
- C.LUI with rd = x0 decodes as `lui x0` (a HINT); only nzimm = 0 is illegal.
- C.FLD, C.FSD, C.FLDSP and C.FSDSP decode when FP is present.
- No image here holds a 16-bit word in any of the taken slots. Q0 100 111 and the two Q1 unary slots stay spare.

**Controls, each reproducing an independent tool's reading.**
- gcc train01: c.ldcsp is 7.89 % of instructions, against the 14-11 stage-2 table's 7.82 % (window 0 only).
- perl window 0: c.scc 6.76 % and c.movc 1.18 %, against regclass.py's 6.75 % and 1.18 %.
- nab's displaced 16-bit FP ops sum to 10.06 % of instructions, against fpcount.py's 10.05 %; c.fld counts
  462,606,165 in both.
- imagick's c.fldsp, c.fld, c.fsd and c.fsdsp counts (318,045, 1,253, 83, 39) equal fpcount.py's exactly.

## Readings

Fetch bytes saved (2 bytes per execution), as a share of the image's executed fetch bytes. "set (B)" is the whole
set with a decode mode (option B), which keeps the FP forms. "net A" also charges option A for the FP forms it turns
back into 4 bytes. "text saved" is static: 2 bytes per site, over the image's text bytes.

| benchmark | image insns | fetch bytes | mapped | c.ldcsp | c.stcsp | c.ldc | c.stc | c.cincoffset16sp | c.movc | c.scc | c.cincoffset | set (B) | net A | text saved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| perl test-sw | 84,330,542,131 | 295,495,936,700 | 96.30 % | 2.68 % | 2.52 % | 1.18 % | 0.28 % | 0.77 % | 0.79 % | 1.19 % | 0.36 % | **9.77 %** | 9.75 % | 9.71 % |
| gcc train01 | 12,221,817,514 | 41,953,184,424 | 100.00 % | 4.60 % | 4.53 % | 2.01 % | 0.42 % | 1.36 % | 1.37 % | 0.71 % | 0.35 % | **15.35 %** | 15.35 % | 10.67 % |
| gcc test | 21,808,608 | 77,645,836 | 99.58 % | 3.19 % | 3.13 % | 0.84 % | 0.35 % | 1.00 % | 1.40 % | 0.77 % | 0.22 % | **10.90 %** | 10.90 % | 10.67 % |
| x264 test | 508,109,995,614 | 1,863,354,368,786 | 95.48 % | 0.65 % | 0.53 % | 0.08 % | 0.02 % | 0.12 % | 0.12 % | 0.03 % | 0.67 % | **2.22 %** | 2.22 % | 6.67 % |
| mcf test | 26,643,806,595 | 87,376,392,670 | 100.00 % | 0.41 % | 0.38 % | 5.54 % | 0.21 % | 0.06 % | 0.12 % | 0.00 % | 0.00 % | **6.72 %** | 6.72 % | 6.09 % |
| lbm test | 6,092,487,932 | 23,242,892,216 | 100.00 % | 0.00 % | 0.00 % | 0.03 % | 0.02 % | 0.00 % | 0.02 % | 0.00 % | 0.00 % | **0.07 %** | -2.93 % | 5.59 % |
| imagick test | 95,383,540 | 354,828,226 | 61.59 % | 0.15 % | 0.14 % | 0.21 % | 0.18 % | 0.05 % | 0.73 % | 0.01 % | 0.00 % | **1.48 %** | 1.30 % | 8.14 % |
| nab test | 7,154,823,390 | 25,574,755,936 | 100.00 % | 1.52 % | 0.20 % | 0.11 % | 0.00 % | 0.05 % | 0.00 % | 0.18 % | 0.34 % | **2.41 %** | -3.22 % | 6.87 % |
| xz test | 49,630,299,500 | 173,387,785,674 | 100.00 % | 0.81 % | 0.59 % | 0.13 % | 0.11 % | 0.14 % | 0.32 % | 0.00 % | 0.05 % | **2.16 %** | 2.16 % | 7.20 % |

Static sites / dynamic executions per candidate:

| benchmark | c.ldcsp | c.stcsp | c.ldc | c.stc | c.cincoffset16sp | c.movc | c.scc | c.cincoffset |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perl test-sw | 20,640 / 3,954,828,308 | 14,370 / 3,728,730,163 | 15,800 / 1,737,983,452 | 5,461 / 415,643,283 | 4,847 / 1,136,474,076 | 14,295 / 1,174,352,613 | 16,140 / 1,762,646,197 | 1,476 / 526,988,610 |
| gcc train01 | 89,657 / 964,277,687 | 60,223 / 949,959,458 | 67,785 / 422,263,566 | 27,952 / 87,456,307 | 22,837 / 286,195,641 | 94,526 / 287,045,928 | 66,981 / 148,845,582 | 7,472 / 73,707,896 |
| gcc test | 89,657 / 1,237,670 | 60,223 / 1,214,171 | 67,785 / 326,559 | 27,952 / 137,748 | 22,837 / 388,237 | 94,526 / 542,021 | 66,981 / 298,124 | 7,472 / 85,670 |
| x264 test | 6,521 / 6,047,289,924 | 3,970 / 4,954,807,189 | 1,924 / 705,116,149 | 1,156 / 224,781,856 | 968 / 1,126,394,631 | 2,141 / 1,115,496,724 | 2,032 / 251,529,390 | 972 / 6,287,724,935 |
| mcf test | 1,339 / 179,080,917 | 1,034 / 164,590,135 | 428 / 2,419,052,924 | 362 / 92,282,325 | 309 / 24,288,946 | 699 / 51,179,962 | 253 / 1,724,412 | 54 / 1,511,767 |
| lbm test | 1,195 / 15,129 | 912 / 15,024 | 368 / 3,050,604 | 263 / 2,628,493 | 283 / 5,244 | 594 / 2,635,160 | 232 / 1,654 | 55 / 210,411 |
| imagick test | 19,242 / 273,935 | 10,932 / 256,698 | 9,503 / 371,156 | 5,204 / 316,271 | 3,721 / 81,719 | 12,164 / 1,302,465 | 12,612 / 14,459 | 888 / 5,889 |
| nab test | 2,667 / 193,769,752 | 2,211 / 25,647,233 | 1,191 / 14,645,608 | 747 / 123,179 | 677 / 5,872,709 | 1,800 / 578,727 | 1,453 / 23,476,202 | 186 / 43,437,196 |
| xz test | 2,142 / 698,891,735 | 1,755 / 515,168,514 | 605 / 113,367,252 | 426 / 93,215,320 | 635 / 122,668,102 | 1,112 / 280,017,798 | 710 / 3,702,906 | 92 / 43,594,418 |

What option A displaces, static / dynamic (each execution becomes 4 bytes), and its fetch cost:

| benchmark | c.fldsp | c.fsdsp | c.fld | c.fsd | option A loss |
|---|---:|---:|---:|---:|---:|
| perl test-sw | 196 / 3,984,559 | 150 / 7,731,058 | 235 / 15,425,896 | 18 / 2,815,703 | -0.02 % |
| gcc train01 | 54 / 27 | 37 / 21 | 67 / 12,475 | 17 / 0 | 0.00 % |
| x264 test | 397 / 1,258 | 297 / 1,005 | 156 / 2,535 | 56 / 768 | 0.00 % |
| mcf test | 2 / 0 | 2 / 0 | 0 / 0 | 0 / 0 | 0.00 % |
| lbm test | 31 / 144,280,834 | 32 / 144,280,890 | 32 / 8,200,006 | 5 / 52,400,002 | -3.00 % |
| imagick test | 2,731 / 318,045 | 1,790 / 39 | 2,445 / 1,253 | 762 / 83 | -0.18 % |
| nab test | 246 / 119,248,133 | 198 / 46,507,161 | 360 / 462,606,165 | 195 / 90,759,707 | -5.63 % |
| xz test | 0 / 0 | 0 / 0 | 2 / 0 | 0 / 0 | 0.00 % |

Option A turns lbm and nab negative (-2.93 % and -3.22 % net). Option B, a decode mode, does not, at the cost of the
mode bit, its flushing CSR write and the ABI agreement the sketch describes.

## x264, and what would move it

x264's executed fetch is dominated by 4-byte integer instructions that no candidate covers:
- lbu 18.1 % of fetch bytes, sub 15.5 % and `cincoffsetimm` 13.4 %;
- then bne, xor and srai.

Near misses, as the fetch-byte share a 16-bit form would save:

| shape | x264 | perl | gcc train01 |
|---|---:|---:|---:|
| byte/halfword access, a register outside x8-x15 | 9.15 % | 0.74 % | 0.41 % |
| `sub`, registers outside the RVC form | 7.75 % | 0.37 % | 0.06 % |
| `cincoffsetimm rd, rd, imm6` (C.ADDI-like; option A has no slot) | 6.37 % | 0.43 % | 0.39 % |
| `xor`, registers outside the RVC form | 2.60 % | 0.05 % | 0.03 % |
| **Zcb byte/halfword shape with a capability base** (c.lbu/c.sb uimm 0..3, c.lh/c.lhu/c.sh 0 or 2) | **1.26 %** | 1.20 % | 0.60 % |
| `cincoffset` three-address | 0.49 % | 1.54 % | 1.30 % |

- **The Zcb row is compiler-only**, the same move as stage 1. The board enables Zcb (`CVA6ConfigZcbExtEn = 1`), and
  its c.lbu expands to the same lbu, which already accesses through the capability base. It would take x264 to
  3.48 % and perl to 10.97 %.
- **The register rows** are today's allocation; what a register-class preference recovers is not measured.
- **The `cincoffsetimm rd, rd, imm6` row needs 5 + 6 payload bits.** On this map the only such space is the four D
  slots, which option A already uses.
- **Control:** integer ops that already fit their RVC form read 0.03 % or less. They are the SDK's uncompressed
  libc and runtime.

## Limits

- **An upper bound on the encoding**: every site that fits is counted as compressed, and register allocation is
  unchanged. A compiler that steered operands into x8-x15 would raise the 3-bit rows. A compiler that could not use
  a form at some site would lower them.
- **Fetch bytes are not I$ misses.** The I$ 64-byte line build was withdrawn, so nothing else on the I-side is in
  flight, but the miss effect of these savings is not measured.
- Only the image's own instructions are counted. The not-mapped shares above are excluded from both the numerator
  and the denominator.
- The PC-relative capability load and the liveness-test instruction get their own census, in this shape, later.
