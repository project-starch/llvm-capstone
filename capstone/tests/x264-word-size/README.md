# x264 writes a malformed H.264 bitstream under the capability ABI: WORD_SIZE is sizeof(void*), 16 bytes (2026-11-28)

**Finding.** SPEC CPU2017's 525.x264_r, built for capstone64 (B5TL1's SDK and flags), writes an H.264 stream that the JM
reference decoder rejects.
- The same source built for plain riscv64 (Plain-matched) writes a stream it decodes.
- The cause is in x264's source, not in the compiler. common/osdep.h:231 defines `WORD_SIZE sizeof(void*)`, and the bit
  writer, common/bs.h, treats WORD_SIZE as the machine word.
- On capstone64, sizeof(void*) is 16 while the writer's accumulator, `intptr_t cur_bits`, is 8.
- **Why no check caught it:**
  - SPEC's own validation of x264 compares only the encoder's reconstructed frames (imagevalidate of --dumpyuv's
    frame_N.yuv), and those are correct.
  - The board graders' x264 reference for BuckBunny_New.264, 0e453c44… (b5b2-board-graders/grade-spec-chain-v2.py:35),
    is the Capstone stream itself.
- No prior record was found: a search of docs/, ports/ and tests/ on dev for WORD_SIZE, bs.h, BuckBunny_New and x264
  decoding found nothing.

## Evidence

All of it was read on the trap twin: kernel 749fed26, stock fw-y, capstone-qemu c42ff84d0799.

**1. The stream fails an independent decoder.**
- **The encode:** one frame of BuckBunny.yuv at `--qp 26` (no rate control), with each arm's x264. This is qpdec.c, run
  as /init. B5TL1: x264_r.strip.dom f686fe9679b04d69; Plain-matched: x264_r_base.plain-matched fe7092b29993f6e5.
- **The two streams:**

  | stream | bytes | first bytes (hex) |
  |---|---:|---|
  | B5TL1 | 835 | `00 00 00 01 67 00 00 40 00 00 0c a3 c6 0c 65 80 …` |
  | Plain-matched | 827 | `00 00 00 01 67 64 00 1f ac 34 e5 01 40 16 e8 40 …` |

- **They differ from byte 5**, inside the first NAL unit, the SPS:
  - **Plain's** SPS begins profile_idc 0x64 (High), level_idc 0x1f (3.1), as expected for 1280×720.
  - **B5TL1's** begins profile_idc 0x00, which is not a valid profile. Plain's bytes `40 00 00 0c a3 c6 0c 65 80`
    appear in it shifted earlier, so bits have been lost.
- **The decoder:** SPEC's own ldecod (the JM reference decoder, the stage's plain RISC-V build) decoded each stream.
  - **Plain-matched:** status 0, one 1280×720 frame, 1,382,400 bytes.
  - **B5TL1:** status 0xf400 (exit 244), no output.
- **At 10 frames the sizes still differ, with rate control on or off:**
  - default rate control, B5TL1 185,393 B against Plain 185,382 B;
  - `--qp 26`, 228,296 B against 228,286 B.
  - So the difference is not in rate control.

**2. The mechanism, in x264's own bit writer.** bstest.c includes x264's unmodified common/osdep.h and common/bs.h (with
-DSPEC, as SPEC builds) and writes a known sequence: 8+8+8+8 bits (0x67 0x64 0x00 0x1f), 13+3 bits, 16 bits and 32
bits, then a flush.

| build | WORD_SIZE | bytes written |
|---|---:|---|
| Plain-matched | 8 | `67 64 00 1f 91 a5 be ef de ad be ef` (correct) |
| B5TL1 (capstone64) | 16 | `de ad be ef 00 00 00 00 00 00 00 00`: only the last word survives |
| B5TL1 with osdep-word-size.patch | 8 | `67 64 00 1f 91 a5 be ef de ad be ef`, equal to Plain |

**Why WORD_SIZE = 16 breaks the writer:**
- bs_init sets `i_left = (WORD_SIZE - offset) * 8`, up to 128 bits for a 64-bit accumulator (bs.h:84).
- bs_write takes the non-`WORD_SIZE == 8` branch, written for 32-bit words (bs.h:115).
- endian_fix byte-swaps only the low 32 bits, since `WORD_SIZE == 8 ? endian_fix64 : endian_fix32` (osdep.h:282).
- Bits shift out of `cur_bits` before they are stored.

**The candidate fix:** osdep-word-size.patch, `#define WORD_SIZE sizeof(intptr_t)`. It makes the Capstone writer equal to
Plain's in this test. It is not applied anywhere.

## What this does not show

- **Whether the full 156-frame stream** (BuckBunny_New.264, 0e453c44…) also fails to decode. The same mechanism
  predicts so, from its first header on, but it is not read.
- **The time effect.** Capstone's x264 runs a different bit-writer branch from Plain's. The encoder's decisions and
  reconstructed frames are equal, so the analysis and motion search, which are most of the run, are the same work. The
  bit writer's share of x264's time is not measured.
- **Other programs.** Whether any other SPEC program uses sizeof(void*) as its word size is not audited.

## Decisions

These are for the runtime lane and the lead, not this folder: a src.alt portability patch (this one) with a re-run of
x264 on every Capstone arm and new grader keys; keeping x264 and reporting its stream as invalid; or dropping its row.
