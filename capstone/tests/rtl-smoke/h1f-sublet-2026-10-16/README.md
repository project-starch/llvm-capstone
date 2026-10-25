# H1 again on the Sublet bitstream, with the fixed Sublet runtime: pre-registration (2026-10-16, before the boots)

This settles the retraction on dev 783fb0f93f7e. `../h1-sublet-2026-10-16` read REVOKE 14-51 % lower than R-43 v2 at
every node count. Its image, `da9c54e578845735`, predates the sublet-runtime fix a07a11194fa0, whose carves misbehave
under this bitstream's full bounds scheme (ISSUES R-55). So the drop was not attributable.

## The image

The R1 harness, rebuilt from lane b0-silicon-runtime, which carries a07a11194fa0, with
`clang version 22.0.0git (llvm-capstone 612b3ec514c0)`:
- the recipe of `da9c54e5` (`h1-v2/build-h1.sh`): the default defines, entry 0x410000, -O1, stack 262,144;
- image **`66764b0e62075e57`**, with **14 SHRINK** words, the fix's instruction (`da9c54e5` has 0);
- emulator records for H1's four invocation types (latency 18 lines, calib 3, nodes 13, linear 15, as `da9c54e5`'s),
  and its pass record in `qemu-pass/`.

## The boots

Three cold boots, h1fs16r1..r3, exactly as `../h1-sublet-2026-10-16`:
- `drivers/lists/h1-v2-calibration.txt` lines 1-12 (six invocations), the stock k800 first and last;
- host 2c9e82d101b48160 on the classic line, `REFUSAL_RECORD=1`.
They run after SLT and P1b, before the size-100 boots.

## Pre-registered readings

- **As H1 and R-43 v2:**
  - latency 9 / 9 / 40 / 48 / 48 per load;
  - `cyc_cyc=2 ret_ret=1` in all three calibration domains;
  - nodes nd = 2n with ok=1 bad=0;
  - linear types 7,1,7,7,7,1,7,1,0,1,0,1;
  - k800 4 twice;
  - no refusal.
- **REVOKE (`rv`) at n = 1 / 4 / 16 / 64 / 256 decides it:**
  - **(a) the drop was the old image's carve:** rv reads as R-43 v2, 144-146 / 193 / 722-725 / 2,873-2,897 /
    11,106-11,293;
  - **(b) the drop is the bitstream's or the monitor's:** rv reads as H1 here, 71 / 145 / 608-634 / 2,336-2,370 /
    9,354-9,692.
  - Anything else is a third reading and is reported as such.
  - The REVOKE bracket is the instruction between two `mcycle` reads, so the tree and toolchain move it little.

**Not established:** N = 3, as H1.
