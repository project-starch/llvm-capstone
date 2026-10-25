# pb13q15 case 08 with its buffer's usable extent filled: read 2026-10-26

**Both readings hold as pre-registered** (PREREGISTRATION.md, lane 8cd19dfb30ee).

The run: one boot per image on qkit7-b2twin3-q8, QEMU domain B, with this tree's runner (9b5e409c92e1fe66) and the
r60 SDK.
- Kit: QEMU 708bbff5663f0e86, module 3325473537072a3a, vexec fad465ce4dea7121.
- Images: defect-08 is 4722c4afa1754616 unflagged and b1f8b63798e3b403 with the fill; control 90 is 76dc9b516c409d38
  in both.

| image | controls | case 08 |
|---|---|---|
| unflagged | bounds-malloc, uaf-malloc, chunk-free-90 fault (as declared) | NO-READING control-failed, `CONTROL-FAILED 785`. This is the batch and bitstream-2 references' reading |
| fill (`-DMC_FILL_USABLE_EXTENT_ON`) | the same | **CAUGHT**: cause 28 in `mc_case_body`, the declared fault site (the scan). The fixed run printed VERDICT FIXED and exited 0 |

The buggy run's fault address ends in `0x…080`, a multiple of 32, which is the read buffer's padded granule (16384 +
16 rounded up to 16416). The fixed run's `Q15_PAD items=2 padded=2 pad_bytes_total=32 pad_bytes_max=16` shows the
16-byte tail pad that the unflagged scan stopped on. This run does not measure the object's base, so "the first byte
past the bound" is consistent with the address, not proven by it.

So the q15 arm catches case 08 once the case's own premise also covers the pad: everything after the buffer is spaces.
The unflagged reading was the case refusing a verdict, not the arm missing the defect.

Bundles: `unflagged/` and `fill/`.
