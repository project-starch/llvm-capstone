# Case 31 (fz10) diagnostic: what silicon decodes at the out-of-window cursor; pre-registered 2026-10-20 before its board boot

**Why.** q15's case 31 is CAUGHT on QEMU (cause 28 at statDecodePage's getVarint32 load) but MISSED on silicon, twice
on tree A (records/q15def and q15rf).
- The window counter measured on QEMU that the load's pointer, `&aData[iOff]`, sits 51,000 bytes past a 20,160-byte
  page buffer, outside the compressed format's representable window.
- Decoding the buffer's fields at that cursor gives the true bounds shifted by one 64 KiB span, which CONTAIN the
  cursor.
- The decode half is measured in simulation (docs/history/19-10-2026_22-30-00_q30-silicon-half-and-the-issue-guard-
  measured.md). Whether the load's bounds check uses that decode is the open link.

**The image.** sublet/diag31-statdecodepage.patch adds to statDecodePage, just before
`iOff += getVarint32(&aData[iOff], nPayload)`, one printed line when `&aData[iOff]` is at or past aData's true end (at
most 4). The line carries aData's bounds (read in window), the cursor, and the LCC base/end of the out-of-window
pointer itself. It adds no load; the program's own load stays the test.
- Built with build-virtual.py's REPRO322_EXTRA_PATCHES hook, which records the patch in build.json as extra_patches:
  the full 33-case set plus both controls.
- SDK t3-freeze-nb, amalgamation 7fc92e19, --ext-src. Case 31 349e5418b72bd614, controls uaf 5acca3a2e66f9360 and
  bounds 6d2028dcccce63e8.
- EXPECT_IMAGES diag31.images.SHA256SUMS (3 images, --only 31).

**QEMU control** (qkit7-rev2, module a5dec6a32c49b3d3):
`DIAG31 adata=0x3f8eb61800:0x3f8eb666c0 cursor=0x3f8eb72de0 past_end=50976 q_decoded=0x3f8eb61800:0x3f8eb666c0`, then
cause 28 at the load with address = cursor: CAUGHT, controls fault. q_decoded equals the true bounds, because QEMU
keeps them in the register.

**Prediction for the board** (the qemu_compact_repr model, decoding aData's fields at the cursor; the image region is
4 MiB-aligned on both platforms, so offsets mod the 64 KiB span carry over):
- q_decoded = aData's bounds shifted by +0x10000 at both ends, so [adata_base+0x10000, adata_end+0x10000), with the
  cursor inside;
- past_end 50976;
- no fault at the load; the case completes (MISSED); controls fault.

**What decides.** q_decoded shifted by one span, with the load not faulting, confirms that silicon checks this load
against cursor-relative bounds: a far overrun aliases instead of trapping. q_decoded equal to the true bounds, with
the case still completing, would put the miss elsewhere (the load not checked at all). A fault would mean the miss
is not deterministic.
