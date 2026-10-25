# pb13q15 case 08 with its buffer's usable extent filled: pre-registration

Written 2026-10-26 before either run. The runtime lane's queue item 3, on the lead's "resume all experiments and all
work".

**The question.** pb13q15 (this corpus's `virtual-nested-pools` arm, patch 0006 with the Q-15 placement repair) reads
case 08 NO-READING:control-failed. The buggy run's own precondition `CHECK(ptr > rcurr + rbytes, 785)` fails because
the space scan stops at the first byte of cache.c's Q-15 tail pad, which is inside the read buffer's bound and is not
a space. The case already fills everything that follows the buffer with spaces. The variant does the same for the
pad, so the scan can leave the bound.

**The change** (shared/corpus.h and case 08):
- `MC_FILL_USABLE_EXTENT(rbuf, rbytes)` sits on the line that sets `rbytes`. With `-DMC_FILL_USABLE_EXTENT_ON`
  (passed through build-cases.sh's new `MC_CASE_CFLAGS`), it fills [rbuf + rbytes, end of rbuf's capability) with
  spaces. Without the flag it is `((void)0)`.
- **Measured before this pre-registration**, at one output path with the r60 SDK and tier3-prebuilt-build.sh's
  options for b13:
  - The unflagged images differ from the pre-edit images only in DWARF 5's per-file MD5 checksums: 32 bytes of
    `.debug_line`, 64 for case 08. With `llvm-objcopy --strip-debug`, all 9 cases and control 90 hash identically.
  - Two builds of the same source are byte-identical.
  - Stripped, two different cases hash differently, so the method can fire.
  - With the flag, only defect-08 changes. Its `mc_case_body` gains an `lcc` that reads the capability's end, and a
    memset.

**The runs**, one pair differing only by the flag:
- **Images:** pb08/b13-unflagged and pb08/b13-fill.
- **Runner:** `tools/run-virtual-cases.py --only 08 --arm virtual-nested-pools`, this tree's runner, as the batch
  reference ran pb13q15.
- **Kit:** qkit7-b2twin3-q8, QEMU domain B.

**Predicted readings:**
- **Unflagged:** NO-READING control-failed, with CONTROL-FAILED 785 on the buggy run and the fixed run FIXED. This is
  the record's reading for pb13q15 08 (batch reference and b2 reference).
- **Fill:** CAUGHT. The fault is a bounds fault, cause 28, in `mc_case_body`, which is the case's declared fault site
  (the scan), at the first byte past the read buffer's capability, before mark(8). The fixed run reads FIXED, exit 0:
  the fix's bound stops the scan inside the buffer, and the fill does not affect that run's verdict.
- The port control (chunk-free-90) and the heap's controls behave as `tools/arms.json` declares for
  virtual-memcached-sublet.

**What counts against the variant:**
- a fault anywhere but the scan;
- a FIXED run that fails;
- completion with VERDICT DEFECT-REPRODUCED.

A fault in cache_free, for example, would mean the fill reached the successor's lifetime slot, which this analysis
rules out: the pad ends at the bound, and the slot belongs to the next malloc object.
