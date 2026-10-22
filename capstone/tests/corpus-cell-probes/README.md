# Probes of the corpus cells the batch bitstream does not read

Each item is pre-registered here and pushed before its run. The cells are listed in
docs/plans/2026-10-22-next-bitstream-whole-corpus-coverage.md, section (b).

## lt0 18 (gh-157335) with application mremap (pre-registered 2026-10-22, before its run)

**The blocker:** on the batch image, `mmap.resize` raises `OSError: [Errno 38]`, because the virtual profile has no
public mremap. llvm-capstone runtime/app-mremap c26b60a61f5b adds the MREMAP_MAYMOVE form as map-copy-unmap. The old
mapping is retired through the unmap service, and the new one is read-write.

**The image:** cpython-lt0 from t3-nomovc-r60, relinked through its own recorded link command with one archive member
replaced.
- The member is `mapping.c.obj` 4340ce84702a, rebuilt here from the branch's mapping.c with the batch SDK's own
  compile command. That command, on the SDK's source mapping.c, reproduces the original member ed0680b8081f.
- Every other member of lt0's runtime archive is unchanged. The relink with the ORIGINAL archive reproduces both batch
  images byte for byte: base 4512fdb6f43675a0, and `-dl` 23a2f0cbd80ddfbe.
- New images: base 1fb7dc2b59e09f46, `-dl` 7b871db70b7ec373. The defined-symbol name set equals the batch image's.

**The run:** qkit7-batch2, one boot. The lt0 arm's two block controls, then case 18 on the NEW image, then case 18 on
the batch image as the same-boot control.

**The prediction:**
- **Batch image:** `OSError: [Errno 38]` at `self.m.resize(new_size)`, as on the reference.
- **New image:** the resize succeeds. The 2-page map becomes 1 page, and the defect then writes index 8191, which was
  checked against the old size, through `mmap_ass_subscript`. It faults in `safe_byte_copy` or `mmap_ass_subscript`,
  the case's declared ASan sites:
  - cause 28 if the write goes through the new 1-page mapping's capability;
  - cause 25 if it goes through a pointer to the retired 2-page mapping.
  So the verdict is CAUGHT; never OSError 38, and never completion.
- **Both controls fault.**

**Result** (2026-10-22 15:14, qkit7-batch2, QEMU domain A, one boot):

| call | image | reading |
|---|---|---|
| control-uaf-block | batch 23a2f0cb | `CONTROL uaf-block RETURNED`, exit 0 |
| control-bounds-block | batch 23a2f0cb | `CONTROL bounds-block RETURNED`, exit 0 |
| 18 | mremap 7b871db7 | domain fault cause 28 in `mmap_ass_subscript`, address 0x3f802f5fff: offset 8191 into a page-aligned one-page mapping |
| 18 | batch 23a2f0cb | `OSError: [Errno 38] Function not implemented`, ABR EXIT 1 |

- **HELD:**
  - with application mremap, the resize succeeds and the defect's write faults at a declared site (`mmap_ass_subscript`);
  - it is the cause-28 branch: the write goes through the new mapping's capability, one page short of index 8191;
  - the same-boot batch image raises OSError 38.
  - So lt0 18 reads CAUGHT once mremap is in the image, and its batch MISSED is a not-reached.
- **NOT HELD: "both controls fault."** That line was my error. The lt0 arm's pymalloc-block controls COMPLETE by design
  (the virtual-malloc arm keeps pymalloc's arenas). The batch reference's lt0 verdict rows record both controls as
  `observed: complete`, and the judge accepts it. They read the same here. The case reading does not depend on them.

## lt1 18 (gh-157335) is record capacity (pre-registered 2026-10-22, before its runs)

**The reading that prompted this:** on qkit7-batch2, standalone probes import everything the case imports, from the
case's own directory, and succeed: `select`, `subprocess`, `test.support.script_helper`. Their module high_water
peaks at 59,646. In the same boot, case 18 raises `SystemError` at `import select` with high_water **64,023** against
capacity 65,536, which is the flat table's stop. The probes' run is /mnt/hdd/<user>/capstone/sel18,
readings.json per call. The case also compiles the large test_mmap module before those imports. So this run tests
whether the SystemError is the record limit surfacing as an allocation that fails without setting an exception.

**The runs:** case 18 on each of three kits, through run-gate.sh (capstone-qemu 98b34967b6's runner):
- qkit7-b2twin-q6, the 16-bit table;
- qkit7-b2twin-q8, the 18-bit table;
- qkit7-batch2-paged-lp, uncapped, with the live-peak counters.

**The prediction:**
- **q6:** `SystemError: error return without exception set` at subprocess.py:125, with high_water within 300 of
  64,023, as on the batch kit.
- **q8 and paged-lp:** no SystemError. `import select` succeeds and the test reaches `self.m.resize(new_size)`,
  where this image (no application mremap) raises `OSError: [Errno 38]`. Exit 1.
  - On q8, high_water is above 64,023.
  - paged-lp gives the case's live peak as a bracket. Only "above 64,001" is predicted.
- **If q8 still reads SystemError,** the record limit is not the cause, and this item is withdrawn.
