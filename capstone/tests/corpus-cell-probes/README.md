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
