# Track B, vidstab: a pool defect inside a third-party library, run as real code (capstone-qemu, 2026-09-29)

**Question.** Until now the vidstab defect ran only as the buffer-pool port's probe case 38, which
models libvidstab as one retained pointer. This run asks two things:

- Does the defect run as FFmpeg's and libvidstab's own code? The defect is the separate-buffer
  path upstream removed in `316531e61c`, and libvidstab keeps a shallow pointer to the plane.
- Does the Sublet port of FFmpeg's pools catch it? Here the stale access is not FFmpeg's at all:
  it is a write inside an opaque library.

**Answer: yes, 12 of 12 cells as registered, on the final images.** The registration is
`PREDICTIONS.md` (`dadb8bcd7e60`) with its addenda (`a2e729d7f93e`, `4cf916b3c4f1`, `d558a207ee57`,
and a fourth correcting the second). Two earlier runs never reached a touch line:

- run 1 halted in the filter's configure (`config_input`);
- run 2 halted in libavfilter's writable copy of frame 1.

Each was stopped by a porting defect beneath FFmpeg, recorded and fixed before the next run
(below). Predictions never changed.

## The experiment

- **libvidstab 1.1.1 is compiled into the image**, pinned in `deps/libvidstab.json`. The build
  compiles every source its CMake build does, with the image's flags. FFmpeg's configure finds it
  through a pkg-config file.
- **A real graph over pool frames.** 16x16 `yuv420p` frames from FFmpeg's `yuv4` decoder go
  through `buffer` → `vidstabtransform` → `buffersink`, with `crop=keep`, the default. The
  transforms file, eight frames of zero motion, is written by the fixture and read by the filter's
  own `config_input`.
- **The sequence:**
  1. Frame 1 (luma 0xA0) goes in while the fixture keeps a reference, so it is not writable.
  2. Frame 1's buffer goes back to the decoder's pool.
  3. A victim frame (0x77), held by the fixture, takes that buffer again.
  4. Frame 2 (0x5B) goes in as the graph's only reference, so it is writable.
- **Fixture 22 links `vf_vidstabtransform.o` as shipped.** Its pad makes frame 1 writable first,
  and libvidstab keeps its own copy.
- **Fixture 23 links it with upstream's fix reversed.** The reversal comes from the fix's own diff
  (`trackb/`).
  - Frame 1 takes the separate-buffer path, and `vsTransformPrepare` keeps `td->src` pointing
    into frame 1's plane.
  - Frame 2 takes the in-place path, finds `td->src` set, and copies frame 2 into it
    (`transform.c:150`).
- **The pair is held to the fix.** Three gates:
  - `make`'s own command must reproduce the archive's object byte for byte;
  - so must the out-of-tree route the revert is compiled by;
  - re-applying the fix to the revert must give back the shipped file.
- **What differs between the two images.** Both fixtures link their object ahead of the libraries.
  - The images differ in that object and in `vsTransformGetDestFrameInfo`, the one libvidstab
    function only the reverted code calls.
  - Every other named symbol (`.L` labels aside) sits either at the same address, or shifted by
    the object's growth (+440 bytes), or by that plus the extra function's 40 bytes (+480).
  - The fixture objects differ as well, in the fixture id and the transforms file's path.
- **A native control ran before any domain image was built.** x86-64, ASan, stock pools, the same
  sequence and revert:
  - the victim reads 0x77 shipped and 0x5b reverted, at the same address;
  - ASan reports nothing, because a pool never frees the plane.
  - That first run's output was not kept. It was re-run with output captured, with the same
    binaries and a revert file byte-identical to the domain's, and gave the same values
    (`result-lines.txt`, section 0).

## Result (final images, N = 3 per cell)

| | poolstock (FFmpeg's pools as shipped) | poolsublet (the Sublet port of the pools) |
|---|---|---|
| **22, fix present** | RETURN `1600177` ×3: the victim keeps its 0x77 | RETURN `1600177` ×3 |
| **23, fix reverted** | RETURN `170015b` ×3: **the victim, a live frame the program holds, now reads frame 2's 0x5b** | **FAULT temporal ×3**, after the touch line, at frame 1's plane (`0x102c07000`) |

The bottom-left cell is the defect's consequence as upstream described it: corruption of memory
the caller no longer owns, observed in the new owner. The bottom-right cell is the protection.

**Where the fault is.** It is in libvidstab's copy into `td->src`, not in anything the fixture
does.

- **The instruction.** The fault is at `memcpy`+0x9c, `stc a5, 0x0(a3)`. `memcpy` is tail-called
  from `vsFrameCopyPlane`'s equal-linesize path, called from `vsFrameCopy`.
- **The registers name the call.** The copy's destination frame is `&td->src`, its source is
  `filter_frame`'s stack `inframe`, and the source plane is frame 2's. That is
  `vsTransformPrepare`'s in-place copy (`transform.c:150`), not either of the KeepBorder copies,
  which involve `td->destbuf`.
- **The address.** The faulting value is frame 1's plane, printed while that plane was live.
- **The value carries metadata** (`value_hi` non-zero). That fits a revoked capability reloaded
  untagged, not an integer. The same load and store completes on poolstock.

**Both arms create the condition.** On both, the victim takes frame 1's address
(`same-address=1`) and frame 2 is writable. The victim's buffer and frame 2's are disjoint.

As in the af_join run, the emulator is deterministic, so N = 3 shows repeatability, not three
independent samples.

## What had to be fixed first (both beneath FFmpeg, both recorded as DIFFERS in `result-lines.txt`)

**1. The Sublet heap's `malloc(0)` halted the domain.** A runtime bug, fixed in `4cf916b3c4f1` and
landed on dev as `b586413fb7f6`.

- The heap carves a one-byte block for a zero request. It then narrowed the capability with the
  original 0, and capstone-qemu refuses an empty shrink.
- **The call is libvidstab's**, for a frame without local motions:
  `vsMotionsToTransform` → `meanMotions` → `localmotions_getx` → `vs_malloc(sizeof(int) * 0)`
  (`transformtype.c:407`). All six first boots, on both arms, halted there in configure.
- It reaches `malloc(0)` because `vs_malloc` is `av_malloc`, and this build disables
  `posix_memalign`, `memalign` and `_aligned_malloc`.
- An earlier version of this record, the second addendum, `4cf916b3c4f1`'s message and
  `b586413fb7f6`'s message on dev, named `vsSimpleMotionsToTransform`, which never runs here.
  That is retracted on dev in `36962080a40c` and in the fourth addendum.
- `malloc(0)` now returns a one-byte object, as musl's own returns a unique pointer. dev's heap had
  the same bug. No earlier result can depend on the old behaviour, because it was a halt.

**2. libavfilter's frame pool rebuilt plane pointers from integers.** A port gap, closed by app
patch 0005 (`d558a207ee57`).

- `framepool.c` aligned each plane as `(uint8_t *)FFALIGN((uintptr_t)p, align)`. The compiler lists
  that line under `-Wcapstone-pointer-roundtrip`, with two more on the audio path.
- No earlier run reached the pool. Fixture 22's own as-shipped copy of frame 1 was the first thing
  to, and `memcpy`, called from `image_copy_plane`, then stored through an integer
  (`value_hi = 0`).
- 0005 applies patch 0002's `cap_align_ptr` to the three sites. The build's round-trip list went
  from 24 sites to 21, with no `framepool.c` entry left.

## What this does not establish

1. **QEMU only** (Q-11): on silicon a stale access retires. The images also depend on
   capstone-qemu fabricating `gp`.
2. **Revoked versus integer is read by hand.** The classifier calls any "requires capability"
   fault at the target temporal, and it classed run 2's integer fault the same way. That the final
   fault is a revoked capability rests on reading the three register dumps.
3. **The poolstock overwrite is inferred, not observed at its site.** The victim reading 0x5b with
   fixture 22 at 0x77 on the same arm attributes it to the revert. The write instruction itself is
   seen only on poolsublet.
4. **Scope.**
   - Zero motion, so the transform is an identity. The defect is in the buffer handling, which
     does not depend on motion. Motion estimation (`vidstabdetect`, pass 1) was not run.
   - `crop=keep` only, 16x16 `yuv420p`, one non-writable → writable alternation.
5. **Writability is set by the fixture.** It keeps its own reference
   (`AV_BUFFERSRC_FLAG_KEEP_REF`). Upstream's trigger was the 8.1 scheduler making some frames
   non-writable: the same condition from a different cause.
6. **This build's configuration.**
   - libvidstab's `malloc(0)` is reached only because this build disables aligned allocation.
   - Patches 0004 and 0005 change every FFmpeg build of the app port. The committed Track A and
     af_join results predate 0005, and neither used the frame pool.
   - The heap fix changes every Sublet-heap image.
7. **Infrastructure and rootfs state.**
   - Two boots of the final run stalled after init, before the login prompt, and were killed by
     hand. The poolsublet one had been frozen 12 minutes; the poolstock one's duration was not
     recorded.
   - Every boot of this lane on the shared `rootfs.ext2` (vidstab tries 1-8, af_join 11-13) logged
     ext4 errors. The remaining cells booted a private copy of the rootfs alone, taken while
     holding the QEMU lock and repaired with `e2fsck`; the private boots logged none. Kernel and
     firmware stayed the shared ones, and the shared image was left untouched.
   - So rep 1 ran on the corrupt shared rootfs and reps 2-3 on the repaired copy.
   - One boot of run 2 failed at guest setup, at the 9p mount (an infrastructure flake, exit 75).
     Every attempt is kept in `result-lines.txt`.

## Files

- `PREDICTIONS.md`: the registration and its four addenda.
- `result-lines.txt`: every run in order, with the native control, the gates and the pair layout.
- `SHA256SUMS`: the final images per arm, and the libvidstab tarball. The paths name artifacts in
  the run's work directory, not files in this folder.
