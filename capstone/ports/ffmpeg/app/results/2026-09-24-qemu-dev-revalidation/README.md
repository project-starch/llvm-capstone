# FFmpeg app on current dev: rebuilt from scratch and rerun against the pushed predictions (QEMU)

**Why this exists.** After the hardening round landed (`6238605`), `dev` took 28 more
commits (not counting merges) before this branch point:
- the domain runtime stack #81–#87;
- C-56's at-exit hook (`hostcall.c:130`, a weak default, now called unconditionally);
- a returning program now ends through `exit()` (`hostcall.c:811`);
- one shared list of libc overrides (`runtime/libc_overrides.sh`, adding `atexit_capability_safe`);
- the runner's refusal of images with undefined weak symbols;
- compiler fixes C-50…C-58. C-55 and C-58 change generated code without any error.

None of these touched this port. So the question was whether the port is still what its results
say, on the `dev` that now exists.

**Verdict (QEMU): the port holds on current dev, unchanged in every result it states.**

1. **Correctness.** Every arm reaches M1–M5, and M5 is bit-identical to native, with the
   flipped-input control firing:
   - level0, shrink, sublet, pool0 and pool2 at 1 s: 30/30, 10 changed hashes on the flip;
   - level0, sublet and pool2 at 5 s: 150/150, 5 changed hashes.
2. **Safety.** All 47 fixture cells came out as pre-registered, against the unchanged
   `host/safety-expect.txt`:
   - fixtures 1–10 and 16 on level0, shrink and sublet;
   - fixtures 11–17 on pool0 and pool2.

   Every predicted fault faulted, every predicted return returned its mark, and pool2's stale
   unref was refused with a clean exit (304). That includes the heap arms' fixture 16, whose
   predictions were pushed at `411732f`.
3. **The pools and the heap do the same work.**
   - **pool2 at 1 s:** payload 315,072 B, 43 blocks, 406 leases taken and revoked, 482 revocation
     walks.
   - **pool2 at 5 s:** 1,846 leases, 2,042 walks, and a heap of split 753 + mrev 3,398, so the
     node spend is again 6,236 per 150-frame decode.

   All equal the previous rounds (`../2026-09-24-qemu-pool-safety/`,
   `../2026-09-24-qemu-hardening/`), though the images differ.
4. **Only the shape of the exit path changed.** A fixture that returns now ends through the
   runtime's `exit()`. The marks survive it, and the classifier's clean-exit requirement for
   pool2 fixture 17 still holds.

**Stalls.** 5 of the 25 fixture boots (21 planned, 4 retries) stalled. Four produced no section;
they are kept in `result-lines.txt` and not counted:
- 3 stalled before login: shrink's first batch twice, and pool0 fixture 16;
- 1 stalled after login, most likely in the guest's copy from the 9p share (pool2 fixture 16).
  Its QEMU was ended after 2.5 minutes of silence.

The fifth, level0's first batch, is counted. It blocked for about 240 s before any fixture ran
(`task ext4lazyinit blocked for more than 120 seconds`) and then completed. So "silent for 2.5
minutes" is not proof that a boot would never have recovered. None of the four uncounted attempts
reached a fixture, and each retry was run whatever the attempt showed.

**The shared guest rootfs was corrupt for this whole round.** Every serial log shows
`EXT4-fs error ... block bitmap corrupt`, raised during boot (seedrng); logs from before 13:17
today show none.
- The shared `rootfs.ext2` was last written at 13:17, and `e2fsck -n` now reports uncorrected
  errors (exit 4).
- The stalls that never recovered end right after that init step.
- So this round's stalls are not shown to be the earlier rounds' class, and the corruption is the
  better-evidenced candidate. Earlier rounds, on a clean rootfs, did stall, so it is not the only
  cause.

The corruption is reported separately; nothing counted here depends on the rootfs beyond booting.

## What changed in the port

- **`host/build-domain.sh` links the shared override list** (`build_musl_overrides`) instead of
  three overrides compiled by hand. That picks up `atexit_capability_safe`, which the port had
  missed when it was added to the list.
- **The FFmpeg library cache key now includes the compiler:** size and mtime of clang and of every
  `libLLVM*`/`libclang*` it loads. This is a shared-libraries build, whose codegen lives in
  `libLLVMCapstoneCodeGen.so`, not the clang binary. Before, libraries compiled before C-50…C-58
  would have been reused silently.
  - Checked both ways by hand, with no artifact kept: the key changes when a copied build's codegen
    library is replaced, and stays the same when nothing changes.
  - A statically linked clang contributes just the binary.
- **Dated notes, no silent rewrites, on statements the runtime merges made stale:**
  - `ffapp_domain.c` (stdout);
  - `ffapp_pool.h` (C-56);
  - `README.md`;
  - `../2026-09-23-qemu-m1-m5/README.md`;
  - `../2026-09-24-qemu-pool-safety/README.md`.

## Method

- **Everything from scratch in a private root** (`CAPSTONE_TMP_ROOT=/tmp/capstone/ffreval-0924`):
  - musl-capstone's libc built with the current compiler, not the shared archive from
    2026-09-23, which predates C-50…C-58;
  - the native references;
  - every arm: level0, shrink, sublet, pool0 and pool2 at 1 s, plus level0, sublet and pool2 at
    5 s.
- **Toolchain:** clang `b7b31421e9fa` (build 2026-09-24 14:45, after the last compiler merge at
  14:07).
  - The project's freshness gate answered "could not check" (rc 2), because it does not cover
    `llvm-ar`.
  - The binaries' mtimes and the musl survey's recorded revision establish freshness instead.
- **Build gates:**
  - native oracle MATCH, and the flip control FIRES (1 s and 5 s);
  - every image FITS the 4 MiB budget (48 M-image lines across the 8 arms, plus the fixture and
    diagnostic images);
  - the link control fires;
  - `scan-addi-sp.py` (C-50's by-value gate) finds 0 hits;
  - 178 images carry 0 undefined weak symbols. The old images still carry
    `w __capstone_at_exit`, and the runner refuses them.
- **Runs:** every fixture cell once (N = 1) against the unchanged `host/safety-expect.txt`, and
  M1–M5 once per arm.
  - One predicted fault per boot, placed last.
  - Every attempt's log is kept.
  - A boot that stalls is retried, never counted.
- **The images are not the run of record's.** The compiler and runtime are new, so the comparison
  is by outcome, not by hash. `SHA256SUMS` lists this round's images.

## Files

- `result-lines.txt`: each M1–M5 boot's section, oracle and control lines, and every fixture
  boot's verdict lines, including the stalled attempts.
- `SHA256SUMS`: this round's M1–M6, fixture and host images for every arm, the 1 s and 5 s inputs,
  and the stock references.
- **What each boot ran.** The per-boot sidecars (`safety-*.log.sha256`) hash the host's share
  directory at prep time, 90 of 90 matching their arm's built images. They do NOT prove what the
  guest booted:
  - `run-safety.sh` prepares one shared share directory BEFORE it takes the QEMU lock
    (`host/run-safety.sh:46-55` vs `:67`, and `run-qemu.sh` likewise);
  - in this round a retry launched beside the queue replaced the share under a running boot. Both
    boots involved stalled before any fixture, so no verdict came from them.

  **What ties each counted verdict to its image:**
  - the loader-printed segment size in every counted fixture section matches that arm's image,
    47 of 47. 20 of those sizes also occur in another arm, and for those the other two checks
    carry it;
  - no other job's preparation falls inside a counted boot's window (audit);
  - arms that share a size differ in predicted outcome on other fixtures (pool0 vs pool2 on 12,
    13, 15 and 17).
- **The harness race is fixed since:** `run-safety.sh` and `run-qemu.sh` now use a share per
  invocation (`mktemp`), and hash exactly that directory. The runs in this folder predate the fix.
- **Other work in this private root, serialized by the same lock:** the port's M-infra gate boots
  and their private rootfs copies (`br-*`). It used its own share directory.
- **The comment edits cannot have changed the images.** The notes in `ffapp_domain.c` and
  `ffapp_pool.h` were made after the build. The level0 and pool2 arms rebuilt from the edited
  tree, into a separate directory, match `SHA256SUMS` byte for byte: 48 of 48, `sha256sum -c`
  exit 0. The check is shown to fail on one altered digest.
