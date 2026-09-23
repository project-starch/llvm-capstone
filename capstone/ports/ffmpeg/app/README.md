# FFmpeg as a full application in a Capstone domain

**Scope.** FFmpeg 9.0.1, configured down to a single-threaded file-to-file program
(matroska demuxer → mpeg4 decoder), running whole inside a pure-capability domain on the
musl-capstone libc. It prints one framemd5-style line per decoded frame.

This is **not** `ports/ffmpeg/buffer-pool/`. That port replays recorded pool calls with the
decoder running natively; nothing here touches it. It *is* the same workload: the "short"
recording from `buffer-pool/host/record.sh:23-34`.

- **Plan and evidence:** `capstone/docs/plans/2026-09-23-ffmpeg-full-app-port.md`.
- **Source pin:** `upstream.json`, verified by sha256 before every build.

## Status

| milestone | state |
|---|---|
| **M0** builds and links as a domain, within budget | **reached 2026-09-23** (apollo) |
| M1–M5 | **blocked**: the shared QEMU `rootfs.ext2` is corrupt (inode 623), and its repair belongs to the board lane. `host/run-qemu.sh` is ready |
| M6 | silicon ABI, see the plan's §4 (one-translation-unit question) |

**What M0 establishes:**

**All six images link.**
- They are the five staged milestones plus the M5 flipped-input control.
- The libraries are `libavutil`, `libavcodec` and `libavformat`, built for `capstone64` with 0
  errors.

**The budget fits, sized the way the kernel module sizes it:**
- `code_len` = 3,618,720 B, the `PT_LOAD` span through `p_memsz`, including the 1.5 MiB level0
  heap arena.
- A `.capstone_domreq` declaration adds 256 KiB of stack.
- The allocation is then **4 MiB = the order-10 ceiling**, with **305,248 B of headroom**. That is
  tight: the arena cannot grow much.

**The negative control fires.** Relinked without `hostcall.o`, the image leaves exactly
`__capstone_hostcall` undefined. FFmpeg's libc calls reach the hostcall, and nothing else is
missing.

**The compiler flags 21 pointer round trips (`-Wcapstone-pointer-roundtrip`), down from 28.**
- The 7 removed are exactly the sites the patches fix.
- 16 of the remaining 21 are integers carried in a `void*` and never dereferenced. Their lowering
  is a no-op, so they are safe.
- The last 5 are off the demux/decode path; see the plan's §3.

**What M0 does NOT establish:** that the image runs. Every runtime claim starts at M1.

## Build and run

```
bash host/build-native.sh    # the oracle: stock ffmpeg makes the workload + reference framemd5;
                             # the native build of this decode core must MATCH it, and the
                             # 1-byte-flipped input must NOT (positive control)
bash host/build-domain.sh    # M0: runtime objects, cross-configured FFmpeg, 6 images,
                             # budget gate, negative control, guest host
bash host/run-qemu.sh 1      # M1 .. 5; takes $CAPSTONE_QEMU_LOCK; stage 5 also runs the
                             # flipped-input control and compares both against the reference
```

**From a git worktree:** it has no LLVM build, and its buildroot submodule is empty. Export
`CAPSTONE_LLVM_BUILD_DIR=<main clone>/llvm/cmake-build-debug` and
`CAPSTONE_BUILDROOT_DIR=<main clone>/capstone/caplifive-buildroot`.

**Prerequisite:** `ports/musl-capstone/build-musl-capstone.sh`. If the LLVM build has no
`llvm-ar`, run it with `CAPSTONE_LLVM_AR=/usr/bin/llvm-ar-18`; the archive format is
target-independent.

All work products go under `$CAPSTONE_TMP_ROOT/ffmpeg-app/` (`FFAPP_WORK`). The tunables are
`FFAPP_ARENA_BYTES` (1.5 MiB), `FFAPP_STACK_BYTES` (256 KiB) and `FFAPP_JOBS` (48, apollo).

## Layout

| path | what |
|---|---|
| `patches/0001-capstone-log-callback-stays-a-pointer.patch` | `log.c` kept the log callback as `atomic_uintptr_t`. That initializer does not compile for capstone64, and a call through it would jump through an untagged integer |
| `patches/0002-capstone-keep-pointer-provenance.patch` | `av_x_if_null`, `av_stristr`, `avcodec_fill_audio_frame`'s const cast, and `frame.c`'s three `FFALIGN`-on-a-pointer sites. They now align by adding an offset, and NULL stays NULL |
| `src/shared/ffapp_decode.c` | the decode core, staged `M1..M5` (`ffapp_decode.h` has the codes) |
| `src/native/ffapp_native.c` | native entry, the oracle's self-check |
| `src/capstone-domain/ffapp_domain.c` | `capstone_main`: sets `__environ` (`getenv` needs it), compile-time input and stage |
| `src/linux-guest/ffapp_host.c` | the guest host: musl-capstone stdio-probe's host with a bounded 200k-round loop |
| `host/` | prepare, build, run, and `compare-md5.py` (negative-tested: mismatch, truncation, a control that cannot fire, and an empty reference all fail) |

**Why no CMake yet.** `ports/README.md` describes CMake presets, and this port follows the shell
precedent of SQLite's domain build instead. A domain image here needs a cross `configure`, runtime
objects, a budget gate and a link-time control, and none of those is in `capstone-domain.cmake`
today. Moving them there is shared-infrastructure work, so it goes to `dev` as its own change.

## Known limits

- **One workload:** 320x180, 1 s. 640x360 needs 2.2 MB of native heap and does not fit one
  region.
- **The heap is measured natively only** (0.71 MB peak). The capability build's own peak is
  measured at M4.
- **`localtime_r` and `mktime` are UNRESOLVED** (timezone), and are not expected on the decode
  path.
- **`input.mkv` bytes are NOT reproducible** across generations: Matroska writes a random
  SegmentUID, and two generations differ in 44 bytes. The oracle is the decoded frames, which
  are identical (`663177980a01…`, as committed in the buffer-pool measurements).
