# FFmpeg as a full application in a Capstone domain

Current builds use the [shared ABI-v2 SDK and runner](../../common/application/README.md).
Set `CAPSTONE_VM_STATE` to a running application VM before `host/run-qemu.sh all`.
The runner checks real process results and compares frame hashes with the native
reference and changed-input control. `host/run-safety.sh` uses separate stdout,
stderr, QEMU diagnostics and fault records; set `FFAPP_SAFETY_OUT` to a new output
directory. The private `ffapp.user` HostCall executable is removed. Historical
serial logs below retain their original interpretation and binary identities.
The qualified source setting is `-O1 -fno-omit-frame-pointer`; configure's default
`-O3 -fomit-frame-pointer` exposed an unaligned capability store in
`h263_decode_init_vlc` with the selected compiler (C-70 in the
[issue registry](../../../docs/ref/ISSUES.md)).


**Scope.** FFmpeg 9.0.1, configured down to a single-threaded file-to-file program
(matroska demuxer → mpeg4 decoder), running whole inside one Capstone domain on the
musl-capstone libc, on QEMU. It prints one framemd5-style line per decoded frame.
- **Not "pure-capability".** That wording was withdrawn on 2026-09-23 and survived here until
  now: the domain relies on QEMU's fabricated `gp`; see Status.
- **Correct is not safe.** The run of record's heap has arena-wide bounds and no revocation.
  What each heap arm actually protects is measured in `results/2026-09-23-qemu-safety/`.

This is **not** `ports/ffmpeg/buffer-pool/`. That port replays recorded pool calls with the
decoder running natively; nothing here touches it. It *is* the same workload: the "short"
recording from `buffer-pool/host/record.sh:23-34`.

- **Plan and evidence:** `capstone/docs/plans/2026-09-23-ffmpeg-full-app-port.md`.
- **Source pin:** `upstream.json`, verified by sha256 before every build.

## Historical qualification (before the delegated migration)

| milestone | state |
|---|---|
| **M0** builds and links as a domain, within budget | **reached 2026-09-23** (apollo) |
| **M1–M5** | **reached 2026-09-23 on QEMU.** Per-frame MD5 is **bit-identical to native** (30/30, reference built from unpatched FFmpeg), and the flipped-input control fires. Run of record: `results/2026-09-23-qemu-m1-m5/`. **Relies on QEMU's fabricated cursor-0 `gp`**, which cannot exist on silicon; with fabrication off, the domain dies in musl-capstone's start code. That is common to every domain on this ABI |
| **Safety** | **measured 2026-09-23 on QEMU**, `results/2026-09-23-qemu-safety/`, with three heap arms (`FFAPP_HEAP`):<br>• **`level0`** (the run of record): heap overflows and every temporal error succeed.<br>• **`shrink`:** per-object heap bounds; overflows fault, temporal errors succeed.<br>• **`sublet`:** per-object bounds plus revoke on free; on QEMU every heap-overflow and temporal fixture faults, while the deployed silicon is documented to let a stale data access retire (ISSUES Q-11, measurements §7r).<br>M1–M5 are bit-identical on all three. Globals share their GlobalMerge group's bounds on this ABI (not on M6's) |
| **Pools** | **measured 2026-09-24 on QEMU**, `results/2026-09-24-qemu-pool-safety/`: FFmpeg's own `AVBufferPool` buffers and every refstruct object under the buffer-pool port's Sublet leases, inside the whole decoder (`FFAPP_HEAP=sublet FFAPP_POOL=0\|2`).<br>• **Correctness:** M1–M5 bit-identical 30/30 in every completed run.<br>• **Fixtures:** 14/14 pool fixtures as pre-registered. Stale pool reads and a stale unref succeed on `pool0` and fault or are refused on `pool2`.<br>• **Silicon caveat:** as for the heap arms (Q-11).<br>• **Hardened 2026-09-24** (`results/2026-09-24-qemu-hardening/`): every fixture cell at N = 3 (141 runs, all as predicted); fixture 16's stock control measured; a 5-second, 150-frame workload bit-identical on level0, sublet and pool2 |
| **Sublet port of the pools** | **measured 2026-09-29 on QEMU**, `../sublet/results/2026-09-29-qemu/`: FFmpeg's `AVBufferPool` and refstruct pools ported onto Sublet (`FFAPP_POOL=sublet`), their storage lent LINEAR by the Sublet heap, a return is a revoke; `FFAPP_POOL=stock` is the one-macro control.<br>• M1–M5 bit-identical 30/30 on both arms, flip control firing.<br>• Pool fixtures 11–17, 20, 21 as pre-registered, N = 3 |
| **Track B** | **measured 2026-09-29 on QEMU**: pool defects as FFmpeg's real code, each fixed defect re-introduced by the exact reverse of its upstream fix.<br>• **af_join** (fixtures 18/19), `results/2026-09-29-trackb-afjoin/`: 12 of 12 as registered.<br>• **vidstab** (fixtures 22/23, libvidstab compiled in), `results/2026-09-29-trackb-vidstab/`: 12 of 12 as registered.<br>• vp9 documented, not run; h264_refs not run (plan `docs/plans/2026-09-25-ffmpeg-full-port-and-sublet.md`, "Track B outcome") |
| M6 | the gp-captable (silicon) ABI, which is also what removes the fabricated-`gp` dependence. See the plan's §4 (one-translation-unit question) |

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

**At M0 the compiler flagged 21 pointer round trips (`-Wcapstone-pointer-roundtrip`), down from 28.** Patch 0003 later removed one more, leaving 20.
- The 7 removed are exactly the sites the patches fix.
- 16 of the remaining 21 are integers carried in a `void*` and never dereferenced. Their lowering
  is a no-op, so they are safe.
- The last 5 are off the demux/decode path; see the plan's §3.

**What M0 did not establish, and M1–M5 then did:** that the image runs.

Three defects stood between M1 and M5. They are recorded in the result folder, and fixed as
follows:
- **stdout lost after its first line:** fixed in the domain entry. Since 2026-09-24 the runtime
  also flushes, because a returning program now ends through `exit()` (merge `556863938d46`).
- **`EFAULT` reading a file on the 9p share:** fixed on `dev`, host bounce buffer
  `e852b3951476`.
- **Compiler miscompile ISSUES.md C-50:** worked around by patch 0003, plus a build gate.

## Build and run

```
bash host/build-native.sh    # the oracle: stock ffmpeg makes the workload + reference framemd5;
                             # the native build of this decode core must MATCH it, and the
                             # 1-byte-flipped input must NOT (positive control)
bash host/build-domain.sh    # M0: runtime objects, cross-configured FFmpeg, 6 images,
                             # budget gate and shared application SDK
bash host/run-qemu.sh all    # M1..M5 + the flipped-input control in the selected VM, then the MD5
                             # comparison. Also accepts a single stage 1..5.
```

**From a git worktree:** it has no LLVM build, and its buildroot submodule is empty. Export
`CAPSTONE_LLVM_BUILD_DIR=<main clone>/llvm/cmake-build-debug` and
`CAPSTONE_BUILDROOT_DIR=<main clone>/capstone/caplifive-buildroot`.

The recipe builds musl privately. Select the VM with `CAPSTONE_VM_STATE`;
platform binaries and memory settings belong to that VM's configuration.

All work products go under `$CAPSTONE_TMP_ROOT/ffmpeg-app/` (`FFAPP_WORK`). The tunables are
`FFAPP_ARENA_BYTES` (1.5 MiB), `FFAPP_STACK_BYTES` (256 KiB) and `FFAPP_JOBS` (48, apollo).

## Layout

| path | what |
|---|---|
| `patches/0001-capstone-log-callback-stays-a-pointer.patch` | `log.c` kept the log callback as `atomic_uintptr_t`. That initializer does not compile for capstone64, and a call through it would jump through an untagged integer |
| `patches/0003-capstone-mpegpicture-pool-opaque-is-a-real-pointer.patch` | works around ISSUES.md C-50. `ff_mpv_alloc_pic_pool` passes a real pointer to a static mode, not an int through `void*` |
| `patches/0004-capstone-ff-field-at-keeps-alignment.patch` | works around ISSUES.md C-69. `FF_FIELD_AT` states the pointee's alignment under `__CAPSTONE__`, so libavfilter's format negotiation stops loading pointers byte by byte, which drops their tags |
| `patches/0005-capstone-framepool-keeps-pointer-provenance.patch` | libavfilter's `framepool.c` aligned its planes through `uintptr_t`; it now uses 0002's `cap_align_ptr` at the three sites |
| `deps/libvidstab.json`, `trackb/` | Track B: the pinned libvidstab release that `--enable-libvidstab` builds into the image, and upstream's vidstab fix, which fixture 23 links reversed |
| `host/scan-addi-sp.py` | C-50 build gate: an integer address off `sp`/`s0` used as a load/store base |
| `patches/0002-capstone-keep-pointer-provenance.patch` | `av_x_if_null`, `av_stristr`, `avcodec_fill_audio_frame`'s const cast, and `frame.c`'s three `FFALIGN`-on-a-pointer sites. They now align by adding an offset, and NULL stays NULL |
| `src/shared/ffapp_decode.c` | the decode core, staged `M1..M5` (`ffapp_decode.h` has the codes) |
| `src/native/ffapp_native.c` | native entry, the oracle's self-check |
| `src/capstone-domain/ffapp_domain.c` | ordinary `main`, input path from argv and compile-time stage |
| `../../common/application/` | shared SDK, launcher transport and safety verification |
| `host/` | prepare, build, run, and `compare-md5.py` (negative-tested: mismatch, truncation, a control that cannot fire, and an empty reference all fail) |

The shell recipe drives FFmpeg's cross configure and the shared CMake application SDK.

## Limits of the historical workload qualification

- **One workload:** 320x180, 1 s. 640x360 needs 2.2 MB of native heap and does not fit one
  region.
- **The heap is measured natively only** (0.71 MB peak). The capability build's own peak is
  **UNRESOLVED**: M5 completes inside the 1.5 MiB arena, which bounds it but does not measure it.
- **`localtime_r` and `mktime` are UNRESOLVED** (timezone), and are not expected on the decode
  path.
- **`input.mkv` bytes are NOT reproducible** across generations: Matroska writes a random
  SegmentUID, and two generations differ in 44 bytes. The oracle is the decoded frames, which
  are identical (`663177980a01…`, as committed in the buffer-pool measurements).
