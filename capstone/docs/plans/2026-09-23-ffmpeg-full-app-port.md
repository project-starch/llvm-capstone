# FFmpeg as a full application in a Capstone domain — port plan (2026-09-23)

**Status:** a plan, not the port. It lives on branch `ffmpeg-app`.

**M0 was REACHED on 2026-09-23 (`eb53d4e`):**
- **Images:** six domain images link: M1–M5, plus the M5 flipped-input control.
- **Budget:** `code_len` is 3,618,720 B, including a 1.5 MiB heap arena. With the
  `.capstone_domreq`-declared 256 KiB stack that is a 4 MiB allocation, exactly the order-10
  ceiling, with 305,248 B spare.
- **Negative control:** it fires.
- **Pointer round trips:** 21 compiler-flagged, down from 28 after the two patches.

The port is in `capstone/ports/ffmpeg/app/`.

**M1–M5 were REACHED the same day, on QEMU.** FFmpeg demuxes and decodes all 30 frames inside a
domain, **bit-identical to native framemd5**, and the flipped-input control fires. The run of
record is `ports/ffmpeg/app/results/2026-09-23-qemu-m1-m5/`, at `41355570eda7`.
- **Friday target:** it was M0 + M1, and is exceeded.
- **Rootfs:** runs booted a private repaired copy of the rootfs; the shared image is untouched.
- **Three defects were found on the way:**
  - stdout lost after its first line (port);
  - `EFAULT` on 9p reads (host, fixed on `dev` `e852b3951476`);
  - compiler miscompile **C-50** (`dev` `0e5b7b991629`; patch 0003 plus a gate).
- **Audited the same day.** Three adversarial audits confirmed the result and exposed instrument
  holes, all since fixed. A fresh run of record was made at `5f05b2148b40`, with the reference
  now built from unpatched FFmpeg.
- **The audits also found the result's main limit.** On this `link.ld` ABI, every global access
  uses a cursor-0 `gp` that **QEMU fabricates and silicon cannot represent**. It happened
  ≥75,000 times in the run. With fabrication off, the domain dies in musl-capstone's start code
  before `capstone_main`. So M1–M5 prove FFmpeg correct under capability enforcement in QEMU,
  not runnable on the board as built.
- **What remains is M6** (silicon ABI) and the §4 one-translation-unit question. M6 is what
  removes the `gp` dependence.

**Question answered:** can FFmpeg 9.0.1 run file-to-file inside a Capstone domain the way SQLite
does, and what is the shortest path to evidence?

Every number below comes from a probe run on apollo on 2026-09-23. The probe build is throwaway: it
sits under `/tmp/capstone/ffmpeg-probe/` and is not committed. Anything the probes could not settle
is marked **UNRESOLVED**.

## Why FFmpeg, and the one-paragraph answer

Of the candidate applications, FFmpeg is the only one that reduces to a single-threaded,
file-to-file program. PostgreSQL forks per connection; tshark needs `epan/` plus GLib; memcached and
httpd are pools by architecture.

**The probes show the port is further along than expected.**
- The minimal FFmpeg configures for `capstone64` on apollo. It compiles to three domain libraries
  with **zero errors after one 3-line patch**.
- It links into a domain image with nothing undefined: 1.53 MB of text, which becomes a
  3.62 MB `code_len` once the 1.5 MiB heap arena is in `.bss`. It fits the 4 MiB domain limit
  at the reference frame size, **with only 305 KB to spare**, and only with a `.capstone_domreq`
  declaration. Without one the module would allocate `2 × code_len`, which does not fit.
- The compiler's own `-Wcapstone-pointer-roundtrip` finds every pointer→integer→pointer site:
  **28 of them**. 16 are provably harmless, and the other 12 have 1–3-line fixes.
- **The first milestone (M0: builds and links as a domain) is within reach by Friday.**
- M1 onward needs a working QEMU guest. The shared `rootfs.ext2` is currently corrupt (inode 623);
  that repair belongs to the board lane.

## The workload, pinned to the recordings the paper already uses

The native recordings behind the buffer-pool traces are made by
`ports/ffmpeg/buffer-pool/host/record.sh:23-34`:
- demuxer **matroska** → decoder **mpeg4** → muxer **framemd5**;
- `-threads 1`;
- the input is generated natively from `testsrc2` with `-c:v mpeg4 -q:v 3`.

The probe regenerated the **short** workload (1 s, 320x180, 30 frames) with those exact command
lines. Its `stock.framemd5` hashes to `663177980a01…aeda89`, **identical to the committed value** in
`buffer-pool/results/measurements/20260919-replay/measurements.json`. ~~The input the repo never recorded now has a hash: `input.mkv` = `2148cce4921727e8…38d5d1`.~~
**Withdrawn the same day.** `input.mkv` is not byte-reproducible: Matroska writes a random
SegmentUID, and two generations from the same binary and command line differ in 44 bytes. Their
decoded framemd5 is identical (`663177980a01…`). So that hash identified one file, not the
workload. The oracle is, and always was, the decoded frames. The repo's missing `input.mkv` hash
stays missing for the same reason.

**The domain program is a small libav\* driver, not the `ffmpeg` CLI.** The CLI requires
libavfilter, and the image budget below has no room for it. The driver:
- opens the file, reads packets and decodes them;
- hashes each frame packed with `av_image_copy_to_buffer(align=1)`, which is exactly what framemd5
  hashes (codec `rawvideo`);
- prints the frame lines.

The encoder, lavfi and the filters stay native: they only produce the input.

**Native validation of the oracle.** The probe driver (`decode_md5.c`, 80 lines) reproduces all 30
reference MD5s **exactly**. Flipping one byte of `input.mkv` (offset 77,850) changes 10 of the 30
MD5s, so the comparison is shown to fire.

## 1. Configuration

```
--enable-cross-compile --cc=$CAPSTONE_CLANG --ld=$CAPSTONE_LLVM_BIN/ld.lld --arch=riscv64 --target-os=none
--extra-cflags="<musl flags: -target capstone64-unknown-elf +m +a -ffreestanding -fno-builtin -nostdinc
                 -isystem musl/{arch/capstone64,arch/generic,obj/include,include} -isystem <clang resource>
                 -D_GNU_SOURCE -O1 -Wno-int-conversion -fno-jump-tables -ffunction-sections -fdata-sections>"
--extra-ldflags="-e main --no-warn-mismatch"
--extra-libs="libc-capstone.a <runtime objects>"
--disable-everything --disable-autodetect --disable-doc --disable-network --disable-asm
--disable-pthreads --disable-programs --disable-debug --disable-iconv
--disable-swresample --disable-swscale --disable-avfilter --disable-avdevice
--enable-demuxer=matroska --enable-decoder=mpeg4 --enable-parser=mpeg4video --enable-protocol=file
--enable-static --disable-shared
```

**Notes, each learned by running it:**
- **`--disable-iconv` is required.** `--disable-autodetect` leaves `CONFIG_ICONV=1`.
- **Configure needs a real capstone link, or it aborts.** The capstone64 clang driver hands linking
  to gcc, which fails with "C compiler test failed". Hence `--ld=ld.lld`.
- **Configure's `HAVE_*` link tests need the libc archive plus the soft-float builtins present.**
  Without `__floatdisf` the `lrint` test fails, `HAVE_LRINT=0`, and `libm.h`'s static fallbacks
  then collide with musl's `math.h` (275 of the 276 errors in the first build; the other is `log.c`). With them present, the `HAVE_*` values are exactly
  what musl-capstone provides.
- **Post-configure overrides in `config.h`:** `HAVE_POSIX_MEMALIGN 0` and `HAVE_MEMALIGN 0`.
  - `av_malloc` then uses plain `malloc` (`libavutil/mem.c:105-142`).
  - With asm off, `ALIGN` is 16 (`mem.c:65`), which is exactly level0's alignment
    (`musl-capstone/runtime/level0.c:39-42`).
  - musl's `posix_memalign` would sit on an allocator the port does not use.
- `HAVE_MMAP` and `HAVE_NANOSLEEP` stay 1: they link but return `-ENOSYS`, and nothing on the decode
  path calls them.

**Where the build runs: apollo.** The memory note "apollo cannot build the SQLite domain TU" does
**not** apply to this route.
- That failure is headers falling through to the host glibc (`bits/wordsize.h` under
  `/usr/include/<multiarch>`), because `build-sqlite-silicon.sh`'s `COMMON` has no header stubs.
- The musl `-nostdinc -isystem` route never reaches host headers. Configure and all three libraries
  build on apollo.
- The mechanism cited in `plans/2026-09-17-compiler-lane-handover.md:156-173` (the riscv64 Buildroot
  sysroot's `wordsize.h`) is not what fails here. That is flagged for its owner, not edited.

**Probe build, with the one 3-line patch below:**

| library | domain `.a` size |
|---|---:|
| `libavutil` | 2.05 MB |
| `libavcodec` | 2.06 MB |
| `libavformat` | 1.01 MB |

## 2. The libc surface, measured

The native minimal driver imports **101 libc symbols** (`nm -D --undefined-only`). Against
musl-capstone's served set (`runtime/hostcall.c`, README table):

| class | symbols | on the decode path? |
|---|---|---|
| **Pure libc, works** | mem\*/str\*/strto\*, snprintf/printf/puts/fputs, qsort, sscanf, strftime, gmtime_r, errno, abort, strerror_r, the libm set (acos…tanh, exp2, hypot, pow, round, sincos…) | yes |
| **Served by hostcall** | open, read, write, close, lseek, fstat, access, unlink, clock_gettime (→ gettimeofday, clock) | yes (open/read/lseek/fstat/close) |
| **Stub, harmless here** | fcntl (always 0), isatty (ENOTTY → false) | yes |
| **Needs `__environ` set** | getenv (`log.c`) | yes. **The port's `capstone_main` must set it**, as `libc-test/libc_test_domain.c:26-31` does |
| **UNRESOLVED: timezone** | localtime_r, mktime (`log.c`, `parseutils.c`) | only with timestamped logging or date parsing; not expected |
| **Links, returns `-ENOSYS` if called** | nanosleep (`time.o`), opendir/readdir/mkdir/rmdir/rename/lstat (`file.o` dir ops), mkstemp/fdopen (`file_open.o`), setvbuf (`random_seed.o`) | no |
| **glibc-only artifacts** | `__*_chk`, `__libc_start_main`, `__cxa_finalize`, `_ITM_*`, `__stack_chk_fail` | absent under musl, `-ffreestanding` |

**No blocker was found.** Any unexpected syscall shows up at exit as
`capstone-domain: UNSERVED syscalls: …` (`hostcall.c:572-649`), which acts as a runtime check of this
table.

## 3. Pointer-as-integer hazards: found by the compiler, not guessed

`-Wcapstone-pointer-roundtrip` flags every place where an integer derived from a pointer is turned
back into a pointer. Over the three libraries it reports **28 sites; 27 are in objects the driver
actually links**. A separate grep-and-read pass over the same `.c` files agrees on every compiler-flagged site in them. It
misses the one in the `avutil.h` inline helper (it never scanned headers). Its extra hits are
pointer→integer tests (`crc.c:437`, `md5.c:172`, `mpegvideo.c:462`, the `frame.c:619-630` compares),
which never turn back into pointers and are correctly not flagged. A lowering probe shows that
`(void *)(uintptr_t)n` compiles to *no instruction*, so an integer carried in a `void*` is safe
unless it is dereferenced.

| class | sites | fix |
|---|---|---|
| **Compile error, certain** | `libavutil/log.c:441,464,494`: the log callback is kept as `atomic_uintptr_t`. The initializer is rejected ("not a compile-time constant"), and it would be called through an untagged integer (`sd`, then `cjalr`). | **3-line patch** to a plain function pointer (the build is single-threaded). Applied in the probe. |
| **Faults if reached, small fix** | `avutil.h:313` `av_x_if_null` (opt.c to-string and help only); `avstring.c:61,65` `av_stristr`; `frame.c:135,191,200` `FFALIGN` on a pointer; `codec utils.c:402` (audio); `avsscanf.c:913` (`%p`); `allformats.c:605,626` device lists (null while avdevice is disabled) | **Patch all up front, 1–3 lines each.** Use a plain cast, or `p + (FFALIGN(a,n) - a)`. Cheaper than finding them one QEMU run at a time. |
| **Faults if reached, larger** | `options.c:81,119`: iterator state packed into a pointer's high bits (child-class iteration) | Patch **only if reached**. It is not on the demux/decode path. |
| **Safe: integer in `void*`, never dereferenced** | 16 iterator sites (`allcodecs`, `parsers`, `bitstream_filters`, `protocols` ×2, `avio`, `options.c:445`, `channel_layout`, `iamf` ×5, `mpegpicture`, `allformats` ×2) | none |

**Other hazard classes checked over the linked sources:**
- `__thread` / `_Thread_local` and constructors: **0**. Nothing depends on `.init_array`, which
  never runs.
- `ff_thread_once` without pthreads: 8 sites, and they build. **UNRESOLVED** until M2 exercises one.
- C11 atomics: 54 sites; they build with `+a`.
- Unaligned capability copies (`string_bounds_safe.c:110-132`): **UNRESOLVED**. They show up only at
  runtime, M2 onward.

**This is no longer the main schedule risk.** The ABI and memory are.

## 4. Memory and ABI: the real risks

**Budget at the reference frame size (320x180) against the 4 MiB `dom_data` cap**
(`sqlite/run-speedtest1-measure.sh:30-31`):

| item | size | source |
|---|---:|---|
| image text + data, domain link with `--gc-sections` | 1.53 MB + 0.26 MB bss | measured, `llvm-size` |
| heap | native peak **0.71 MB** (731 allocs) | measured with a malloc-counting shim |
| level0 arena to hold it | ~1.5 MiB | `-DCAPSTONE_LEVEL0_ARENA_BYTES` (`level0.c:35`); the default 256 KiB is too small |
| stack | ~0.25 MiB | SQLite's 2 MiB default is not needed. **UNRESOLVED** until measured at M4 |
| **total** | ~~≈ 3.3 MiB~~ **measured at M0: `code_len` 3,618,720 + 8 KiB + 256 KiB = 3.89 MB, which rounds to a 4 MiB allocation, 305,248 B spare** | the module's power-of-two rounding (`modcapstone/module/capstone.c:152-161`) takes the rest of the slack |

Two caveats on the heap figure:
- The capability build inflates pointer-heavy structs. The 0.71 MB is native; the domain figure is
  **UNRESOLVED** until M4.
- 640x360 needs **2.2 MB** of native heap and **does not fit**. It needs SQLite's CMA shared-region
  arena (`SPEEDTEST1_REGION_ARENA=1`), and is out of scope for the first pass.

**ABI: two different domain ABIs exist, and only one takes multi-file programs today.**
- **The musl consumers** (`stdio-probe`, `file-probe`, `libc-test`) link multi-TU with
  `my_first_domain/link.ld`. This is the port's path for M0–M5, on QEMU.
- **The silicon image path** (`sqlite/build-sqlite-silicon.sh`) uses `-capstone-gp-captable`, a
  two-pass `link-gpfree.ld`, and `start-gp-captable-interp.S`. It requires **every globals-owning
  file in one translation unit**: `getGpCaptableIndex` numbers globals per module and positionally.
  SQLite is an amalgamation. FFmpeg is ~130 linked objects with colliding statics.
- **UNRESOLVED:** whether that constraint still holds, and whether any gp-captable image has been
  built against musl. This is **the** question for M6, to put to the compiler lane before M5 lands.

## 5. Milestones, each of which returns a result

Each milestone is a separate build of the same driver, `-DSTOP_AT=n`, so a failure always says
where. This follows CLAUDE.md, "make every run return". The driver prints `STAGE Mn …` markers
(already present in the probe driver).

| # | milestone | returns | depends on |
|---|---|---|---|
| **M0** | builds and links as a domain with the **real runtime objects** (`start-musl.o`, `hostcall.o`, `tls.o`, `setjmp.o`, `level0.o` with a 1.5 MiB arena, soft-float builtins per `beebs/build-beebs-softfloat-common.sh`) and `link.ld`, under a checked 4 MiB budget | a `.dom` and its size | nothing; apollo only |
| **M1** | `capstone_main` runs and returns `STAGE M1` | the marker | a working QEMU guest (**rootfs repair, board lane**) |
| M2 | `avformat_open_input` + `find_stream_info` return; the stream is `320x180` | stream params | M1; the §3 patches |
| M3 | first packet read | its size (10,574 natively) | M2 |
| M4 | first frame decoded | frame 0's MD5 | M3 |
| **M5** | all 30 frames | **MD5 lines identical to `stock.framemd5`** | M4 |
| M6 | silicon-ABI build → the first run worth board time | same oracle | M5 and the §4 ABI question |

**Friday target: M0 for certain, and M1 if QEMU is back.** M0 is compile and link work that the
probes have mostly done already. What remains is writing the host side and the link line, both from
the `stdio-probe` template.

**Host side.** `musl-capstone/stdio-probe/` is the template: `stdio_probe_host.c` serves the file
hostcalls, and `run-stdio-probe.sh` runs under `flock $CAPSTONE_QEMU_LOCK`. The FFmpeg host
additionally stages `input.mkv` in the guest share.

## 6. Acceptance

- **Pass:** M5. The domain's per-frame MD5 lines are **bit-identical** to the native `stock.framemd5`
  for the short workload. Compare the hash column, 30 of 30.
- **Positive control:** in the same boot, the domain also decodes the 1-byte-flipped
  `input.flip.mkv`. Its MD5s **must differ** from the reference, as they do natively (10 of 30). A
  comparison that cannot fail is not evidence.
- **Negative control:** the native driver, with the same config and input, must match
  `stock.framemd5`. This is already shown.
- **QEMU first.** Board runs belong to apollo-board and are not scheduled here. **M6 is the one
  milestone worth a board run.** M5 on QEMU says nothing about silicon-only defects, and the ladder's
  history says those exist.

## 7. Stretch, tied to the paper: replay fidelity

With M5 in hand, record the real decoder's `AVBufferPool` get/return sequence **inside the domain**
and compare it with the recorded-trace replay the paper evaluates (`buffer-pool/host/memory/`,
`analyze.py`). Agreement is direct evidence that replay is faithful, and the paper's evaluation
method rests on replay.
- This touches the buffer-pool lane's trace format.
- **Do it only with that lane's agreement.** Nothing here modifies `ports/ffmpeg/buffer-pool/`.

## 8. Estimates and risks

Code time is kept separate from latency; latency dominates as soon as a guest is needed.

| step | code | iteration latency |
|---|---|---|
| M0: runtime link line, host, CMake/scripts, patches from §3 | ~1 day | build ~1 min (`-j48`), no guest |
| M1–M2 | ~0.5 day | QEMU boot + queue behind `$CAPSTONE_QEMU_LOCK`; **blocked while the rootfs is corrupt** |
| M3–M5 | 1–2 days, mostly runtime surprises (unaligned capability copies, arena size, stack) | a few QEMU runs per surprise |
| M6 | **UNRESOLVED**: depends on the one-TU answer. From days (if multi-TU gp-captable exists or is cheap) to a compiler-lane project (if not) | board time via apollo-board |

**Risks, in order:**
1. **The QEMU guest is unavailable** (corrupt rootfs), which blocks M1+.
2. **The silicon one-TU constraint** (M6).
3. **Domain heap above the native 0.71 MB estimate**, which would mean raising the arena or taking
   the region-arena escape.
4. **Runtime-only capability faults** (tag loss in copies), found one run at a time.
5. **Soft-float builtins that are missing**, which surface as link errors at M0; cheap.

**Out of scope:** threads, network, hwaccel, asm, libavfilter and the `ffmpeg` CLI, the encoder, the
640x360 workload, any board run before M6.

## 9. Where the work lives

| what | where |
|---|---|
| **Everything, this plan included** | branch `ffmpeg-app`, worked in its own `git worktree` so the shared tree's foreign edits never ride along |
| **Port directory** | `capstone/ports/ffmpeg/app/`, per `ports/README.md:43-60` (`upstream.json`, `patches/`, `src/capstone-domain/`, `host/`, `tests/`, `results/`). **Never** under `ports/ffmpeg/buffer-pool/`, which is the external collaborator's lane |
| **Shared-infrastructure fixes** | to `dev` directly, as their own commits: musl-capstone runtime or hostcalls, `ports/common` toolchain. Other lanes depend on them |
| **Probes** | `/tmp/capstone/ffmpeg-probe/`, never committed |

The throwaway probe tree is at `/tmp/capstone/ffmpeg-probe/`. It has:
- the verified tarball;
- the native, stock and cross (`xdomain/`) builds;
- the driver;
- the `log.c` patch;
- the lists of round-trip sites and libc imports;
- the heap shim.

It is the starting point for M0.
