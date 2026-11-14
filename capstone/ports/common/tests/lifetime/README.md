The short tests compile the patched allocators against a trap-route SDK. They
include the real allocator code. The test header replaces the selected CDERIVE
result with zero before executing the instruction. It can also force CRENEW to
return zero without ending the old lifetime. Normal operations use the real ISA.
FFmpeg's current patch has no CDERIVE sites, so its injection targets CRENEW.

Run on the build host with `CAPSTONE_LLVM_BIN` pointing to the SDK toolchain.
Choose a fresh output directory under `CAPSTONE_TMP_ROOT`. Supply prepared
SQLite 3.22 sources, the pinned FFmpeg 9.0.1 archive and the revision before the
fixes as the negative control.

```sh
bash capstone/ports/common/tests/lifetime/build.sh \
  "$SDK" "$SQLITE_SOURCES" "$FFMPEG_ARCHIVE" "$OUT" "$BASELINE"
bash capstone/ports/common/tests/lifetime/boot.sh \
  "$OUT/bin" "$BASE_INITRAMFS" "$QEMU" "$FIRMWARE" "$KERNEL" "$OUT/boot"
```

Set `CAPSTONE_QEMU_LOCK` to the shared evaluation lock when using that host.
The boot uses one hart and `sublet.dom_binfmt=1`. It runs from a separate tmpfs
working directory, as required by the loader. Python needs `pexpect`.

SQLite checks failed heap initialization, unchanged allocation counts, preservation
of a failed realloc's original bytes, both lookaside free lists, benign setup
failure with owned and supplied buffers, renewal failure and page-cache recovery.
The generator output must match the block patch exactly. FFmpeg checks both
renewal failures, entry destruction, subsequent allocation and outstanding objects
that survive pool uninitialization. Separate stale reads must fault with cause 25.
The old SQLite patch must fault or abort under the same injection. The old FFmpeg
patch must fail both its buffer and refstruct controls. The boot checker rejects
missing binaries, unexpected statuses and stale reads without cause 25.

These tests do not exhaust the hardware lifetime table or force generation wrap.
They model the documented zero result at the allocator call site. The stale
checks use actual kernel traps. The build uses the application port's FFmpeg
patch sequence and libavutil. It does not run application workloads.
