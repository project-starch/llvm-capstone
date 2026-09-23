# FFmpeg as a full application in a Capstone domain: M1–M5 on QEMU, bit-identical

**Verdict.** FFmpeg 9.0.1 (matroska demuxer → mpeg4 decoder), running whole inside one
pure-capability domain on musl-capstone:
- **demuxes and decodes all 30 frames** of the reference workload;
- its per-frame MD5s are **identical to native `ffmpeg -f framemd5`** on the same input (30/30);
- the flipped-input **positive control FIRES** in the same boot: 10 of 30 hashes change,
  exactly as they do natively.

This is QEMU, not silicon. See "What this does not establish".

## Identity

| | |
|---|---|
| source | branch `ffmpeg-app` at **`41355570eda7`**, FFmpeg 9.0.1 (`upstream.json`) plus patches 0001–0003 |
| images | `SHA256SUMS`, hashed **before** the boot; the guest booted byte-identical copies (re-verified after) |
| budget | `code_len` 3,619,024 B, allocation 4 MiB (order-10 ceiling) |
| C-50 gate | 0 hits in 354,666 instructions |
| pointer round trips flagged by the compiler | 20 |
| emulator | capstone-qemu, main clone build; `run-domain-smoke.py`, one boot, `$CAPSTONE_QEMU_LOCK` held |
| guest rootfs | a **private, fsck-repaired copy** of the shared `rootfs.ext2` (inode 623, `/var/lib/seedrng`). The shared image was left untouched: its hash was verified unchanged |
| workload | the buffer-pool "short" recording (`buffer-pool/host/record.sh`): 1 s, 320x180, 30 frames. Reference `stock.framemd5` = `663177980a01…`, as committed in `buffer-pool/results/measurements/20260919-replay/measurements.json` |

The input's own bytes vary per generation (Matroska writes a random SegmentUID), so the input
and its flipped twin are hashed here for this run only. The decoded frames are the stable oracle.

## Result (from `result-lines.txt`)

Each milestone image returned exactly the stage it was built for, all in the same boot:

| image | host requests | `capstone_main` |
|---|---:|---|
| M1 main | 1 | 1 |
| M2 `avformat_open_input` + `find_stream_info` → `streams=1 video=0 320x180` | 21 | 2 |
| M3 first packet, `size=10574` (native: 10574) | 23 | 3 |
| M4 first frame | 26 | 4 |
| M5 `frames=30 packets=30` | 122 | 5 |
| M5, flipped input (control) | 122 | 5 |

Then, outside the guest (`host/compare-md5.py`, whose failure paths are negative-tested):

```
oracle: reference 30 frames, candidate 30 frames, 0 hash mismatches -> MATCH
control: 30 frames, 10 changed hashes -> FIRES
```

No `capstone-domain: UNSERVED syscalls` line was printed: every syscall FFmpeg made was served.

## What it took, in order (details in the branch history)

1. **M0 (`eb53d4e`):** a cross-configured minimal FFmpeg links as a domain, within the 4 MiB
   budget only with a `.capstone_domreq` declaration. Two patches keep pointer provenance.
2. **Lost stdout:** musl goes fully buffered after its first flush (`ENOTTY`), and the runtime
   never flushes on return. The domain entry now sets stdout line-buffered and flushes.
3. **`EFAULT` on 9p reads:** the host `pread` into the region mapping failed for zero-copy 9p.
   This is fixed for everyone on `dev` (`e852b3951476`), and the port also stages its inputs to
   `/tmp`.
4. **A compiler miscompile, C-50** (`dev` `0e5b7b991629`): an integer-valued pointer in a
   by-value union is stored through an integer address. Worked around by patch 0003, and
   guarded by a build gate.

Every one of these was localised by a run that **returned** a result: staged images, a
diagnostic image built from a separate object, and a matched pair differing in one variable.
None was localised by guessing from a hang.

## What this does not establish

- **Silicon.** This is the musl / `link.ld` ABI on QEMU. The board needs the gp-captable ABI (M6)
  and its one-translation-unit question (plan §4). Silicon-only defects are not ruled out.
- **Heap and stack headroom under the capability build.** The run fits and completes, but the
  domain's peak heap was not measured, and 305 KB of allocation slack is little.
- **Any other workload.** One file, one frame size. 640x360 does not fit one region.
- **Performance.** TCG timing means nothing here.

## Reproduced from scratch, same day

A fresh `FFAPP_WORK` (`/tmp/capstone/ffmpeg-app-verify`) was run from the committed branch:
- the tarball was re-fetched and hash-verified;
- `build-native.sh`, `build-domain.sh` and `run-qemu.sh all` were run.

Same verdict: M1–M5 each returned its own stage, `oracle ... MATCH` (30/30), and the control FIRES
(10/30).

**The build is deterministic.** Against `SHA256SUMS` above, all six domain images and the guest
host rebuilt byte-identical (`OK`), and so did the reference `stock.framemd5`. Only `input.mkv` and
`input.flip.mkv` differ, as expected, because of the random SegmentUID.

The musl-capstone archive and the private rootfs copy were reused, not rebuilt. The first attempt
failed on a transient `Connection reset by peer` from the tarball mirror. `prepare-source.sh` now
retries.
