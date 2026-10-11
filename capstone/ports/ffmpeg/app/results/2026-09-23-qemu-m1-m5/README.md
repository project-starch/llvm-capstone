# FFmpeg as an application in a Capstone domain: M1–M5 on QEMU, bit-identical

**Verdict.** FFmpeg 9.0.1 (matroska demuxer → mpeg4 decoder) runs whole inside one Capstone
domain on musl-capstone, under capability enforcement, on QEMU.
- It **demuxes and decodes all 30 frames** of the reference workload.
- Its per-frame MD5s are **identical to native `ffmpeg -f framemd5`** on the same input (30/30).
- The native reference is built from **unpatched** FFmpeg.
- In the same boot, the flipped-input positive control decodes all 30 frames and changes 10
  hashes, exactly as it does natively.

**Read "What this does not establish" before citing it.** In particular, the domain gets its
global-access capability (`gp`) from a QEMU convenience that cannot exist on silicon.

**CORRECTION (2026-09-23): this is a correctness result, not a memory-safety result.** "Under
capability enforcement" was true but was read as more than it says. The heap is musl-capstone's
`level0`:
- every heap pointer carries the bounds of the whole 1.5 MiB arena;
- `free` revokes nothing.

So in THIS run an overflow from one heap object into another, and every use after free, go
undetected. Measured, not inferred, in `../2026-09-23-qemu-safety/`. The same decode is also
bit-identical there with per-object heap bounds, and with revocation on free, on QEMU only; the
deployed silicon is documented not to stop a stale data access (that folder, verdict item 5).

## Identity

| | |
|---|---|
| source | branch `ffmpeg-app` at **`5f05b2148b40`**: FFmpeg 9.0.1 (`upstream.json`) plus patches 0001–0003 |
| images | `SHA256SUMS`, hashed **before** the boot. The copies the guest booted were re-verified identical afterwards |
| reference | `stock.framemd5` from stock `ffmpeg` built from the **pristine** tarball. Hash `663177980a01…`, the same value the buffer-pool lane committed from its own independent stock build (`buffer-pool/results/measurements/20260919-replay/measurements.json`) |
| budget | `code_len` 3,619,024 B, plus 8 KiB, plus the declared 256 KiB stack, gives a 4 MiB allocation. **The kernel module's own line agrees for every image:** `code size = 3619024, tot_size = 400000` (six lines, in `result-lines.txt`) |
| gates | See the table below |
| emulator | capstone-qemu (main clone build), `run-domain-smoke.py`, one boot, `$CAPSTONE_QEMU_LOCK` held |
| guest rootfs | A **private, fsck-repaired copy** of the shared `rootfs.ext2` (inode 623, `/var/lib/seedrng`). The shared image was left untouched, and its hash was verified unchanged |

The gates, each shown to fire before it was trusted:

| gate | result in this run | shown to fire on |
|---|---|---|
| C-50 scan | 0 hits in 354,669 instructions | the reproducer and three evading layouts |
| budget and layout | 9/9 images fit, none has `.capstone_gp_initdesc` | a gp-captable image, for the layout check |
| negative link control | fires | — |
| per-section verdict | every image reached its own milestone | a doctored log |
| `compare-md5.py` | MATCH, control fires | a mismatch, a truncated output, an empty reference, a 0-frame control, and a control that cannot fire |

The input's bytes vary per generation, because Matroska writes a random SegmentUID. So the
input and its flipped twin are hashed for this run only; the decoded frames are the stable
oracle.

## Result (`result-lines.txt`)

Each image returned exactly the milestone it was built for, checked **inside its own section**
of the serial log:

| image | host requests | `capstone_main` |
|---|---:|---|
| M1 main | 1 | 1 |
| M2 `avformat_open_input` + `find_stream_info` → `streams=1 video=0 320x180` | 21 | 2 |
| M3 first packet, `size=10574` (native 10574) | 23 | 3 |
| M4 first frame | 26 | 4 |
| M5 `frames=30 packets=30` | 122 | 5 |
| M5 on the flipped input (control) | 122 | 5 |

```
oracle: reference 30 frames, candidate 30 frames, 0 hash mismatches -> MATCH
control: 30 frames, 10 changed hashes -> FIRES
```

No `capstone-domain: UNSERVED syscalls` line was printed.

## What this does not establish

**1. Silicon: not run, and this ABI cannot run there as-is.**

The musl-capstone `my_first_domain/link.ld` ABI reaches globals through a `gp` with cursor 0.
Its own start code does this (`start-musl.S`, `.Lpcrel_domret_entry`:
`auipc; addi; cincoffset t0, gp, t0; stc`). capstone-qemu **fabricates** that `gp` when an
untagged one reaches `CINCOFFSET`, and its source says the result "CANNOT EXIST ON SILICON"
(`target/riscv/op_helper.c`).

This run relied on it: the fabrication counter, which logs every 1000th, reached **#75,000**.
Measured on the previous run-of-record images (same ABI and runtime), in two separate boots:

- **`CAPSTONE_GP_FABRICATE=0`:** QEMU stops before the shell prompt; the domain never runs.
- **`CAPSTONE_GP_FABRICATE=0 CAPSTONE_GP_STANDIN=1`**, i.e. a representable image-covering `gp`
  as the monitor would deliver on silicon: the domain takes a capability fault (cause 7) at
  `pc 0x102000064`. That is in musl-capstone's start code, **before `capstone_main`**.

This is the known reason silicon uses the gp-captable ABI (prior art:
`my_first_domain/start-fpga-gpseed.S`). The board therefore needs **M6**: the port rebuilt on
that ABI, with its one-translation-unit question (plan §4). **It is the same dependency every
musl-capstone domain on this ABI has, not an FFmpeg defect.**

**2. Patch 0002's `frame.c` changes.**

They are correct by reading, but almost certainly **never executed** here. The decoder
allocates frames through `avcodec_default_get_buffer2`, and only `hwcontext.o` and `encode.o`
call `av_frame_get_buffer`. So the MD5 match says nothing about them.

**3. Heap and stack headroom.**

The capability build's peak heap is unmeasured; it is bounded above by the 1.5 MiB arena. The
256 KiB stack is a guess. The declared total is about 298 KiB below the next power of two.

**4. Determinism in general.**

The earlier run of record's images rebuilt byte-identical from scratch, on this host, with this
LLVM build and a reused `libc-capstone.a`. That is reproducibility here, not a general claim.

**5. Memory safety.** See the correction at the top and `../2026-09-23-qemu-safety/`:
- **Heap:** arena-wide bounds, and no temporal safety.
- **Globals:** at -O1 on this ABI, the globals GlobalMerge packs into one block share that
  block's bounds. The gp-captable ABI disables GlobalMerge.
- **Stack:** an escaping 64-byte array was bounded exactly.

**6. Any other workload, and performance.**

This is one file at one frame size; 640x360 does not fit one region. TCG timing means nothing.

**Instrument notes.**
- `run-domain-smoke.py`'s substring markers are satisfied by the shell's echo of the guest
  command. They are kept as a smoke check only. The verdict is the per-section `REACHED` line
  plus `compare-md5.py` (see the header of `run-qemu.sh`).
- The flip image is built from the same objects except its entry object. It is not a
  byte-level twin of M5: the longer path string shifts the layout.

## How it got here (branch history)

1. **M0:** a cross-configured minimal FFmpeg links as a domain. It fits the 4 MiB budget only
   with a `.capstone_domreq` declaration. Patches 0001–0002 keep pointer provenance.
2. **Three defects between M1 and M5,** each localised by a run that returned a result:
   - **stdout lost after its first line:** musl goes fully buffered, and the runtime never
     flushes on return. Fixed in the port; the runtime caveat is recorded on `dev`.
     UPDATE 2026-09-24: the runtime now ends a returning program through `exit()`, which
     flushes (merge `556863938d46`). This folder's runs predate that.
   - **`EFAULT` on 9p reads:** host bounce buffer, `dev` `e852b3951476`.
   - **Compiler miscompile C-50:** `dev` `0e5b7b991629`, corrected in `ef636ba1d05f`. Worked
     around by patch 0003, and gated.
3. **The first run of record,** at `41355570eda7`, passed, and was reproduced from scratch
   with byte-identical images.
4. **Three adversarial audits** confirmed the result and exposed instrument holes: a
   reference sharing the patches, a 0-frame control counted as firing, an evadable C-50 scan,
   echo-satisfied markers, and a `pipefail` trap. They also found the `gp` dependence above.
   All the holes are fixed in `5f05b2148b40`, and **this** run of record was made from
   scratch afterwards.
