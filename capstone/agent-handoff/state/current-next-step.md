# Current recommended next step

## 2026-07-30 (late) — SQLite on silicon: wedged in the domain's FIRST entry (R-12 live)

### RETRACTED: "SBI arguments 1-3 arrive as zero"

The previous version of this file said argument positions 1, 2 and 3 arrive at the monitor
as zero, and named the trap glue as the next place to look. **That was an artifact of a
stale boot, not a defect.** On a verified boot the same call reports

```
LC: rgid=12 prm=1 rev=2 dom=0      <- what the host passed
RGID:0000000C  APRM:00000001  AREV:00000002   <- what the monitor received: CORRECT
```

Do not resurrect the argument-marshalling investigation. `DOM_CREATE`'s packing of
`globals_off` into the high half of `arg3` is a deliberate design choice to avoid a struct
change in two submodule trees, not a workaround for a register limit.

### Why the diagnostics were wrong, and what now prevents it

Two independent staleness bugs, both in our own harness, both fixed:

1. **`cold_boot()` loaded the wrong stored image.** It is defined in
   `run_ladder_perf_fpga.py`, so its `monitor load_image images/{IMG_NAME}` resolves
   `IMG_NAME` in THAT module — a hardcoded ladder image. `run_sqlite_baked_fpga.py` uploads
   under a content-hashed name, so the board booted a **July 19** initramfs. It had worked
   only while callers exported `FPGA_FW_NAME`, which both modules read from the environment;
   the content-hash default removed that accidental coupling. `cold_boot` now takes
   `img_name`.
2. **`A=opensbi-rebuild` links the firmware BEFORE buildroot regenerates the images**
   (`fw_payload.bin` 20:30:11 → `rootfs.cpio.gz` 20:30:24 → `Image` 20:30:36), so it embeds
   the PREVIOUS generation. Re-running the same target does not converge; relink once more
   after a generation whose images already contain what you staged.

Gates that now fail the run instead of misleading it:

* **firmware freshness** — decompresses the firmware's own initramfs and requires the local
  `sqlite_host.user` / `sqlite_silicon.dom` bytes to be present. Not mtimes (13 s apart in
  one `make`), not `Image[:4096]` (kernel header, identical across generations), and not the
  `.gz` prefix (**gzip embeds an mtime**, so a correct firmware compares unequal).
* **stale boot** — `sha256sum` on the board must match the local build, or the run aborts
  before executing anything.

### Exact position now

Deterministic, ~4 minutes, on a boot with both binaries hash-verified on the board:

```
Globals offset = 0x140000        host built correctly
SQ: A/dom-ok id=0                create_dom works
SQ: B/mkregion1 C/mkregion2      both create_regions work
SQ: D/mapped r1=12 r2=14         both map_regions work
SQ: E/share1                     first shared_region_annotated
ECSA/EXTC/FNCC/ARG1              dispatch reached, arguments correct
SHA0 .. SHA5                     the monitor's whole share path completes
BASE:81D27000 ALEN:00001000      the region capability is minted
<silence, then idle-abort>       SHA6 never fires
```

`SHA5` then silence means the wedge is at `sbi_capstone.c:984`,
`d = __domcallsaves(d, CAPSTONE_DPI_REGION_SHARE, r)` — the monitor calls INTO the domain
and the domain never returns. **The monitor is exonerated for this failure.**

### THE NEXT STEP — R-12, and the discriminator is already running

This share entry is the domain's **first execution**, so the interp glue's builder runs
here. From the domain's own descriptor (`.capstone_gp_initdesc`, read at its section
offset): **`count = 1059`**, plus the table split, so ~1060 `split`s. The rev-node pool is
a fixed **1024**-entry bump allocator whose `head` is **10 bits**, so allocation #1025 wraps
to id 0 and reuses live ids silently (`overflow_flag` reaches only a debug LED). This is the
first time execution has ever got far enough for R-12 to be reachable, and it is
hardware-only — the same domain passes under QEMU.

`INTERP_BUILD_LIMIT=<N>` clamps carve iterations while leaving the cap-table geometry
byte-identical; it is plumbed through `build-sqlite-silicon.sh` via `INTERP_EXTRA_CFLAGS`.
A build at **900** is under the pool size.

* `SHA6` appears at 900 and not unclamped ⇒ **R-12 confirmed** as the current blocker.
* wedges at 900 too ⇒ R-12 is not what is biting; the builder itself fails on this domain,
  and the next split is `bigblob` (which passes with SQLite's create-time geometry) versus
  SQLite's descriptor — i.e. bisect the limit, since geometry is held constant.

R-12's fix is genuinely large (widen the pool = RTL/board owner; one capability per section
= an ABI change that costs the per-object property the paper claims; reclaim on drop = RTL
implements drop as invalidate-only). Confirming it is still worth the run: it converts a
predicted risk into a measured, named blocker with a bounded fix, which is a publishable
sentence either way.

### Tools that now exist — use them, do not rebuild them

* **Monitor errors print to the UART** (I-4). 28+ tags. `SHA5` is the last marker before
  M-mode is left, so "SHA5 then silence" localises to the domain in one run.
* **Run-scoped UART** at `/tmp/capstone/sqlite-run-scoped.txt`.
* **Idle-abort** (`SQLITE_RUN_IDLE=75`): a wedge costs ~75 s instead of 15 min.
* **`SQLITE_HOST` / `SQLITE_DOM`** accept shell snippets, so an arbitrary probe runs without
  a firmware rebuild (this is how the loader was cleared: `ld.so <prog>` by hand).
* **`bigblob`** rung: SQLite's create-time geometry, PASSES. Same-firmware control.

### Hard-won rules (each cost real time)

1. **NEVER grep the runner's stdout log for board state.** It carries the whole accumulated
   UART buffer, across boots and sessions. Only `/tmp/capstone/sqlite-run-scoped.txt` is
   provenance-safe. This produced two false conclusions in one day: a zeroed-argument
   "defect", and a report that the host binary was present and size-matched when that run's
   own scoped read said `No such file or directory`. Two contradicting observations mean one
   is out of provenance, not that the board is flaky.
2. **Tags are numeric constants** — `capstone-c` materialises them with `lui`+`addi`, so
   grepping the firmware for `"SPLA"` proves nothing. Check decimal immediates in the
   regenerated `.c.S`.
3. **Delete `components/opensbi/lib/sbi/sbi_capstone_dom.c.S`** before any monitor rebuild,
   or it relinks stale. That is the path declared as `CAPSTONE_S_OUTPUT` in
   `caplifive-system/sw/buildroot/Makefile:5` — NOT the rsync'd copy under `build/build/`,
   which is what got deleted for a whole session with no effect.
4. **A report line must fit 16 bytes** (the 16550 TX FIFO). `TAG:` + 8 hex + CRLF = 15.
5. **Absolute paths in board commands**, and in shell commands after any `cd`.
6. **Compare artifacts by CONTENT.** Stale and current SQLite domains are both 1,623,008
   bytes, so a size check says "match".
7. **Rebuilding the kernel invalidates `modcapstone`.** Order: stage →
   `linux-rebuild-with-initramfs` → `modcapstone-rebuild` → `opensbi-rebuild`, then
   `opensbi-rebuild` ONCE MORE so the firmware embeds the final images.
8. **`build-sqlite-host.sh` must use the caplifive-BUILDROOT libcapstone** — the only copy
   with the globals-offset packing.

### Retracted — do not resurrect without new evidence

"More than one global fails"; "a 16-byte global fails"; unrepresentable capability bases in
the glue; coarse capability tag granularity; two-regions-required; **"SBI arguments 1-3
arrive as zero"** (stale boot); and "the monitor is at fault" in general — `bigblob` passes
on identical firmware, and `SHA5` now proves the monitor completes its share path.

### Status against the descope ladder

Level 3 — "SQLite green in the silicon config under QEMU plus a documented board attempt
naming the specific blocker" — is met, and the blocker is now named to a specific
instruction (`sbi_capstone.c:984`) with the mechanism identified and a discriminator
running. Level 1 (existence proof on silicon) is not.
