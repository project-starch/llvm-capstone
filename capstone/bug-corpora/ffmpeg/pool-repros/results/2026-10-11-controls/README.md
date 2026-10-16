# FFmpeg nested-lifetime controls, 2026-10-11: parent, sibling, successor, stale

The FFmpeg 9.0.1 row of the pre-registration
[`docs/plans/2026-10-11-nested-lifetime-controls.md`](../../../../../docs/plans/2026-10-11-nested-lifetime-controls.md)
(committed in 53325bba). The nested allocator is the `AVBufferPool` entry: handed out by
`av_buffer_pool_get`, released by `av_buffer_unref`, which returns the entry to its pool
(`pool_release_buffer`). The protected arm is buffer-pool patch 0003 on 0001 (`FFPOOL_SUBLET=ON`); the
release arm is the same source without 0003. Sources:
[`controls/nested-lifetime/`](../../controls/nested-lifetime/README.md). Every fact below is in
`results.json`, with each run's own result lines.

**All 36 predictions held**: 6 per boot, 3 boots per arm. The positive control fired as
registered: the stale read completed on release, reading C's data, and faulted on protected with
cause 25 at the labelled probe.

## Outcomes

Boots 1-3 per arm, identical in every outcome. In the address columns, each run is its own
process, so its addresses differ. In every run, C's start address equals A's.

| control | arm | outcome (3 of 3) | access sites (call site; probe function) | addresses (boot 1 / 2 / 3) |
|---|---|---|---|---|
| 91 parent | release | completed: parent read 0 mismatches vs A's 0x5A; write 0xC3 reads back; next hand-out C is the same entry and reads 0xC3 | `91_…/case.c:62` read, `:67` write, `:71` read back, `:99` C read; `read_probe` `case.c:30`, `write_probe` `case.c:37` | A = entry.data = C: `0x3fbd5aa440` / `0x3fa7a38440` / `0x3fa6ac6440` |
| 91 parent | protected | completed: as release, plus through `sublet_life`: read 0xC3, write 0x3C, visible through `entry->data` (`:77`, `:82`, `:85`); C reads 0x3C | as release, plus `:77`, `:82`, `:85` | A = entry.data = sublet_life = sublet_issued = C: `0x3f9a25a440` / `0x3f97d03440` / `0x3f9ccc2440` |
| 92 sibling | release | completed: B 0 mismatches vs 0xB2 after A's release; write 0xB7 reads back | `92_…/case.c:41` read, `:46` write, `:50` read back; `read_probe` `case.c:14`, `write_probe` `case.c:21` | A / B: `0x3f94879440`/`…94c0`, `0x3fb9a8f440`/`…f4c0`, `0x3f85fd5440`/`…54c0` |
| 92 sibling | protected | completed, B intact (same checks) | same | A / B: `0x3f81135440`/`…54c0`, `0x3fbb8f5440`/`…54c0`, `0x3f8c0cf440`/`…f4c0` |
| 93 successor `run` | release | completed: C == A after 1 get; write 0x77 reads back | `93_…/case.c:70` write, `:74` read; `write_probe` `case.c:27`, `read_probe` `case.c:20` | A = C: `0x3f9c5fe440` / `0x3f833f6440` / `0x3faf778440` |
| 93 successor `run` | protected | completed: C == A after 1 get; write 0x77 reads back | same | A = C: `0x3f8a789440` / `0x3f8db24440` / `0x3fa5fa0440` |
| 93 `stale` | release | **completed**: the read through A's old pointer returned C's 0x77 in all 64 bytes | `93_…/case.c:80`; `stale_read_probe` `case.c:33` | A = C: `0x3f9883b440` / `0x3f8ad46440` / `0x3f812ff440` |
| 93 `stale` | protected | **fault, cause 25**, at `stale_read_probe`+52, `93_…/case.c:36` (link pc `0x20ee0`); fault address = A's address | `93_…/case.c:80` | A = C = fault address: `0x3f8a024440` / `0x3fb0216440` / `0x3fb17f7440`; runtime pc `0x3f89e10ee0` / `0x3fb0010ee0` / `0x3fb1610ee0` |

`case.c:36` is the probe's load, `bad += p[i] != expect;`. The fault came while `stale-read` was the
access in flight: its CTL-ACCESS line was printed and its DONE line never was. The launcher's own
image hash in each fault line equals the staged `ctl-93` (`e8b4456b…`).

Platform controls, every boot on both arms: `bounds-malloc` faulted with cause 28 and `uaf-malloc`
with cause 25. The instrument was live on the release arm too, where every control completed.

## Predictions

| prediction (pre-registration) | release | protected |
|---|---|---|
| control 1 completes; the read returns what the allocator left there | held 3/3 | held 3/3 |
| control 2 completes, B intact | held 3/3 | held 3/3 |
| control 3: C's start address equals A's; access through C completes | held 3/3 | held 3/3 |
| stale access through A: protected rejects with cause 25 at the labelled probe; release completes and reads C's data | held 3/3 | held 3/3 |

## Identity

| what | sha256 / value |
|---|---|
| llvm-capstone | HEAD `53325bba0d5c` (`ports/nested-lifetime-controls`), no tracked file modified, plus the 10 uncommitted files of `controls/nested-lifetime/` listed with their hashes in `results.json` `revision`. The snapshot is `git archive` of `capstone/` and `compiler-rt/lib/builtins/` plus those files; other lanes' untracked files were not in it. The two READMEs were written after the run. |
| QEMU / launcher / module | `af37cc32…` / `3975314a…` / `a1c6cb6b…` (the registered kit, `kit-sublet`) |
| firmware / kernel / rootfs | `44cd6cbc…` / `e833dfb1…` / `62049800…` |
| profile | virtual, exact bounds (`run-staged.py` default), `-smp 1` |
| compiler | `4c6135fe…` (the kit's toolchain, as the 2026-10-11 corpus runs) |
| SDK, built from the snapshot | `capstone-cc` `8e4bd500…`, `libapplication-runtime.a` `e5693f5c…`, builtins `a72ed0bf…`, libc (musl, virtual profile) `a0f5c573…` |
| `libffmpeg-pool.a` | release `8ec39303…`, protected `a3709ae5…` |
| images, release | `ctl-91` `06dbb4ee…`, `ctl-92` `62efc1bd…`, `ctl-93` `e3fdeacb…` |
| images, protected | `ctl-91` `46f8bd17…`, `ctl-92` `39227be3…`, `ctl-93` `e8b4456b…` |
| configuration | library `CMAKE_BUILD_TYPE=Debug`; controls `-std=gnu11 -O1 -g -DFF2_SYSTEM_MEMORY -DCTL_PROTECTED=0|1`; pool `av_buffer_pool_init(64, NULL)` |
| runner / judge | `run-controls.py` `a4ef3ca3…` / `verdicts.py` `c5e58723…` |

The `run` and `stale` modes of control 93 use one image per arm and differ only in argv.

## Wall times

Measured on the shared host; three other agents were active. The musl build took 4 s, the SDK
0.5 s (its ninja log) and each arm's library plus controls 4 s. Six VMs ran in parallel, one per boot
and each with its own lock: 17 s wall per VM, 10.4-10.9 s inside the boot, 18 s for the whole run.
`results.json` timestamps are the run host's clock, which was 1582 s behind UTC.

## Deviations and limits

- **Kit path.** The brief named `~/vkit`; the run host has no such directory. The run used the kit the
  2026-10-11 corpus runs and the reuse-distance run used. Its six platform hashes are the registered
  ones (above, and the corpus's `results/2026-10-11-virtual/*/inputs.json`).
- **How the protected configuration is built.** The brief described it as "FFAPP_SUBLET=1". These
  controls link libavutil the way the corpus does: the buffer-pool port's library with
  `FFPOOL_SUBLET=ON`, the same patch files (0001 + 0003) that `FFAPP_SUBLET=1` applies to the FFmpeg
  application. They do not link the full FFmpeg application image.
- **Compiler not rebuilt from the snapshot.** `llvm/` was not shipped. The compiler is the kit's
  binary (`4c6135fe`), the one the corpus's 2026-10-11 runs used. The SDK, musl and the runtime were
  built from the snapshot.
- **Extras beyond the registration.** Control 91 also accesses through `sublet_life` on the protected
  arm, and ends on both arms with a hand-out C that must be the same entry and see the parent's write.
  N = 3 boots per arm; the registration names no N.
- **What the pointer metadata cannot show.** `CTL-PTR` prints cursor, base and end (LCC). For
  `A.data`, `entry.data`, `sublet_life` and `sublet_issued` these are identical (64 bytes, exact
  bounds). The metadata therefore does not tell the parent capability from the child. What separates
  them is the behaviour: the child faults after the release (93 `stale`), and the parent does not (91).
- **Fault-line field truncated.** `fault.line` in `results.json` is cut at 200 characters, which
  truncates the launcher's image hash. The full line is in each run's `lines`. That is where the
  launcher's hash was checked against the image.
- **The stale probe reads 64 bytes.** On protected it faults at the first load, at A's base address.
  On release all 64 bytes equal C's 0x77.

## On silicon

The same controls ran on silicon on 2026-10-16 (board set 6, `caplifive_sublet_d93a8d66b`). Their readings sit beside
this record's in `../2026-10-16-silicon-controls/`: every silicon run reads as the same control does here.
