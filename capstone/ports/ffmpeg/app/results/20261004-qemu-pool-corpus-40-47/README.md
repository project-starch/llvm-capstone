# The FFmpeg pool corpus on both app-port arms, all four cases — including case 3's first `sublet-port` reading (2026-10-04)

**Question.** Case 3's `sublet-port` rows (fixtures **46/47**) had been predictions since 2026-10-03, and
fixtures 40-45 had one committed measurement made by a driver that no longer exists. Do all four cases
discriminate on FFmpeg's **own** pools, ported to Sublet?

**Verdict: 16 of 16 cells AS PREDICTED. Both external verdicts exit 0.**

| case | fixture pair | `poolstock` (control) | `poolsublet` |
|---|---|---|---|
| 0 `af_join_dedup_bound` | 40 / 41 | DEFECT-REPRODUCED / FIXED | **FAULT cause 24, `case.c:85`** / FIXED |
| 1 `h264_refs_partial_clear` | 42 / 43 | DEFECT-REPRODUCED / FIXED | **FAULT cause 24, `case.c:48`** / FIXED |
| 2 `vidstab_parked_plane_pointer` | 44 / 45 | DEFECT-REPRODUCED / FIXED | **FAULT cause 24, `case.c:33`** / FIXED |
| **3 `vvc_nonref_output_releases_tabs`** | **46 / 47** | DEFECT-REPRODUCED / FIXED | **FAULT cause 24, `case.c:104`** / FIXED |

Every fault landed on a line the registered row names (`85,86,89` / `48,49,52` / `33` /
`104,105,108`), and every one reports a **non-zero `value_hi`** — `13b89602540`, `a3890033c0`,
`14189502500`, `1198a902a20`. That matters: a zero there would be an integer used as an address, not a
capability that lost its authority.

**Four discriminating pairs.** On `poolstock` — upstream's pools, one macro apart — each defect
completes and prints `VERDICT DEFECT-REPRODUCED`; on `poolsublet` the same `case.c` faults at the stale
access. The fixed arm of every case completes on both.

**Case 3 was the last unmeasured arm of the 22 upstream cases** across memcached, tshark and FFmpeg.

## Why 40-45 are new rows and not a confirmation

`dedab28c4a2c` made `src/capstone-domain/ffapp_corpus.c` create `g_refpool`, without which the corpus's
case 3 does not link — the app port has a *second* driver for these cases, alongside the corpus's own
`shared/driver.c`. The committed file that
`../../../../bug-corpora/ffmpeg/pool-repros/results/20260929-qemu-sublet-port/` ran against has **zero**
occurrences of `g_refpool`, so its twelve image hashes cannot be produced from today's source and, on
`poolsublet`, the extra pool takes a lease. **These rows supersede that bundle; they do not confirm it.**
That bundle now carries the caveat in-tree.

## The toolchain, and the caveat it carries

Built with the **qualified** toolchain — `clang 22.0.0git`, project-starch/llvm-capstone
**`7d01722aab88`** — which is the only one on this host that passes the application SDK's ABI gate
(`check_toolchain` accepts it end to end). **Every other measured FFmpeg arm used `b7b31421e9fa`.** So a
cell here is not built by the same compiler as a cell in the heap-arm bundles, and a comparison across
those bundles has to say so. This is also the concrete refutation of the claim retracted in
`5208789e4b9e`: the gate is satisfiable here, and these sixteen images went through it.

## How it ran

`capstone-vm` is this port's only runner and needs ssh; the rootfs has no dropbear and no riscv64
dropbear binary exists here to supply via `--ssh-server`. This bundle used
**`ports/common/application/run-fixtures-9p.py`**: it stages the images and the pinned
`capstone-exec`/`capstone-job`/`capstone.ko` over 9p, issues **the same guest command `capstone-vm`
issues** (`capstone-job <result> -- capstone-exec -- <image>`), merges the launcher's fault record into
that result exactly as `capstone-vm`'s host side does, and then hands the collected
`fx<n>.{json,stdout,qemu}` to **the corpus's own `runners/sublet-port-verdict.py`** — the committed
judge, not a reimplementation.

**The missing merge was a real instrument gap, and it first read as a refutation.** Before the fix, all
four faulting cells reported *"the fault record is for another image (None)"*: `sublet-port-verdict.py:99`
attributes a fault by comparing `result["image_sha256"]` with the image's digest, and that field is added
by the **host** (`ports/common/application/run.py:60-63`), not by `capstone-job`. The faults were correct
all along — each record's `sha256=` matches its image exactly — but a missing host-side field made four
genuine catches look like four misses. Both arms were then re-run end to end with the corrected runner,
so every row here comes from one tool version.

**Platform:** the pinned process-ABI monitor. Application images need the monitor's `PROCESS_*` ecalls;
the installed buildroot `fw_jump.elf` has none (0 occurrences of `context_step` against 41 in the pinned
one). `SHA256SUMS` records the monitor, the emulator, all sixteen images, the launcher and the module.
The runner also exports `CAPSTONE_GP_NONLIN=1` and `CAPSTONE_REV_NODES=65536`, which `capstone-vm` forces
and the earlier ad-hoc scripts did not.

## What this does and does not say

- **It does** give all four cases a measured, discriminating pair on FFmpeg's own ported pools, with each
  fault attributed to a `case.c` line and a non-zero `value_hi`.
- **It does not** make any case live at the pin. All four are fix-reversals; `5c66a3ab51` and the others
  are ancestors of `n9.0.1`.
- **It does not** measure PoisonCap, CheriBSD or `native-detect` for these cases — those platforms are
  absent from this host, and `native-detect` would measure the fixture rather than FFmpeg.
- **It does not** say anything about silicon. Cause 24 is capstone-qemu reloading a revoked capability
  untagged; the deployed silicon lets a stale data access retire (ISSUES Q-11).
- **N = 1 per cell**, one boot per arm.

Files: `result-lines.txt`, `SHA256SUMS`.
