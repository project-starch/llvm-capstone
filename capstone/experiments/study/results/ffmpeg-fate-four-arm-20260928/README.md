# FFmpeg adapted-FATE four-arm pool study

**Policy audit, 2026-09-28.** Historical policy: these measurements use the custom sweep-before-reissue adapter and disable outer libc revocation. Their coincident reuse curves do not describe the published SQLite quarantine policy transferred to FFmpeg. See the [reference-policy audit](../../poisoncap-policy.md).

The complete FFmpeg 9.0.1 Matroska/MPEG-4 decoder passes **24/24** checked
application processes on two adapted FATE inputs: Capstone pool original,
Capstone pool Sublet, CheriBSD PoisonCap spatial, and CheriBSD PoisonCap
temporal, three processes per input and arm. Every process matches the
independent native per-frame decode oracle. These are adapted inputs from
named FATE cases, **not official FATE scores**; the remux and source hashes
are pinned in [fate-mpeg4-inputs.json](../../fate-mpeg4-inputs.json).

| Input | Frames | Issues | Same-block reissues | Gap ≤15 / all issues | Temporal PoisonCap snapshot peak |
|---|---:|---:|---:|---:|---:|
| Xvid IDCT | 20 | 262 | 219 | 83.59% | 224.4 KiB |
| Resolution change | 150 | 1,852 | 1,751 | 93.68% | 167.1 KiB |

All four arms have identical 32-bin release-to-reissue histograms for each
input and repetition. The [figure](fate-pool-memory.pdf) shows that equality
alongside peak snapshot backing in the selective PoisonCap adapter. The
snapshot counter is a **selected component**, not total memory. The temporal
adapter also reports 35 and 191 explicit sweeps, respectively, and cumulative
poison/clear/copy operation spans of 31.61 and 98.55 MiB. Those spans are
neither measured DRAM traffic nor a lower bound on all PoisonCap policies.
The FFmpeg pool's immediate lease reuse offers no Sublet reuse advantage on
these inputs; the older [30-frame four-arm result](../ffmpeg-reuse-gaps-20260927/README.md)
agrees for its own workload.

The published PoisonCap kernel panicked on the protected FATE process because
RISC-V `pmap_extract_and_hold()` only handled L3 mappings while poison probes
could encounter valid L2 superpages. The narrow [kernel patch](../../patches/cheribsd-riscv-superpage-hold.patch)
adds L2 extraction and holds the constituent 4 KiB physical page. We built it
against the pinned CheriBSD source at `214a1a2d939ec065a7ac5452413058e84c3faf61`;
the previous published kernel SHA-256 was
`80a3990df0d526447abfa48004bf6e92e410fe949808b0e201e2533d3177843e`,
and the measured patched kernel SHA-256 is
`ea6e8e06e818616fb7ccf4f2a07b7d6a6ab008b563700937c64de0f703c7d36a`.
Both CheriBSD arms were rerun on **that same patched kernel**, with default
outer revocation disabled and the nested mode selected only by the application
argument. The kernel patch is a correctness fix; it does not change the
PoisonCap nested allocator policy. The original panic remains in the raw
archive as excluded evidence. The exact incremental kernel build command,
build log, patch, and resulting kernel binary are also archived.

Run [collect-ffmpeg-fate.py](../../collect-ffmpeg-fate.py) over the raw
Capstone and new CheriBSD runner directories to regenerate [summary.json](summary.json);
the collector checks every process status, full output hash, frame count,
policy marker, counter reconciliation, and matched release-gap bins. Run
[plot-ffmpeg-fate.py](../../plot-ffmpeg-fate.py) on that summary to regenerate
the PDF and PNG. [archive.json](archive.json) identifies the 10.3 MB raw
archive under `$CAPSTONE_TMP_ROOT`; it omits guest SSH credentials and VM
state. The archive includes the measured application binaries and adapted
inputs. This is a real application workload, not allocator trace replay.

The two operating systems have different outer allocators and address-space
accounting. The plotted snapshot component and lease observer therefore make
no claim about relative total RSS, platform metadata, or full adaptation cost.
