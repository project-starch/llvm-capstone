# FFmpeg adapted-FATE decoder qualification

The complete FFmpeg 9.0.1 Matroska/MPEG-4 decoder passes **18/18** checked
application processes on two adapted FATE inputs. Each of the three qualified
arms—Capstone pool original, Capstone pool Sublet, and CheriBSD PoisonCap
spatial—runs each input three times and matches the independent native
per-frame decode oracle. The 20-frame Xvid input has 262 pool lease issues
and 219 same-block reissues per process; the 150-frame resolution-change input
has 1,852 issues and 1,751 reissues. All 32 release-to-reissue gap bins are
identical across the three arms and all repetitions for each input. These
inputs therefore show no lease-reuse difference among the qualified arms.

The inputs come from the named FFmpeg FATE cases, but were losslessly remuxed
from raw M4V into Matroska for the configured measured decoder. The original
and remuxed streams have identical native per-frame MD5 output. These are
**adapted application inputs, not official FATE scores**. Their source,
remux and oracle hashes are in [fate-mpeg4-inputs.json](../../fate-mpeg4-inputs.json).

The protected PoisonCap mode triggers `panic: Poison probe missing page
0x41400480` in a fresh published CheriBSD guest. It does so when run before
the spatial controls as well. Eagerly touching the pool and omitting the
final explicit `munmap` did not resolve the panic. Later diagnostic binaries
that locked the pool with `mlock`, changed its page protection with
`mprotect`, or combined both also hit the same kernel panic. The interrupted and
diagnostic processes are excluded from the 18 qualified attempts. This is a
**three-arm functional and lease-gap qualification, not a four-arm FATE memory
figure**. The earlier [30-frame four-arm decoder campaign](../ffmpeg-reuse-gaps-20260927/README.md)
remains separate and valid for its own input; its result cannot fill the
missing FATE cell.

[summary.json](summary.json) records each accepted application process,
output hash, pool observer totals and all 32 gap bins. [archive.json](archive.json)
identifies the raw archive under `$CAPSTONE_TMP_ROOT` with logs, measured
binaries, inputs, build manifests and the excluded panic record. The archive
omits guest SSH credentials and VM state. Its SHA-256 is
`ac6e794e5d63f082d0b014588c78ebd780792a5d7f63497bee1fcb404f834080`.
The observer counts integer pool-block identities, not resident memory or
total platform footprint; this qualification establishes no memory-cost or
working-set ranking.
