# FFmpeg 9.0.1 whole-decoder pool memory pilot (2026-09-27)

The configured Matroska/MPEG-4 decoder runs the same 30-frame input as a whole
application in four nested-pool arms: Capstone spatial pool, Capstone Sublet
pool, PoisonCap spatial pool and PoisonCap temporal pool. The run repeats the
stream independently 1, 4 or 16 times. Every stdout transcript, including all
frame MD5 lines, matches the pinned decoder oracle byte for byte: **18/18**
existing Capstone attempts (three per cell) and **6/6** new PoisonCap attempts
(one per cell). The input SHA-256 and binary/source hashes are in [data.json](data.json).

| Streams | Frames | Capstone outer-heap peak, either pool mode | Pool payload used, either platform/mode | PoisonCap snapshot, temporal only | PoisonCap jemalloc allocated after release, spatial → temporal |
|---:|---:|---:|---:|---:|---:|
| 1 | 30 | 416,720 B | 315,072 B | 315,072 B | 678,312 → 973,224 B |
| 4 | 120 | 416,720 B | 315,072 B | 315,072 B | 776,616 → 1,071,528 B |
| 16 | 480 | 416,720 B | 315,072 B | 315,072 B | 614,656 → 932,992 B |

The within-PoisonCap jemalloc increment is 294,912–318,336 B, closely
tracking the adapter's 315,072 B retained snapshot. The Capstone outer-heap
peak and pool payload used are identical between its two modes in all three
workloads. Both platforms give the pool a fixed 4 MiB payload reservation.
Capstone's whole-app outer heap has a separate 16 MiB reservation; platform
node metadata and PoisonCap kernel shadow are excluded. Capstone's outer-heap
peak and PoisonCap's process-wide jemalloc `allocated` are **different ledgers**,
so their absolute heights must not be subtracted across platforms.

The [retention plot](ffmpeg-pool-retention.pdf) shows those two within-platform
pairs in separate panels. The [snapshot/rewrite plot](ffmpeg-pool-snapshot-work.pdf)
shows retained snapshot backing and the temporal adapter's explicit copied,
poisoned and cleared bytes. At 16 streams it copied 110,184,768 B, poisoned
55,249,920 B and cleared 54,934,848 B; these are bytes touched, not bandwidth
or execution time. The Temporal PoisonCap adapter completed 45, 183 and 735
sweeps for 1, 4 and 16 streams respectively; the Capstone pool reported 406,
1,624 and 6,496 Sublet revokes. Those are different operations and are not
plotted as equivalent units. Both PoisonCap modes use the **same binary** with
mode 0 or 2 selected explicitly; the matching published-platform libc runs
with automatic outer revocation off so the inner pool policy is isolated.

The new whole-application entry is
[ffmpeg-poisoncap-decode.c](../../ffmpeg-poisoncap-decode.c); the existing
`comparison-build.py --app ffmpeg --platform poisoncap --nested-pool poisoncap`
builds it from `prepare-source.sh --pool`, linking the already ported PoisonCap
pool backend. Sources and build products stay under `/tmp/capstone`, and no
benchmark suite is vendored. `data.json` retains per-phase measurements and
raw-log digests; raw VM transcripts remain outside the repository. Redraw the
PDF and PNG figures with:

```sh
python3 capstone/experiments/study/plot-ffmpeg-pool-memory.py \
  --data capstone/experiments/study/results/ffmpeg-pool-memory-20260927/data.json \
  --out /tmp/ffmpeg-pool-memory-figures
```

This is an exploratory, single PoisonCap repetition of a configured decoder,
not the FFmpeg FATE suite, an encoding benchmark or a total-memory result.
Multiple PoisonCap repetitions and broader recognized FFmpeg workloads are
needed before a paper claim. The prior Capstone attempts used the shared
persistent VM; the new PoisonCap attempts shared one CheriBSD guest boot.
