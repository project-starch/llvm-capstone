# Repeat runs of the FFmpeg and Wireshark reuse traces: pre-registration, 2026-10-12

Committed and pushed before any run. A reading that differs from what is written here is a
finding; this file is not edited afterwards.

## The question

`2026-10-11-nested-reuse-distance.md` (16d2171a3edc) measured reuse distance at each application's
nested allocator. SQLite, PostgreSQL, CPython, Perl and memcached have three runs per build and
mruby two. FFmpeg and Wireshark have one run per build
(`ports/ffmpeg/app/results/2026-10-11-reuse-distance`,
`ports/wireshark/app/results/2026-10-11-reuse-distance`), so the figure cannot show their
run-to-run range. This run adds two more runs per build for both programs.

## Method

Unchanged from the 2026-10-11 records: the same instrumented images (each image's sha256 must
equal the one in that record's `reuse.json`, otherwise the run is reported and not merged), the
same workloads (FFmpeg: `capstone-vexec ffmpeg.dom input.mkv 1`, 30 frames; Wireshark: `tshark -r
C -V -n` once per capture of the app port's set), the same runners and parsers, the same platform
(QEMU af37cc32, launcher 3975314a, module a1c6cb6b). Each run is its own boot. The observer's
`error` must be zero and each run's output must still equal the record's oracle (FFmpeg: the 30
frame hashes; Wireshark: stdout per capture), otherwise the run is reported and not merged. The new
runs are added to each `reuse.json` as repetitions in the format the other programs use, keeping
the first run as the plotted one.

## Predictions

Both workloads are single-threaded with fixed input, and the first runs gave identical histograms
on both builds at every boundary (FFmpeg `buffer-pool` and `refstruct-pool`, Wireshark `wmem-block`
and `wmem-block-fast`). For each program, boundary and build:

- each new run's histogram equals the first run's, bucket for bucket;
- each new run's issue, release and reuse counts equal the first run's;
- the stock and protected histograms remain identical.

Runs not finished by 10:00 local on 2026-10-12 are reported as not measured.
