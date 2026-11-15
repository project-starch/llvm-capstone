#!/usr/bin/env bash
# Copy raw Annex B FATE samples (h264-conformance, hevc-conformance) losslessly into Matroska and
# check each copy with the host's decoder against the FATE reference
# (docs/plans/2026-11-15-ffmpeg-established-workload.md, "Annex B inputs remuxed into Matroska"):
#
#   remux-annexb.sh FATE_SUITE REFS OUT SUBSET.tsv
#
# FATE_SUITE is an rsync copy of rsync://fate-suite.ffmpeg.org/fate-suite/, REFS the tests/ref/fate
# directory of the FFmpeg 9.0.1 source tree, OUT a new directory. For every SUBSET line whose sample
# starts with mkv/ (TEST<TAB>mkv/<fate path>.mkv<TAB>hash), it writes OUT/<fate path>.mkv with
#
#   ffmpeg -nostdin -v error -r 25 -f h264|hevc -i SAMPLE -map 0:v:0 -c:v copy \
#          -bsf:v setts=pts=DTS -f matroska -y OUT
#
# (the October adapter of experiments/study/fate-mpeg4-inputs.json, plus setts=pts=DTS because
# Matroska needs a timestamp on every packet), then decodes the copy with the same host ffmpeg
# (-fps_mode passthrough, framecrc) and prints MATCH or DIFF against the reference's size and
# checksum columns. The host's ffmpeg must be on PATH; its version goes into the first line.
set -euo pipefail
[[ $# == 4 ]] || { echo "usage: remux-annexb.sh FATE_SUITE REFS OUT SUBSET.tsv" >&2; exit 2; }
SUITE=$1 REFS=$2 OUT=$3 SUBSET=$4
mkdir -p "$OUT"
echo "host $(ffmpeg -version | head -1)"
columns() { grep -v '^#' | awk -F', *' '{print $5, $6}'; }
while IFS=$'\t' read -r test sample _; do
  case $sample in mkv/*) ;; *) continue ;; esac
  src=${sample#mkv/}; src=${src%.mkv}
  case $src in h264-conformance/*) fmt=h264 ;; hevc-conformance/*) fmt=hevc ;; *) echo "$test: not Annex B" >&2; exit 2 ;; esac
  dst=$OUT/$src.mkv
  mkdir -p "$(dirname "$dst")"
  ffmpeg -nostdin -v error -r 25 -f $fmt -i "$SUITE/$src" -map 0:v:0 -c:v copy -bsf:v setts=pts=DTS \
    -f matroska -y "$dst"
  got=$(ffmpeg -nostdin -v error -i "$dst" -fps_mode passthrough -f framecrc - | columns | sha256sum)
  want=$(columns < "$REFS/${test#fate-}" | sha256sum)
  [[ $got == "$want" ]] && echo "$test MATCH $(sha256sum < "$dst" | cut -c1-64)" || echo "$test DIFF"
done < "$SUBSET"
