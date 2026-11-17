#!/usr/bin/env bash
# analyze-p1.sh OUT [KIT...]: the P1 suite's T3/T1 reports from whichever kit transcripts exist, as 2026-10-28's recipe
# (results/2026-10-28-t3heap-bitstream3/README.md) with the arms T1,T3 only. Rerun as each kit lands.
#   transcripts: $P1_OUT/<kit>/board.txt (default P1_OUT = <work>/spec/board/out-nat-p1suite-b3), board label $P1_BOARD (b3)
#   per kit: p1-check.py (the pre-registered gate, against the bitstream-3 references) -> OUT/check-<kit>.txt, and
#            parse-board.py -> OUT/runs-<kit>.tsv (+ .ncnt.tsv: per-process native_counters deltas beside BENCH sums)
#   then over the kits present:
#     main.tsv    the 15 allocating programs (p1k1..3), beebs *-libc labelled beebs-libc    -> OUT/T1-T3
#     all103.tsv  every kit present, the same labels                                       -> OUT/all103/T1-T3
#     groups.tsv  all103 with beebs-libc folded back into beebs (BEEBS one group of 93)    -> OUT/groups/T1-T3
# A kit whose check or parse fails is reported and left out of the reports. The body is a function (see pipeline.sh).
main() {
  set -uo pipefail
  local V; V=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
  local W=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}
  local OUT=${1:?usage: analyze-p1.sh OUT [KIT...]}; shift
  local IN=${P1_OUT:-$W/spec/board/out-nat-p1suite-b3} B=${P1_BOARD:-b3} K=$W/perf/p1suite/kits R=$V/results
  local kits=("$@"); [ ${#kits[@]} -gt 0 ] || kits=(p1k1 p1k2 p1k3 p1a1 p1a2 p1a3 p1a4)
  mkdir -p "$OUT"; local ok=() k ref
  for k in "${kits[@]}"; do
    local t=$IN/$k/board.txt
    [ -f "$t" ] || { echo "$k: no transcript $t"; continue; }
    case $k in p1k*) ref=$R/2026-10-28-t3heap-bitstream3/runs-t3k${k#p1k}.tsv.gz;; *) ref=$R/2026-10-27-t3all-bitstream3/runs-t3a${k#p1a}.tsv.gz;; esac
    python3 "$V/p1/p1-check.py" "$t" "$K/$k.steps.tsv" --ref "$ref" > "$OUT/check-$k.txt" 2>&1; local rc=$?
    echo "$k: p1-check rc=$rc $(tail -1 "$OUT/check-$k.txt") ($(head -1 "$OUT/check-$k.txt"))"
    python3 "$V/parse-board.py" "$t" --board "$B" --boot "p1-$k" --out "$OUT/runs-$k.tsv" > "$OUT/parse-$k.txt" 2>&1; local pc=$?
    echo "$k: parse-board rc=$pc $(tail -1 "$OUT/parse-$k.txt")"
    [ $rc = 0 ] && [ $pc = 0 ] && ok+=("$k")
  done
  [ ${#ok[@]} -gt 0 ] || { echo "no kit passed; no report"; return 1; }
  echo "reporting over: ${ok[*]}"
  local lab='BEGIN{OFS="\t"} {if (NR>1 && $3=="beebs" && $4 ~ /-libc$/) $3="beebs-libc"; print}'
  local first=1 heap=0
  : > "$OUT/all.tsv"; : > "$OUT/run.tsv"
  for k in "${ok[@]}"; do
    if [ $first = 1 ]; then cat "$OUT/runs-$k.tsv" >> "$OUT/all.tsv"; first=0; else tail -n +2 "$OUT/runs-$k.tsv" >> "$OUT/all.tsv"; fi
    case $k in p1k*) if [ $heap = 0 ]; then cat "$OUT/runs-$k.tsv"; heap=1; else tail -n +2 "$OUT/runs-$k.tsv"; fi >> "$OUT/run.tsv";; esac
  done
  awk -F'\t' "$lab" "$OUT/all.tsv" > "$OUT/all103.tsv"
  awk -F'\t' 'BEGIN{OFS="\t"} {if (NR>1 && $3=="beebs-libc") $3="beebs"; print}' "$OUT/all103.tsv" > "$OUT/groups.tsv"
  local a=(--arms T1,T3 --boot 10000 --seed 20261026)
  if [ $heap = 1 ]; then awk -F'\t' "$lab" "$OUT/run.tsv" > "$OUT/main.tsv"; python3 "$V/perf-analyze.py" report "$OUT/main.tsv" --out "$OUT/T1-T3" "${a[@]}"; fi
  python3 "$V/perf-analyze.py" report "$OUT/all103.tsv" --out "$OUT/all103/T1-T3" "${a[@]}"
  python3 "$V/perf-analyze.py" report "$OUT/groups.tsv" --out "$OUT/groups/T1-T3" "${a[@]}"
  for d in T1-T3 all103/T1-T3 groups/T1-T3; do [ -f "$OUT/$d/per-suite.tsv" ] && { echo "== $d/per-suite.tsv"; cat "$OUT/$d/per-suite.tsv"; }; done
}
main "$@"; exit
