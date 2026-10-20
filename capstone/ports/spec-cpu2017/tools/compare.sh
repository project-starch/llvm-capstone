#!/bin/bash
# The arms side by side: every benchmark at one size, every arm of BUILD_RECORD/arms.json, REPS times,
# one freshly booted guest per run pinned to its own host CPU. The CPU list is cut into groups of one
# CPU per arm; the arms of one benchmark start together on one group, longest benchmark first, and the
# next benchmark takes a group as soon as one is free. The arm-to-CPU assignment within a group rotates
# between repetitions. A sampler writes the host load average and every guest QEMU's CPU time once a
# minute into OUT/load.tsv; tools/compare-report.py reads it all and flags repetitions whose host load
# broke the pre-registered bound.
# usage: compare.sh KIT SPEC_ROOT TOOLCHAIN PLATFORM BUILD_RECORD SIZE OUT REPS CPU_LIST [MAX_LOAD]
#   CPU_LIST  comma-separated host CPUs, a multiple of the number of arms, at least one group
#   MAX_LOAD  host 1-minute load average above which a repetition is flagged (default: guests + 8)
set -uo pipefail
KIT=$(cd "$1" && pwd) S=$(cd "$2" && pwd) TC=$(cd "$3" && pwd) PL=$(cd "$4" && pwd) BREC=$(cd "$5" && pwd)
SIZE=$6 OUT=$7 REPS=$8 CPUS=$9 MAX_LOAD=${10:-}
[ -e "$OUT" ] && { echo "compare: $OUT exists" >&2; exit 2; }
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
M=$KIT/manifest.lock.json
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/compare.log"; }
mapfile -t BENCH < <(python3 -c "import json,sys;print('\n'.join(json.load(open(sys.argv[1]))['workloads']['benchmarks']))" "$M")
if [ -n "${BENCHMARKS:-}" ]; then read -r -a BENCH <<< "$BENCHMARKS"; fi   # a subset, as for build.sh
mapfile -t LABELS < <(python3 -c "import json,sys;print('\n'.join(json.load(open(sys.argv[1]))))" "$BREC/arms.json")
IFS=, read -r -a CPU <<< "$CPUS"
A=${#LABELS[@]}
[ $(( ${#CPU[@]} % A )) = 0 ] && [ ${#CPU[@]} -ge $A ] || { echo "compare: the CPU list must hold whole groups of $A (one CPU per arm), ${#CPU[@]} given" >&2; exit 2; }
GROUPS_N=$(( ${#CPU[@]} / A ))
[ -n "$MAX_LOAD" ] || MAX_LOAD=$(( ${#CPU[@]} + 8 ))
# Longest first (train wall minutes of the qualification, q4), so the long poles start in the first wave.
ORDER="502.gcc_r 500.perlbench_r 544.nab_r 525.x264_r 505.mcf_r 538.imagick_r 557.xz_r 519.lbm_r"
SORTED=(); for b in $ORDER; do for x in "${BENCH[@]}"; do [ "$x" = "$b" ] && SORTED+=("$b"); done; done
for x in "${BENCH[@]}"; do case " $ORDER " in *" $x "*) ;; *) SORTED+=("$x");; esac; done
python3 - "$OUT/plan.json" "$SIZE" "$REPS" "$MAX_LOAD" "${LABELS[@]}" -- "${SORTED[@]}" -- "${CPU[@]}" <<'PY'
import json, sys
out, size, reps, max_load = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
rest = sys.argv[5:]; i = rest.index('--'); labels = rest[:i]; rest = rest[i+1:]; j = rest.index('--'); bench = rest[:j]; cpus = rest[j+1:]
groups = [cpus[g:g + len(labels)] for g in range(0, len(cpus), len(labels))]
plan = {"size": size, "repetitions": reps, "max_load": max_load, "labels": labels, "benchmarks": bench, "cpus": cpus, "groups": groups,
        "rule": "benchmarks start longest first, each on the first free group of CPUs, its arms together; within the group the arm order rotates by the repetition number; a benchmark's group in a repetition is recorded in assignment when it starts"}
json.dump(plan, open(out, 'w'), indent=1)
PY
log "plan: $A arms (${LABELS[*]}), ${#SORTED[@]} benchmarks, $REPS repetitions, size $SIZE, $GROUPS_N groups of $A CPUs, max load $MAX_LOAD"
# Load sampler, until the runs are done.
( while [ ! -f "$OUT/RUNS_DONE" ]; do
    printf '%s\t%s\t' "$(date -u +%FT%TZ)" "$(cut -d' ' -f1-3 /proc/loadavg)"
    for p in /proc/[0-9]*; do
      [ -r "$p/cmdline" ] || continue
      case "$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null)" in "$PL/qemu-system-riscv64 "*"$OUT/"*) printf '%s:%s ' "${p#/proc/}" "$(awk '{print $14+$15}' "$p/stat" 2>/dev/null)";; esac
    done; echo
    sleep 60
  done ) > "$OUT/load.tsv" 2>/dev/null &
sampler=$!
# One benchmark = the arms together on one group: a subshell that returns when its last arm has.
triple() { # rep bench group-index
  local rep=$1 b=$2 g=$3 a cpu lab pids=()
  for a in $(seq 0 $((A - 1))); do
    lab=${LABELS[$a]}; cpu=${CPU[$(( g * A + (a + rep - 1) % A ))]}
    mkdir -p "$OUT/rep$rep/$lab"
    echo "$(date -u +%FT%TZ) rep $rep $b $lab cpu $cpu group $g" >> "$OUT/assignment.log"
    taskset -c "$cpu" bash "$KIT/tools/run.sh" "$KIT" "$S" "$TC" "$PL" "$BREC" "$SIZE" "$b" "$OUT/rep$rep/$lab/$b" "$lab" \
      > "$OUT/rep$rep/$lab/$b.out" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
}
for rep in $(seq 1 "$REPS"); do
  log "repetition $rep: start"
  declare -A busy=()   # pid -> group
  free=(); for g in $(seq 0 $((GROUPS_N - 1))); do free+=("$g"); done
  for b in "${SORTED[@]}"; do
    while [ ${#free[@]} = 0 ]; do
      wait -n -p done_pid "${!busy[@]}" || true
      [ -n "${done_pid:-}" ] && { free+=("${busy[$done_pid]}"); unset "busy[$done_pid]"; }
    done
    g=${free[0]}; free=("${free[@]:1}")
    triple "$rep" "$b" "$g" & busy[$!]=$g
    log "rep $rep $b started on group $g (CPUs ${CPU[*]:$((g * A)):$A})"
  done
  [ ${#busy[@]} = 0 ] || wait "${!busy[@]}"
  for b in "${SORTED[@]}"; do for lab in "${LABELS[@]}"; do
    log "rep $rep $lab $b $(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['verdict'],d.get('reported_seconds'),'s qemu',d.get('qemu_cpu_seconds'),'s |',';'.join(d['reasons']))" "$OUT/rep$rep/$lab/$b/result.json" 2>&1)"
  done; done
done
touch "$OUT/RUNS_DONE"; wait "$sampler" 2>/dev/null
python3 "$KIT/tools/compare-report.py" "$OUT" > "$OUT/report.md" 2> "$OUT/report.err" || log "report failed (see report.err)"
log "done: $OUT/report.md"
