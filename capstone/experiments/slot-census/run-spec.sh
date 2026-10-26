#!/bin/bash
# Slot census of SPEC CPU2017 rate (C and C++), native x86, under the slotcensus LD_PRELOAD tracer.
#
# usage: run-spec.sh SPEC_ROOT OUT_DIR [SIZE] [BENCHMARK...]
#   SPEC_ROOT  an installed SPEC CPU2017 tree (shrc, config/Example-gcc-linux-x86.cfg)
#   OUT_DIR    new directory for this pass: controls, one census file and one runcpu log per benchmark, provenance
#   SIZE       test, train or ref (default ref)
# Steps: build the tracer and the controls; run the controls (a failing control exits 75 with no census); write
# config/slotcensus.cfg from SPEC's own GCC example (label census, the system gcc, -O2, the portability flags below);
# build every benchmark once; run each benchmark in its own runcpu, all in parallel, each writing its census to
# OUT_DIR/<benchmark>.jsonl; runcpu validates every output with SPEC's own tools.
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SPEC_ROOT=$(cd "$1" && pwd); OUT=$2; SIZE=${3:-ref}; shift 3 2>/dev/null || shift $#
BENCH=("$@")
[ ${#BENCH[@]} -gt 0 ] || BENCH=(500.perlbench_r 502.gcc_r 505.mcf_r 520.omnetpp_r 523.xalancbmk_r 525.x264_r
                                  531.deepsjeng_r 541.leela_r 557.xz_r 508.namd_r 510.parest_r 511.povray_r
                                  519.lbm_r 526.blender_r 538.imagick_r 544.nab_r)
[ -e "$OUT" ] && { echo "run-spec: $OUT exists" >&2; exit 2; }
mkdir -p "$OUT/logs"; OUT=$(cd "$OUT" && pwd)
TOOL=$OUT/tool

# 1-2. the tracer and the controls
. "$HERE/tracer.sh"
build_tracer
run_controls

# 3. the SPEC config, from SPEC's example
cd "$SPEC_ROOT" && source shrc > /dev/null || exit 1
CFG=config/slotcensus.cfg
sed -e 's/^%define label mytest .*/%define label census/' \
    -e 's#^%   define  gcc_dir .*#%   define  gcc_dir        /usr#' \
    -e 's/^   OPTIMIZE       = -g -O3 .*/   OPTIMIZE       = -O2/' \
    config/Example-gcc-linux-x86.cfg > "$CFG"
cat >> "$CFG" <<EOF

# --- slot census ---
%ifndef %{census_out}
%   define census_out /dev/null
%endif
default=base:
   EXTRA_CFLAGS   = -fcommon -fpermissive
   EXTRA_CXXFLAGS = -std=gnu++14 -fpermissive
502.gcc_r=base:
   EXTRA_CFLAGS   = -fcommon -fpermissive -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -fgnu89-inline
526.blender_r=base:
   EXTRA_CFLAGS   = -fcommon -fpermissive -std=gnu99
default:
   preENV_LD_PRELOAD    = $TOOL/slotcensus.so
   preENV_SLOTCENSUS_OUT = %{census_out}
EOF
cp "$CFG" "$OUT/slotcensus.cfg"

# 4. provenance
{
  provenance_tracer
  echo "size $SIZE"
  echo "benchmarks ${BENCH[*]}"
} > "$OUT/provenance.txt"

# 5. build every benchmark once (the compilers run under the tracer too; census_out is /dev/null here)
s=$(date +%s)
runcpu --config=slotcensus --action=build --define build_ncpus=32 "${BENCH[@]}" > "$OUT/logs/build.log" 2>&1
echo "build rc=$? $(( $(date +%s) - s ))s" | tee -a "$OUT/provenance.txt"
# runcpu exits 0 even when a benchmark fails to build; it says so in the log
grep -E '^Build errors' "$OUT/logs/build.log" | tee -a "$OUT/provenance.txt"

# 6. run each benchmark in its own runcpu, in parallel
s=$(date +%s)
for b in "${BENCH[@]}"; do
  ( t=$(date +%s)
    runcpu --config=slotcensus --nobuild --action=run --size="$SIZE" --iterations=1 --noreportable \
           --define census_out="$OUT/$b.jsonl" "$b" > "$OUT/logs/$b.log" 2>&1
    echo "$b rc=$? $(( $(date +%s) - t ))s" >> "$OUT/runs.txt" ) &
done
wait
echo "runs $(( $(date +%s) - s ))s" | tee -a "$OUT/provenance.txt"
echo CENSUS_DONE >> "$OUT/runs.txt"
