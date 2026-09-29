#!/usr/bin/env bash
# Dynamic cost of CapstoneLiveSourceCopy on CoreMark, as an instruction count.
#
# One compiler, the same sources built twice: "rule" (the default) and "keep"
# (+movc-keeps-integer-source, every copy a movc). The portme is built with -DCAPSTONE_CORE_ICOUNT,
# so CoreMark's "Total ticks" is QEMU's retired-instruction count over the timed region, exact under
# -icount shift=0. A rewrite turns one movc into stc+ldc, or into a lone ldc when the slot already
# holds the source, so rule minus keep is the number of pair stores executed. This is an instruction
# count, not a cycle count: what an adjacent stc/ldc costs in cycles needs the RTL or the board.
#
# The rule arm also runs with CAPSTONE_MOVC_NULL_SCALAR=1 (the RTL's movc) and must still validate.
#
# Env: OUT (output dir), CAPSTONE_QEMU_BINARY (one with the MOVC switch), the usual test env.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../capstone-test-env.sh"
DEST=${OUT:?set OUT to an output directory}
unset OUT
REPO=$CAPSTONE_REPO_ROOT
CM=$REPO/capstone/benchmarks/coremark
RQ=$REPO/capstone/tests/runtime-qemu
ITER=${COREMARK_ITERATIONS:-10}
mkdir -p "$DEST"

run_arm() {  # arm switch
  local arm=$1 sw=$2 build=$DEST/$1/build log=$DEST/$1-sw$2.log
  CAPSTONE_MOVC_NULL_SCALAR=$sw capstone_with_qemu_lock python3 "$RQ/run-domain-smoke.py" \
    --share-dir "$build" --log-file "$log" --timeout-multiplier 4 \
    --qemu-extra-arg=-icount --qemu-extra-arg=shift=0,sleep=off \
    --guest-command 'cp /mnt/host/coremark_host.user /tmp/cm.user && chmod 0755 /tmp/cm.user && /tmp/cm.user /mnt/host/coremark_capstone.dom' \
    --success-marker 'Correct operation validated' > "$DEST/$arm-sw$sw.txt" 2>&1
  echo "rc=$?" >> "$DEST/$arm-sw$sw.txt"
}

for arm in rule keep; do
  mkdir -p "$DEST/$arm/build"
  extra=""
  [[ $arm == keep ]] && extra="-Xclang -target-feature -Xclang +movc-keeps-integer-source"
  printf '#!/usr/bin/env bash\nexec "%s" "$@" -DCAPSTONE_CORE_ICOUNT %s\n' "$CAPSTONE_CLANG" "$extra" > "$DEST/$arm/clang"
  chmod +x "$DEST/$arm/clang"
  if ! CLANG=$DEST/$arm/clang OUT_DIR=$DEST/$arm/build COREMARK_ITERATIONS=$ITER \
         bash "$CM/build-coremark-capstone.sh" > "$DEST/$arm-build.txt" 2>&1 ||
     ! OUT_DIR=$DEST/$arm/build bash "$CM/build-coremark-host.sh" >> "$DEST/$arm-build.txt" 2>&1; then
    echo "build failed for $arm: $DEST/$arm-build.txt"; exit 2
  fi
  run_arm "$arm" 0
done
run_arm rule 1

python3 - "$DEST" "$ITER" <<'PY'
import re, sys
d, it = sys.argv[1], int(sys.argv[2])
def read(tag):
    try: t = open(f"{d}/{tag}.log", errors="replace").read()
    except OSError: return None, False
    m = re.findall(r"Total ticks\s*:\s*(\d+)", t)
    return (int(m[-1]) if m else None), ("Correct operation validated" in t)
rows = {tag: read(tag) for tag in ("rule-sw0", "keep-sw0", "rule-sw1")}
for tag, (ticks, ok) in rows.items():
    print(f"{tag}: ticks {ticks} validated {ok}")
(r, rok), (k, kok), (_, r1ok) = rows["rule-sw0"], rows["keep-sw0"], rows["rule-sw1"]
if r is None or k is None or not (rok and kok):
    print("no measurement: a run did not validate or printed no ticks"); sys.exit(2)
if r == k == 0:
    print("both counts are 0: csrdicount read nothing (was -icount passed?)"); sys.exit(2)
print(f"coremark {it} iterations: rule {r} keep {k} delta {r-k} ({100*(r-k)/k:.2f} %), per iteration {(r-k)/it:.0f}")
print(f"rule under the RTL's movc (switch 1): {'validated' if r1ok else 'NOT VALIDATED'}")
sys.exit(0 if r1ok else 1)
PY
