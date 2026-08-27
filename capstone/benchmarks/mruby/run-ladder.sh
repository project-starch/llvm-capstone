#!/usr/bin/env bash
# Build every mruby ladder stage and run them in ONE boot, ascending.
#
#     tools/gen-specimen.sh cases/smoke.rb && ./run-ladder.sh
#
# One boot, because booting costs two to three minutes and dominates a short run,
# so N stages as N sessions is mostly boot time. Ascending, because a wedge takes
# the domain with it and everything after the first wedge is lost -- which is not a
# limitation to engineer around: the first stage that fails to return IS the
# bisection point.
#
# Every stage returns a marker rather than running to a failure, so a run always
# yields data. A stage that returns nothing is itself the result.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

STAGES=${STAGES:-"0 1 2 3 4"}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/mruby-silicon}
SHARE=${SHARE_DIR:-$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-share}
LOG=${LOG_FILE:-$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-mruby.log}
mkdir -p "$SHARE"

CMD=""
for s in $STAGES; do
  echo "== building stage $s"
  MD_STAGE="$s" DOM_NAME="mruby-s$s" bash "$SCRIPT_DIR/build-mruby-silicon.sh" >/dev/null
  cp -f "$OUT_DIR/mruby-s$s.dom" "$SHARE/"
  CMD="$CMD/mnt/host/capstone-test.user /mnt/host/mruby-s$s.dom 1; "
done

echo "== one boot, stages $STAGES ascending"
CAPSTONE_QUIET_GP=1 python3 "$REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE" --log-file "$LOG" --timeout-multiplier 6 \
  --guest-command "$CMD" \
  "$SHARE/mruby-s$(echo $STAGES | awk '{print $1}').dom" || true

python3 - "$LOG" <<'PY'
import re, sys
t = open(sys.argv[1], 'rb').read().decode('utf8', 'replace')
vals = [int(x) & 0xFFFFFFFF for x in re.findall(r'retval = (\d+)', t)]
tag = [v for v in vals if (v >> 16) == 0x6D52]
print(f"\n  markers: {len(tag)}")
for v in tag:
    stage, code = (v >> 8) & 0xFF, v & 0xFF
    if stage == 4:
        print(f"    stage 4  ruby returned {code}")
    elif stage == 3:
        print(f"    stage 3  region took, {code} pages"
              + ("   <-- ZERO pages means the region did NOT take" if code == 0 else ""))
    else:
        print(f"    stage {stage}  code 0x{code:02X}" + ("  OK" if code == 1 else ""))
for l in t.splitlines():
    if 'capability fault' in l:
        print("   " + l.strip()[:130])
if not tag:
    print("   NO MARKER AT ALL -- read the log before calling this a failure;"
          " a boot flake and a wedge look the same from here")
PY
