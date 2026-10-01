#!/usr/bin/env bash
# Structured application results, with the original registered predictions.
# FFAPP_SAFETY_EXPECT / FFAPP_SAFETY_VERDICT: another set of predictions and the verdict that
# reads them, for images this script runs but the port's classifier cannot judge (the bug
# corpus's cases, whose fault is named by source line: bug-corpora/ffmpeg/pool-repros/runners/).
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ARM=${1:?level0, shrink, sublet, pool0, pool2, poolsublet or poolstock}; shift
case $ARM in
 level0) DOMAIN=domain ;; shrink|sublet) DOMAIN=domain-$ARM ;;
 pool0|pool2|poolsublet|poolstock) DOMAIN=domain-sublet-$ARM ;;
 *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac
WORK=${FFAPP_WORK:-/tmp/capstone/ffmpeg-app}
JUDGE=()
[ -z "${FFAPP_SAFETY_EXPECT:-}" ] || JUDGE+=(--expect "$FFAPP_SAFETY_EXPECT")
[ -z "${FFAPP_SAFETY_VERDICT:-}" ] || JUDGE+=(--verdict "$FFAPP_SAFETY_VERDICT")
exec python3 "$HERE/../../../common/application/check-safety.py" \
 --state "${CAPSTONE_VM_STATE:?running VM state required}" --port ffmpeg --arm "$ARM" \
 --images "$WORK/$DOMAIN" --out "${FFAPP_SAFETY_OUT:?new result directory required}" "${JUDGE[@]}" "$@"
