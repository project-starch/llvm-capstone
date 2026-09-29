#!/usr/bin/env bash
# Structured application results, with the original registered predictions.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ARM=${1:?level0, shrink, sublet, pool0, pool2, poolsublet or poolstock}; shift
case $ARM in
 level0) DOMAIN=domain ;; shrink|sublet) DOMAIN=domain-$ARM ;;
 pool0|pool2|poolsublet|poolstock) DOMAIN=domain-sublet-$ARM ;;
 *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac
WORK=${FFAPP_WORK:-/tmp/capstone/ffmpeg-app}
exec python3 "$HERE/../../../common/application/check-safety.py" \
 --state "${CAPSTONE_VM_STATE:?running VM state required}" --port ffmpeg --arm "$ARM" \
 --images "$WORK/$DOMAIN" --out "${FFAPP_SAFETY_OUT:?new result directory required}" "$@"
