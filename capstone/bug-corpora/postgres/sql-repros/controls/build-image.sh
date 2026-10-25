#!/usr/bin/env bash
# The postgres branch of capstone/ports/common/application/build-virtual.sh (at 53325bba), with
# this directory's build-domain.sh in place of the port's, so the image also links the controls'
# module (pgcorpus_lifetime). Everything else is build-virtual.sh's: a fresh output, the virtual
# profile, build.py with the same arguments.
#
#   bash build-image.sh NEW_OUTPUT                      # release: patch 0003 not applied
#   PGSU_NESTED=sublet bash build-image.sh NEW_OUTPUT   # protected: memory-contexts patch 0003
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../../tests/capstone-test-env.sh" >/dev/null
[[ $# == 1 ]] || { echo 'usage: build-image.sh NEW_OUTPUT' >&2; exit 2; }
OUT=$(realpath -m "$1")
[[ ! -e "$OUT" ]] || { echo "use a fresh output directory: $OUT" >&2; exit 2; }
mkdir -p "$OUT/source"
export CAPSTONE_APPLICATION_PROFILE=virtual
export PG_SU_ROOT=$OUT/source
bash "$HERE/build-domain.sh"
NESTED=none
[[ ${PGSU_NESTED:-none} != sublet ]] || NESTED=postgres
python3 "$CAPSTONE_REPO_ROOT/capstone/ports/common/application/build.py" --app postgres \
  --root "$OUT/source" --profile virtual --nested "$NESTED" --toolchain "$CAPSTONE_LLVM_BUILD_DIR" \
  --input-revision HEAD --stack-bytes 1048576 --out "$OUT/image"
