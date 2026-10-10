#!/usr/bin/env bash
# Fresh source builds: changing only the final link cannot migrate physical objects.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../tests/capstone-test-env.sh" >/dev/null
[[ $# == 2 ]] || { echo 'usage: build-virtual.sh {cpython|postgres|ffmpeg|tshark} NEW_OUTPUT' >&2; exit 2; }
APP=$1
OUT=$(realpath -m "$2")
[[ ! -e "$OUT" ]] || { echo "use a fresh output directory: $OUT" >&2; exit 2; }
case "$APP" in cpython|postgres|ffmpeg|tshark) ;; *) echo "unsupported application: $APP" >&2; exit 2;; esac
# Specialized decoder pools keep their source recipe's link.
# The generic relinker does not carry their extra allocator objects.
if [[ $APP == ffmpeg && -n ${FFAPP_POOL:-} ]]; then
  echo 'use the application source recipe directly for this inner allocator variant' >&2
  exit 2
fi
mkdir -p "$OUT/source"
export CAPSTONE_APPLICATION_PROFILE=virtual
PORTS=$CAPSTONE_REPO_ROOT/capstone/ports
NESTED=none
LIBC_FLAGS=()
case "$APP" in
  cpython)
    export CPY_ROOT=$OUT/source
    bash "$PORTS/cpython/app/prepare-cpython-capstone.sh"
    source "$CPY_ROOT/build/capstone-env.sh"
    python3 "$PORTS/cpython/app/survey-cpython-capstone.py" "$CPY_ROOT/build" --jobs "${JOBS:-12}"
    python3 "$PORTS/cpython/app/link-cpython-capstone.py" "$CPY_ROOT/build" \
      --native "$CPY_ROOT/build-python" --out "$CPY_ROOT/link"
    [[ ${CPY_SUBLET:-0} != 1 ]] || NESTED=cpython
    ;;
  postgres)
    export PG_SU_ROOT=$OUT/source
    bash "$PORTS/postgres/app/build-domain.sh"
    [[ ${PGSU_NESTED:-none} != sublet ]] || NESTED=postgres
    ;;
  ffmpeg)
    export FFAPP_WORK=$OUT/source
    bash "$PORTS/ffmpeg/app/host/build-domain.sh"
    ;;
  tshark)
    export TS_WORK=$OUT/source
    for lib in zlib pcre2 c-ares libgpg-error libgcrypt libxml2 glib; do
      bash "$PORTS/wireshark/app/deps/build-$lib.sh"
    done
    bash "$PORTS/wireshark/app/host/cross-build.sh"
    bash "$PORTS/wireshark/app/host/build-domain.sh"
    source "$PORTS/wireshark/app/deps/env.sh"
    LIBC_FLAGS=(--musl "$TS_MUSL" --libc "$TS_LIBC_ARCHIVE")
    ;;
esac
python3 "$HERE/build.py" --app "$APP" --root "$OUT/source" \
  --profile virtual --nested "$NESTED" --toolchain "$CAPSTONE_LLVM_BUILD_DIR" \
  --input-revision HEAD "${LIBC_FLAGS[@]}" --out "$OUT/image"
