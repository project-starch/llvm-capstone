#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../tests/capstone-test-env.sh"
out=${1:?Output directory}
musl=${2:?Prepared Capstone musl source tree}
shift 2
export CAPSTONE_APPLICATION_PROFILE=virtual
for patchfile in "$here/../../ports/musl-capstone/musl-patches/"*.patch; do
    if ! patch -d "$musl" -p1 -R --dry-run -s -f < "$patchfile" >/dev/null 2>&1; then
        echo "musl source is not current: run prepare-musl-capstone.sh (missing $(basename "$patchfile"))" >&2
        exit 2
    fi
done
# These options affect libc now, as well as the runtime. The policy symbol in
# malloc-mapping.c makes mixing differently configured archives a link error.
heap_flags=()
for setting in "$@"; do
    case "$setting" in
      -DCAPSTONE_HEAP_LIFETIMES=*|-DCAPSTONE_HEAP_OBJECT_BOUNDS=*|-DCAPSTONE_HEAP_SCRUB_FREE=*)
        key=${setting%%=*}; value=${setting#*=}
        case "${value^^}" in
          ON|TRUE|YES|1) value=1 ;;
          OFF|FALSE|NO|0) value=0 ;;
          *) echo "invalid heap option: $setting" >&2; exit 2 ;;
        esac
        heap_flags+=("$key=$value") ;;
    esac
done
export MUSL_CAPSTONE_EXTRA_CFLAGS="${MUSL_CAPSTONE_EXTRA_CFLAGS:-} ${heap_flags[*]}"
mkdir -p "$out/libc"
mkdir -p "$out/libc/obj"
rm -f "$out/libc/obj/"*.o "$out/libc/libc-capstone-virtual.a"
python3 "$here/../../ports/musl-capstone/survey-musl-capstone.py" "$musl" \
    --profile virtual --objects "$out/libc/obj" --list-failures --jobs "${JOBS:-16}"
"$CAPSTONE_LLVM_BIN/llvm-ar" rcs "$out/libc/libc-capstone-virtual.a" "$out/libc/obj/"*.o
bash "$here/../../ports/common/application/build-sdk.sh" "$out/sdk" "$musl" \
    "$out/libc/libc-capstone-virtual.a" -DCAPSTONE_APPLICATION_VIRTUAL=ON \
    -DCAPSTONE_APPLICATION_DATA_BYTES=2097152 "$@"
