#!/usr/bin/env bash
# Build the two virtual-profile SDKs of the microbenchmark comparison, one per arm:
#   T1  Sublet spatial only:   -DCAPSTONE_HEAP_LIFETIMES=OFF -DCAPSTONE_HEAP_SCRUB_FREE=OFF
#   T2  spatial + temporal:    the defaults (both ON)
# Both with CAPSTONE_VIRTUAL_MOVC_KEEP=0 (board images carry -movc-keeps-integer-source, C-32).
# Both arms share ONE prepared musl tree, so sdk.json's "musl" field and every libc object agree.
#
# usage: build-sdks.sh OUT_ROOT MUSL_CACHE_DIR [SOURCE_TREE]
#   -> OUT_ROOT/sdk-T1, OUT_ROOT/sdk-T2, OUT_ROOT/heap-layout.json, OUT_ROOT/sdk-integrity.txt,
#      OUT_ROOT/sdk-source.txt
# SOURCE_TREE (default: the tree this script is in) is the checkout whose runtime and musl port the
# SDKs are built from -- e.g. a worktree carrying a runtime fix not yet on this branch. Its commit
# and dirty state are recorded in sdk-source.txt.
# Requires CAPSTONE_LLVM_BIN and CAPSTONE_LLVM_BUILD_DIR in the environment.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=${1:?output root}; cache=${2:?directory holding musl-1.2.5.tar.gz}
src=$(cd -- "${3:-$here/../../..}" && pwd)
: "${CAPSTONE_LLVM_BIN:?}" "${CAPSTONE_LLVM_BUILD_DIR:?}"
export CAPSTONE_REPO_ROOT=$src CAPSTONE_VIRTUAL_MOVC_KEEP=0 JOBS=${JOBS:-12}
unset CAPSTONE_CLANG
mkdir -p "$out/musl-cache"
cp --update=none "$cache"/*.tar.gz "$out/musl-cache/"
{ echo "source_tree $src"
  echo "head $(git -C "$src" rev-parse HEAD)"
  echo "dirty_tracked_files $(git -C "$src" status --porcelain --untracked-files=no | wc -l)"
  git -C "$src" status --porcelain --untracked-files=no -- capstone/runtime capstone/ports/musl-capstone | sed 's/^/dirty /'
  echo "compiler $CAPSTONE_LLVM_BIN"
  echo "clang_sha256 $(sha256sum "$CAPSTONE_LLVM_BIN/clang" | cut -c1-64)"
} > "$out/sdk-source.txt"
cd "$src"
musl=$(MUSL_CACHE_ROOT=$out/musl-cache bash capstone/ports/musl-capstone/prepare-musl-capstone.sh 2> "$out/musl.log" | tail -1)
[[ -d $musl ]] || { echo "musl preparation failed (see $out/musl.log)" >&2; exit 1; }
for arm in T1 T2; do
  extra=()
  [[ $arm == T1 ]] && extra=(-DCAPSTONE_HEAP_LIFETIMES=OFF -DCAPSTONE_HEAP_SCRUB_FREE=OFF)
  rm -rf "$out/sdk-$arm"
  bash capstone/runtime/virtual/build-sdk.sh "$out/sdk-$arm" "$musl" "${extra[@]}" > "$out/sdk-$arm.log" 2>&1 \
    || { echo "SDK $arm failed (see $out/sdk-$arm.log)" >&2; exit 1; }
done
python3 - "$out/sdk-T1/sdk/sdk.json" "$out/sdk-T2/sdk/sdk.json" <<'PY'
import json, sys
want = {0: (False, False), 1: (True, True)}
for i, path in enumerate(sys.argv[1:]):
    c = json.load(open(path))
    if (c["heap_lifetimes"], c["heap_scrub_free"]) != want[i]:
        sys.exit(f"{path}: heap switches {c['heap_lifetimes']},{c['heap_scrub_free']} != {want[i]}")
    if "-movc-keeps-integer-source" not in c["codegen"] or "+movc-keeps-integer-source" in c["codegen"]:
        sys.exit(f"{path}: codegen does not carry -movc-keeps-integer-source")
print("sdk.json switches OK")
PY
python3 "$here/check-sdks.py" "$out/sdk-T1" "$out/sdk-T2" > "$out/sdk-integrity.txt" \
  || { echo "SDK integrity check FAILED (see $out/sdk-integrity.txt)" >&2; exit 1; }
tail -1 "$out/sdk-integrity.txt"
python3 "$here/heap-layout.py" --sdk "$out/sdk-T1/sdk" --sdk "$out/sdk-T2/sdk" --out "$out/heap-layout.json" > /dev/null
echo "heap layout record: $out/heap-layout.json"
