#!/bin/bash
# Build the pinned llvm-capstone toolchain and the virtual application SDK from the source archive
# that manifest.lock.json names. Every input is checked against the manifest first.
# usage: toolchain.sh KIT WORKDIR
#   KIT      this kit (manifest.lock.json, artifacts/)
#   WORKDIR  empty or absent; gets src/, build/, musl/, sdk/, tmp/ and toolchain-record.txt
set -euo pipefail
KIT=$(cd "$1" && pwd) W=$2
[ -e "$W" ] && [ -n "$(ls -A "$W" 2>/dev/null)" ] && { echo "toolchain: $W is not empty" >&2; exit 2; }
mkdir -p "$W"; W=$(cd "$W" && pwd)
m() { python3 -c "import json,sys;d=json.load(open(sys.argv[1]));[d:=d[k] for k in sys.argv[2].split('.')];print(d)" "$KIT/manifest.lock.json" "$1"; }
R=$W/toolchain-record.txt; : > "$R"
note() { echo "$*" | tee -a "$R"; }
check() { # file expected-sha256
  local got; got=$(sha256sum "$1" | cut -c1-64)
  [ "$got" = "$2" ] || { echo "toolchain: $1 is $got, manifest says $2" >&2; exit 1; }
}
SRC=$KIT/artifacts/sources/$(m toolchain.archive.file)
MUSL=$KIT/artifacts/sources/$(m toolchain.musl.file)
check "$SRC" "$(m toolchain.archive.sha256)"
check "$MUSL" "$(m toolchain.musl.sha256)"
note "llvm-capstone commit $(m toolchain.commit) tree $(m toolchain.tree) archive $(m toolchain.archive.sha256)"
note "musl $(m toolchain.musl.sha256) $(basename "$MUSL")"
mkdir -p "$W/src" "$W/build" "$W/musl-src" "$W/tmp"
zstd -dc "$SRC" | tar -C "$W/src" -xf -
T=$W/src/llvm-capstone
cp "$MUSL" "$W/musl-src/"
JOBS=${JOBS:-$(nproc)}
cd "$W/build"
cmake -G Ninja "$T/llvm" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_TARGETS_TO_BUILD=Capstone -DLLVM_ENABLE_PROJECTS='clang;lld' -DLLVM_PARALLEL_LINK_JOBS=4 \
  -DLLVM_INCLUDE_BENCHMARKS=OFF > cmake.log 2>&1
ninja -j "$JOBS" clang lld llc opt FileCheck count not llvm-config llvm-objdump llvm-nm llvm-ar llvm-objcopy \
  llvm-readobj llvm-readelf llvm-symbolizer split-file llvm-mc > ninja.log 2>&1
note "host $(uname -srm) cmake $(cmake --version | head -1 | awk '{print $3}') ninja $(ninja --version) cc $(cc --version | head -1)"
note "clang $(sha256sum bin/clang-[0-9]* | cut -c1-64) $("$W/build/bin/clang" --version | head -1)"
( cd "$T" && export CAPSTONE_LLVM_BUILD_DIR=$W/build CAPSTONE_LLVM_BIN=$W/build/bin CAPSTONE_TMP_ROOT=$W/tmp \
    MUSL_CACHE_ROOT=$W/musl-src CAPSTONE_APPLICATION_PROFILE=virtual MUSL_SURVEY_JOBS=$JOBS &&
  source capstone/tests/capstone-test-env.sh > /dev/null 2>&1 &&
  MUSLDIR=$(bash capstone/ports/musl-capstone/prepare-musl-capstone.sh | tail -1) &&
  OUT_DIR=$W/musl bash capstone/ports/musl-capstone/build-musl-capstone.sh &&
  bash capstone/ports/common/application/build-sdk.sh "$W/sdk" "$MUSLDIR" "$W/musl/libc-capstone.a" ) > "$W/sdk.log" 2>&1
note "sdk $(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get('profile'),d.get('cc'))" "$W/sdk/sdk.json")"
note "libc $(sha256sum "$W/musl/libc-capstone.a" | cut -c1-64)"
# The SDK's complete content, so a reader can tell two SDKs apart file by file.
( cd "$W/sdk" && find -L . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum ) > "$W/sdk-inventory.sha256"
note "sdk-inventory $(sha256sum "$W/sdk-inventory.sha256" | cut -c1-64) ($(wc -l < "$W/sdk-inventory.sha256") files)"
note TOOLCHAIN_DONE
