#!/usr/bin/env bash
# Build lua-SDL2 #75 at vulnerable (96491c0^) and fixed (96491c0) as ASan Lua C
# modules against system SDL2 2.32 + the shared Lua 5.4 toolchain. Also builds a
# non-ASan pair so run.sh can corroborate the UAF under valgrind (SDL2 itself is
# not ASan-instrumented, so the raw read-of-freed-window shows up under valgrind;
# under ASan the same free is caught by SDL's own "Invalid window" guard).
# Idempotent; everything lands under ./.work. Reproduced 2026-08-03.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); LC=$(cd "$HERE/.."; pwd); W="$HERE/.work"; mkdir -p "$W"

pkg-config --exists sdl2 || { echo "system SDL2 (pkg-config sdl2) required" >&2; exit 2; }

# Shared Lua toolchain: "user" build mode just needs the header dir (it selects
# code by LUA_VERSION_NUM). The module must NOT link Lua.
LUA=$("$LC/_toolchain/build-lua.sh")
if [ "$LUA" = SYSTEM ]; then LI=$(pkg-config --variable=includedir lua5.4); else LI="$LUA"; fi

[ -d "$W/luasdl2" ] || git clone https://github.com/Tangent128/luasdl2 "$W/luasdl2"
FIX=96491c0                 # PR #77: commonPush -> commonPushUserdata + mustdelete=0
VULN="$FIX^"                # its parent: the vulnerable tree

b(){ # $1=commit $2=outdir  $3=extra C flags
  git -C "$W/luasdl2" checkout -q "$1"
  rm -rf "$W/$2"
  cmake -S "$W/luasdl2" -B "$W/$2" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DWITH_LUAVER=user -DLUA_INCLUDE_DIR="$LI" \
    -DWITH_IMAGE=Off -DWITH_MIXER=Off -DWITH_TTF=Off -DWITH_NET=Off \
    -DCMAKE_C_FLAGS="-g -O0 -fno-omit-frame-pointer $3" \
    -DCMAKE_MODULE_LINKER_FLAGS="$3" >"$W/cmake.$2.log" 2>&1 \
    || { echo "cmake configure failed for $2 (see $W/cmake.$2.log)" >&2; return 1; }
  cmake --build "$W/$2" --target SDL -j4 >>"$W/cmake.$2.log" 2>&1 \
    || { echo "build failed for $2 (see $W/cmake.$2.log)" >&2; return 1; }
  [ -f "$W/$2/SDL.so" ] || { echo "no SDL.so for $2" >&2; return 1; }
}

b "$VULN" build-vuln         "-fsanitize=address"
b "$FIX"  build-fixed        "-fsanitize=address"
b "$VULN" build-vuln-noasan  ""
b "$FIX"  build-fixed-noasan ""
git -C "$W/luasdl2" checkout -q "$VULN"   # leave the tree on the vulnerable commit
echo "built: $W/{build-vuln,build-fixed,build-vuln-noasan,build-fixed-noasan}/SDL.so"
