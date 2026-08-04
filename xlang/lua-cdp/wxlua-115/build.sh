#!/usr/bin/env bash
# Build the toolchain for wxLua #115: the standalone `wxLua` interpreter at BOTH
# the vulnerable and the fixed commit, against system wxWidgets 3.2 + the shared
# toolchain Lua 5.4, with AddressSanitizer + debug info.
#
# The ONLY code difference between the two commits is one file,
# modules/wxbind/src/wxcore_menutool.cpp (the fix adds a %ungc /
# wxluaO_undeletegcobject call to AppendSubMenu; the .i interface file it is
# generated from is not compiled). So we configure + full-build once at the
# vulnerable commit, snapshot it, then `git checkout` the fixed tree and let the
# incremental build recompile just that one TU + relink. Two self-contained
# artifact dirs come out: .work/vuln and .work/fix.
#
# Idempotent; everything lands under ./.work. Reproduced 2026-08-03.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LC=$(cd "$HERE/.." && pwd)              # xlang/lua-cdp
W="$HERE/.work"; mkdir -p "$W"

VULN=b5ffaccac0bbb2587952a932e5f80abc7c083a35   # parent of the fix = vulnerable
FIX=ded8e0a3e6b19bbb752c68282a9e37e9b88b7582    # "Fixed double freeing ... (#115)"
WXREPO="$W/wxlua"; SRC="$WXREPO/wxLua"; BUILD="$W/build"

# --- shared toolchain Lua 5.4 (prebuilt; build-lua.sh is idempotent) ----------
LUA_OUT=$("$LC/_toolchain/build-lua.sh")
if [ "$LUA_OUT" = "SYSTEM" ]; then
  # ponytail: best-effort system branch. This machine has no lua5.4-dev, so the
  # from-source path below is what actually runs; upgrade here if a host ships one.
  LUA_INC=$(pkg-config --variable=includedir lua5.4)
  LUA_LIB=$(pkg-config --variable=libdir lua5.4)/liblua5.4.so
else
  LUA_INC="$LUA_OUT"; LUA_LIB="$LUA_OUT/liblua5.4.so"
fi
[ -f "$LUA_LIB" ] && [ -f "$LUA_INC/lua.h" ] || { echo "toolchain lua 5.4 not found ($LUA_LIB / $LUA_INC/lua.h)" >&2; exit 2; }

command -v wx-config >/dev/null || { echo "wx-config (wxWidgets 3.2) not found" >&2; exit 2; }
echo "wxWidgets $(wx-config --version), lua inc=$LUA_INC"

# --- fetch wxLua --------------------------------------------------------------
[ -d "$WXREPO/.git" ] || git clone --filter=blob:none https://github.com/pkulchenko/wxlua "$WXREPO"

ASANF="-fsanitize=address -fno-omit-frame-pointer -g -O0"

configure() {
  [ -f "$BUILD/CMakeCache.txt" ] && return 0
  # wxLuaBind_COMPONENTS=core;base  -> only build the wxcore+wxbase bindings
  #   (wxMenu lives in wxcore); avoids the heavy adv/aui/gl/... binding modules.
  # wxWidgets_COMPONENTS adds html+adv+net so the core binding's stray refs
  #   (wxHtmlHelpController, wxSocketBase, ...) resolve at link time without
  #   building those bindings.
  # CMAKE_POLICY_VERSION_MINIMUM=3.5 lets CMake 4.x accept wxLua's 2.8 minimum.
  cmake -S "$SRC" -B "$BUILD" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_BUILD_TYPE=Debug \
    -DwxWidgets_CONFIG_EXECUTABLE="$(command -v wx-config)" \
    -DwxWidgets_COMPONENTS="html;adv;net;core;base" \
    -DwxLuaBind_COMPONENTS="core;base" \
    -DwxLua_LUA_LIBRARY_USE_BUILTIN=FALSE \
    -DwxLua_LUA_LIBRARY_VERSION=5.4 \
    -DwxLua_LUA_INCLUDE_DIR="$LUA_INC" \
    -DwxLua_LUA_LIBRARY="$LUA_LIB" \
    -DCMAKE_CXX_FLAGS="$ASANF" \
    -DCMAKE_C_FLAGS="$ASANF" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"
}

build_into() { # $1 = commit  $2 = dest dir
  git -C "$WXREPO" checkout -q "$1"
  configure
  cmake --build "$BUILD" --target wxLua_app -- -j"$(nproc)"
  rm -rf "$2"; mkdir -p "$2"
  cp -a "$BUILD/bin/Debug/wxLua" "$2/wxLua"
  cp -a "$BUILD"/lib/Debug/*.so* "$2/"
  printf '%s' "$LUA_INC" > "$2/.lua_inc"     # run.sh reads this for LD_LIBRARY_PATH
}

echo "== building VULNERABLE ($VULN) =="; build_into "$VULN" "$W/vuln"
echo "== building FIXED      ($FIX) =="; build_into "$FIX"  "$W/fix"
git -C "$WXREPO" checkout -q "$VULN"          # leave the tree on the vulnerable commit

echo "built: $W/vuln/wxLua and $W/fix/wxLua (ASan, wxWidgets 3.2, lua 5.4)"
