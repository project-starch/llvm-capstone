#!/usr/bin/env bash
# Build lgi's C core against the shared toolchain Lua 5.4, in two trees:
#   fixed/  = pinned HEAD (7a2276f) as-is
#   vuln/   = HEAD with the issue-122 fix commit 94f970d8 REVERTED
#             ("Make objects unusable in the __gc metamethod" -> the metatable-nil
#              guard in record_gc). Reverting it re-opens the cairo.Region UAF.
#
# Why revert-onto-HEAD instead of checking out the historical vulnerable tree:
# the pre-fix tree (2017) predates lgi's Lua 5.4 support (it calls the 3-arg
# lua_resume) and will NOT compile against the toolchain Lua. Reverting only the
# 3-line guard onto a buildable HEAD is the faithful minimal reconstruction.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); LC=$(cd "$HERE/.."; pwd); W="$HERE/.work"; mkdir -p "$W"
PIN=7a2276f; FIX=94f970d8

command -v valgrind >/dev/null || { echo "need valgrind" >&2; exit 2; }
pkg-config --exists 'gobject-introspection-1.0 gmodule-2.0 libffi cairo cairo-gobject gio-2.0' \
  || { echo "missing GI/glib/cairo dev deps" >&2; exit 2; }

LUA=$("$LC/_toolchain/build-lua.sh")
if [ "$LUA" = SYSTEM ]; then LI=$(pkg-config --variable=includedir lua5.4); LL="-l lua5.4"
else LI="$LUA"; LL="-L$LUA -llua5.4"; fi

[ -d "$W/src/.git" ] || git clone https://github.com/lgi-devs/lgi "$W/src"
git -C "$W/src" fetch -q origin 2>/dev/null || true
git -C "$W/src" reset -q --hard "$PIN"

build(){ # $1 = tree dir
  make -C "$1/lgi" clean >/dev/null 2>&1 || true
  make -C "$1/lgi" CC=cc LUA_CFLAGS="-I$LI" LUA_LIB="$LL" >/dev/null
  [ -f "$1/lgi/corelgilua51.so" ] || { echo "build failed: $1" >&2; exit 3; }
}

rm -rf "$W/fixed" "$W/vuln"
cp -r "$W/src" "$W/fixed"
cp -r "$W/src" "$W/vuln"

# vuln = revert the issue-122 guard commit
git -C "$W/vuln" revert --no-commit "$FIX"
grep -q 'make the record unusable' "$W/vuln/lgi/record.c" \
  && { echo "revert did not remove the guard" >&2; exit 3; }

build "$W/fixed"
build "$W/vuln"
echo "built: $W/{vuln,fixed}/lgi/corelgilua51.so"
