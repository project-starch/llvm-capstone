#!/usr/bin/env bash
# Build lgi's C core against the shared toolchain Lua 5.4, in two trees:
#   fixed/  = pinned HEAD (7a2276f) as-is
#   vuln/   = HEAD with the issue-65 fix (commit 358371fd, "Fix marshalling array
#             2c with transfer != none") REVERTED for the C-array field-set path.
#
# The fix introduced array_detach() so a struct field that TAKES OWNERSHIP of a
# marshalled GArray gets only its container detached (data transferred to C),
# instead of the guard g_array_unref-ing the whole GArray. Pre-fix, the guard
# freed the data too, leaving the C struct field dangling. We reconstruct that by
# forcing the GI_ARRAY_TYPE_C/ARRAY guard back to g_array_unref. (A plain
# `git revert 358371fd` conflicts because the surrounding code moved since 2013;
# and the historical pre-fix tree predates Lua 5.4 support and won't build. This
# one-token edit is the faithful minimal reconstruction.)
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); LC=$(cd "$HERE/.."; pwd); W="$HERE/.work"; mkdir -p "$W"
PIN=7a2276f

command -v valgrind >/dev/null || { echo "need valgrind" >&2; exit 2; }
pkg-config --exists 'gobject-introspection-1.0 gmodule-2.0 libffi gio-2.0' \
  || { echo "missing GI/glib/gio dev deps" >&2; exit 2; }

LUA=$("$LC/_toolchain/build-lua.sh")
if [ "$LUA" = SYSTEM ]; then LI=$(pkg-config --variable=includedir lua5.4); LL="-l lua5.4"
else LI="$LUA"; LL="-L$LUA -llua5.4"; fi

[ -d "$W/src/.git" ] || git clone https://github.com/lgi-devs/lgi "$W/src"
git -C "$W/src" fetch -q origin 2>/dev/null || true
git -C "$W/src" reset -q --hard "$PIN"

build(){ make -C "$1/lgi" clean >/dev/null 2>&1 || true
  make -C "$1/lgi" CC=cc LUA_CFLAGS="-I$LI" LUA_LIB="$LL" >/dev/null
  [ -f "$1/lgi/corelgilua51.so" ] || { echo "build failed: $1" >&2; exit 3; }; }

rm -rf "$W/fixed" "$W/vuln"
cp -r "$W/src" "$W/fixed"
cp -r "$W/src" "$W/vuln"

# vuln = re-open issue 65: field-set C-array guard frees the data segment again.
perl -0pi -e 's/\? array_detach : g_array_unref/? g_array_unref \/*lgi-65 VULN*\/ : g_array_unref/' \
  "$W/vuln/lgi/marshal.c"
grep -q 'lgi-65 VULN' "$W/vuln/lgi/marshal.c" || { echo "vuln edit did not apply" >&2; exit 3; }

build "$W/fixed"
build "$W/vuln"
echo "built: $W/{vuln,fixed}/lgi/corelgilua51.so"
