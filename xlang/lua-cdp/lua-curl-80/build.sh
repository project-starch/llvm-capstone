#!/usr/bin/env bash
# Build Lua-cURLv3 (lcurl) at the vulnerable (b2e9474) and fixed (56b4d05)
# commits, with AddressSanitizer, against:
#   - the shared reference Lua 5.4 toolchain (../_toolchain, read-only), and
#   - the system libcurl 8.18 (pkg-config libcurl; NOT rebuilt).
# Idempotent; everything lands under ./.work. Reproduced 2026-08-04.
#
# WHY the SSL_CACERT one-line patch (the only env-blocker):
#   The 2016 lcurl source maps every CURLE_* code to a name via a switch in
#   src/lcerr_easy.h.  In libcurl >= 8, CURLE_SSL_CACERT is a *deprecated alias*
#   with the SAME numeric value as CURLE_PEER_FAILED_VERIFICATION, so the two
#   `case` labels collide -> "duplicate case value" (a hard C error, not a
#   warning).  We drop the single aliased line.  It is an error-string mapping
#   ONLY -- it is not on the easy/multi lifetime path this case exercises.
#   (Analogous to lua-openssl #141's -DOPENSSL_NO_SM2 build shim.)
#
# NOTE (see CASE.md): the pure *native* "easy handle freed while a multi still
#   holds its CURL*" UAF does NOT reproduce on libcurl 8.18 -- libcurl's
#   Curl_close() auto-calls curl_multi_remove_handle() (since <= 7.50), so it
#   self-protects.  The reproducible bug here is lcurl's OWN dangling pointer
#   (easy->multi -> a collected multi userdata); it is libcurl-version
#   independent and observed by ASan inside the instrumented lcurl.so.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LC=$(cd -- "$HERE/.."; pwd)
W="$HERE/.work"; mkdir -p "$W"

VULN=b2e9474c6bf975a0f1813a387e2c0ec21d3064f3   # parent of the fix: multi::close leaves easy->multi dangling
FIX=56b4d05c17b406d790721b1921314a5da7da2c58    # "Fix. Cleanup easy references when calls multi::close"

# ---- reference Lua 5.4 from the shared toolchain (read-only) ---------------
LUA54=$(cd -- "$LC/_toolchain/.work/lua54" 2>/dev/null && pwd || true)
[ -n "$LUA54" ] && [ -x "$LUA54/lua-shared" ] || {
  echo "BLOCKED: shared Lua 5.4 toolchain not found (run ../_toolchain/build-lua.sh)" >&2
  exit 2; }
echo "using shared Lua 5.4 at: $LUA54"

# ---- system libcurl (do NOT rebuild) --------------------------------------
pkg-config --exists libcurl || { echo "BLOCKED: libcurl not found via pkg-config" >&2; exit 2; }
echo "using system libcurl $(pkg-config --modversion libcurl)"
CURL_CFLAGS=$(pkg-config --cflags libcurl)
CURL_LIBS=$(pkg-config --libs libcurl)

# ---- clone lcurl ----------------------------------------------------------
if [ ! -d "$W/lcurl/.git" ]; then
  echo "== cloning Lua-cURLv3 =="
  git clone --quiet https://github.com/Lua-cURL/Lua-cURLv3 "$W/lcurl"
fi

build_at() { # $1 = commit, $2 = output subdir
  local commit="$1" out="$W/$2"
  mkdir -p "$out"
  git -C "$W/lcurl" checkout -q "$commit"
  # single-line, bug-unrelated build shim (see header)
  sed -i -E 's/^\s*ERR_ENTRY\s*\(\s*SSL_CACERT\s*\).*/\/* SSL_CACERT alias dropped for libcurl>=8 *\//' \
    "$W/lcurl/src/lcerr_easy.h"
  cc -shared -fPIC -g -O0 -fsanitize=address -fno-omit-frame-pointer -w \
     -I"$LUA54" $CURL_CFLAGS \
     "$W"/lcurl/src/*.c \
     -o "$out/lcurl.so" \
     -L"$LUA54" -llua5.4 $CURL_LIBS \
     2>"$W/make.$2.log" \
    || { echo "build failed for $commit (see $W/make.$2.log)" >&2; tail -5 "$W/make.$2.log" >&2; return 1; }
  git -C "$W/lcurl" checkout -q -- src/lcerr_easy.h    # restore for the next checkout
  [ -f "$out/lcurl.so" ] || { echo "no lcurl.so for $commit" >&2; return 1; }
  echo "built $2 lcurl.so @ ${commit:0:7}"
}

build_at "$VULN" vuln
build_at "$FIX"  fixed
git -C "$W/lcurl" checkout -q "$VULN"   # leave the tree on the vulnerable commit

echo "OK: prerequisites ready ($W/{vuln,fixed}/lcurl.so). Now run ./run.sh"
