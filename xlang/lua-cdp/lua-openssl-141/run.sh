#!/usr/bin/env bash
# CHECK (not just a runner) for lua-openssl #141, by DIFFERENTIAL:
#   VULNERABLE 0017afa -> ASan heap-use-after-free of the EVP_CIPHER_CTX, freed
#                         by c:close() and re-freed by the userdata __gc (GCTM).
#   FIXED      a436c36 -> clean run (prints NO-DOUBLE-FREE), no ASan report.
# PASS only if BOTH hold. That differential is what makes this a verified
# reproduction of the cross-domain double-free, not an incidental crash.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
W="$HERE/.work"
SSL="$W/openssl-install"
REPO="$W/lua-openssl"
LUA54=$(cd -- "$HERE/../_toolchain/.work/lua54" 2>/dev/null && pwd || true)
ASAN=$(cc -print-file-name=libasan.so)

VULN=0017afa23dcbbab91a307cd6ae07f60f7427e02f     # parent of the #141 fix
FIX=a436c363aa6f963c48ce3c103e16b941ebbfea45      # "fix issue #141": FREE_OBJECT nulls the ptr

[ -x "$LUA54/lua-shared" ]              || { echo "run ./build.sh first (no shared lua)" >&2; exit 2; }
[ -f "$SSL/lib/libcrypto.so.1.1" ]      || { echo "run ./build.sh first (no OpenSSL 1.1.1)" >&2; exit 2; }
[ -d "$REPO/.git" ]                     || { echo "run ./build.sh first (no lua-openssl)" >&2; exit 2; }

# gcc 15 turns several old-C warnings into errors; -DOPENSSL_NO_SM2 drops the
# one module whose public header (openssl/sm2.h) 1.1.1 does not ship. Neither
# touches the cipher double-free path.
PERMISSIVE="-Wno-error=incompatible-pointer-types -Wno-error=int-conversion \
-Wno-error=implicit-function-declaration -Wno-error=discarded-qualifiers -Wno-deprecated-declarations"

build_at() { # $1 = commit -> builds an ASan openssl.so in $REPO
  git -C "$REPO" checkout -q "$1" || return 1
  git -C "$REPO" submodule update --init --recursive --quiet
  make -C "$REPO" clean >/dev/null 2>&1
  make -C "$REPO" CC=cc LUAV=5.4 \
    TARGET_FLAGS="-g -O0 -fsanitize=address -fno-omit-frame-pointer -DOPENSSL_NO_SM2 $PERMISSIVE" \
    OPENSSL_CFLAGS="-I$SSL/include" OPENSSL_LIBS="-L$SSL/lib -lssl -lcrypto" \
    LUA_CFLAGS="-I$LUA54" LUA_LIBS="-L$LUA54 -llua5.4 -fsanitize=address" \
    >"$W/make.$1.log" 2>&1 \
    || { echo "build failed for $1 (see $W/make.$1.log)" >&2; return 1; }
  [ -f "$REPO/openssl.so" ] || { echo "no openssl.so built for $1" >&2; return 1; }
}

run_trigger() { # -> stdout+stderr of the trigger under ASan
  LD_PRELOAD="$ASAN" LD_LIBRARY_PATH="$SSL/lib:$LUA54" LUA_CPATH="$REPO/?.so;;" \
    ASAN_OPTIONS="detect_leaks=0:abort_on_error=0" \
    "$LUA54/lua-shared" "$HERE/trigger.lua" 2>&1
}

# Outputs go to FILES (grepped below): `... | grep -qa` under pipefail can
# nondeterministically fail on an early match (SIGPIPE); grep on a file cannot.
VOUT="$W/vuln.out"; FOUT="$W/fix.out"
echo "== vulnerable ($VULN) =="; build_at "$VULN" || exit 1; run_trigger >"$VOUT" 2>&1; head -6 "$VOUT"
echo "== fixed ($FIX) ==";      build_at "$FIX"  || exit 1; run_trigger >"$FOUT" 2>&1; tail -3 "$FOUT"
git -C "$REPO" checkout -q "$VULN"   # leave the case on the vulnerable tree

# Vulnerable: heap-use-after-free on the EVP_CIPHER_CTX, with the SECOND free
# reached from the GC finalizer (openssl_cipher_ctx_free @ cipher.c:551 <- GCTM).
vuln_uaf=0
if grep -qa 'heap-use-after-free' "$VOUT" \
   && grep -qa 'openssl_cipher_ctx_free src/cipher.c:551' "$VOUT" \
   && grep -qa 'GCTM' "$VOUT" \
   && grep -qa 'EVP_CIPHER_CTX_free' "$VOUT"; then vuln_uaf=1; fi

# Fixed: no ASan report AND the trigger ran to completion.
fix_clean=0
if grep -qa 'NO-DOUBLE-FREE' "$FOUT" && ! grep -qa 'AddressSanitizer' "$FOUT"; then fix_clean=1; fi

echo "--- verdict: vuln_uaf=$vuln_uaf fix_clean=$fix_clean ---"
if [ "$vuln_uaf" = 1 ] && [ "$fix_clean" = 1 ]; then
  echo "PASS: lua-openssl #141 reproduced (UAF/double-free of EVP_CIPHER_CTX on $VULN via __gc; clean on $FIX)"
  exit 0
fi
echo "FAIL: differential not satisfied (vuln_uaf=$vuln_uaf fix_clean=$fix_clean)" >&2
exit 1
