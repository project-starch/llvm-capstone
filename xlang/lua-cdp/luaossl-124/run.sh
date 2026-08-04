#!/usr/bin/env bash
# CHECK (not just a runner) for luaossl #124, by DIFFERENTIAL:
#   VULNERABLE 5be1b44 -> ASan heap-use-after-free of the X509_STORE: freed once
#                         by the Lua store userdata __gc (xs__gc -> X509_STORE_free)
#                         and again by SSL_CTX_free (sx__gc), both driven by the
#                         Lua GC (GCTM). Two ownership domains, one C object.
#   FIXED      1ae7073 -> clean run (prints NO-DOUBLE-FREE), no ASan report.
# PASS only if BOTH hold. That differential is what makes this a verified
# reproduction of the cross-domain double-free, not an incidental crash.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
W="$HERE/.work"
SSL="$W/openssl-install"
REPO="$W/luaossl"
LUA54=$(cd -- "$HERE/../_toolchain/.work/lua54" 2>/dev/null && pwd || true)
ASAN=$(cc -print-file-name=libasan.so)

VULN=5be1b44a6a60f32c660cc4ee09d60e676cd8c81a     # parent of the #124 fix
FIX=1ae707300bf99805bd93744020c60cf60cdc2294      # "Fix SSL_CTX_set1_cert_store refcounting issues. Closes #124"

[ -x "$LUA54/lua-shared" ]              || { echo "run ./build.sh first (no shared lua)" >&2; exit 2; }
[ -f "$SSL/lib/libcrypto.so.1.1" ]      || { echo "run ./build.sh first (no OpenSSL 1.1.1)" >&2; exit 2; }
[ -d "$REPO/.git" ]                     || { echo "run ./build.sh first (no luaossl)" >&2; exit 2; }

# gcc 15 turns several old-C warnings into errors; luaossl's 2018 code trips a
# couple. Neither touches the X509_STORE ownership path.
PERMISSIVE="-Wno-error=incompatible-pointer-types -Wno-error=int-conversion \
-Wno-error=implicit-function-declaration -Wno-error=discarded-qualifiers -Wno-deprecated-declarations"

# Build ONE _openssl.so (all luaopen__openssl_* entry points) at commit $1 into
# $2, and lay out the require() tree: the two .lua wrappers the reproducer needs,
# plus _openssl.so which Lua's all-in-one searcher uses for _openssl.ssl.context
# and _openssl.x509.store.
build_at() { # $1 = commit, $2 = out dir
	local commit="$1" out="$2"
	git -C "$REPO" checkout -q "$commit" || return 1
	git -C "$REPO" submodule update --init --recursive --quiet
	cp "$REPO/config.h.guess" "$REPO/src/config.h"
	rm -rf "$out"; mkdir -p "$out/openssl/ssl" "$out/openssl/x509"
	cc -c "$REPO/src/openssl.c" -o "$out/openssl.o" \
		-I"$LUA54" -I"$REPO/src" -I"$REPO" -I"$SSL/include" \
		-DHAVE_CONFIG_H -DCOMPAT53_PREFIX=luaossl -D_GNU_SOURCE -D_REENTRANT -D_THREAD_SAFE \
		-fsanitize=address -fno-omit-frame-pointer -g -O0 -std=gnu99 -fPIC \
		$PERMISSIVE >"$W/cc.$commit.log" 2>&1 \
		|| { echo "compile failed for $commit (see $W/cc.$commit.log)" >&2; return 1; }
	cc -shared -o "$out/_openssl.so" "$out/openssl.o" \
		-fsanitize=address -L"$SSL/lib" -lssl -lcrypto -lpthread -ldl -lrt -lm \
		>"$W/link.$commit.log" 2>&1 \
		|| { echo "link failed for $commit (see $W/link.$commit.log)" >&2; return 1; }
	cp "$REPO/src/openssl.ssl.context.lua" "$out/openssl/ssl/context.lua"
	cp "$REPO/src/openssl.x509.store.lua"  "$out/openssl/x509/store.lua"
	[ -f "$out/_openssl.so" ] || { echo "no _openssl.so built for $commit" >&2; return 1; }
}

run_trigger() { # $1 = module dir -> stdout+stderr of the trigger under ASan
	LD_PRELOAD="$ASAN" LD_LIBRARY_PATH="$SSL/lib:$LUA54" \
		LUA_CPATH="$1/?.so;;" LUA_PATH="$1/?.lua;;" \
		ASAN_OPTIONS="detect_leaks=0:abort_on_error=0" \
		"$LUA54/lua-shared" "$HERE/trigger.lua" 2>&1
}

# Outputs go to FILES (grepped below): `... | grep -qa` under pipefail can
# nondeterministically fail on an early match (SIGPIPE); grep on a file cannot.
VOUT="$W/vuln.out"; FOUT="$W/fix.out"
echo "== vulnerable ($VULN) =="; build_at "$VULN" "$W/mod_vuln" || exit 1; run_trigger "$W/mod_vuln" >"$VOUT" 2>&1; head -6 "$VOUT"
echo "== fixed ($FIX) ==";      build_at "$FIX"  "$W/mod_fix"  || exit 1; run_trigger "$W/mod_fix"  >"$FOUT" 2>&1; tail -3 "$FOUT"
git -C "$REPO" checkout -q "$VULN"   # leave the case on the vulnerable tree

# Vulnerable: heap-use-after-free on the X509_STORE, with BOTH owners present --
# the Lua store userdata __gc (xs__gc -> X509_STORE_free, the FIRST free) and the
# SSL_CTX (sx__gc -> SSL_CTX_free, the SECOND free), both under the Lua GC (GCTM).
vuln_uaf=0
if grep -qa 'heap-use-after-free' "$VOUT" \
   && grep -qa 'X509_STORE_free' "$VOUT" \
   && grep -qa 'in xs__gc' "$VOUT" \
   && grep -qa 'in sx__gc' "$VOUT" \
   && grep -qa 'SSL_CTX_free' "$VOUT" \
   && grep -qa 'GCTM' "$VOUT"; then vuln_uaf=1; fi

# Fixed: no ASan report AND the trigger ran to completion.
fix_clean=0
if grep -qa 'NO-DOUBLE-FREE' "$FOUT" && ! grep -qa 'AddressSanitizer' "$FOUT"; then fix_clean=1; fi

echo "--- verdict: vuln_uaf=$vuln_uaf fix_clean=$fix_clean ---"
if [ "$vuln_uaf" = 1 ] && [ "$fix_clean" = 1 ]; then
  echo "PASS: luaossl #124 reproduced (cross-domain double-free of X509_STORE on $VULN:"
  echo "      xs__gc[Lua store userdata] + SSL_CTX_free[sx__gc], both via GCTM; clean on $FIX)"
  exit 0
fi
echo "FAIL: differential not satisfied (vuln_uaf=$vuln_uaf fix_clean=$fix_clean)" >&2
exit 1
