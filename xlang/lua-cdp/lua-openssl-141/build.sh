#!/usr/bin/env bash
# Build the prerequisites for lua-openssl #141:
#   1. OpenSSL 1.1.1w from source, into ./.work/openssl-install, built
#      SHARED + enable-crypto-mdebug + AddressSanitizer.
#   2. a clone of lua-openssl (+ its submodules) into ./.work/lua-openssl.
# The reference Lua 5.4 comes from the shared toolchain (read-only), not here.
# Idempotent; everything lands under ./.work. Reproduced 2026-08-03.
#
# WHY OpenSSL 1.1.1 from source (the env blocker):
# The vulnerable lua-openssl commit (0017afa, Jun 2018) is written for the
# OpenSSL 1.0.2 / 1.1.x API and does not build cleanly against this box's
# OpenSSL 3.5.  1.1.1w is the newest release its 2018 code compiles against.
# WHY ASan *inside* libcrypto: the second free runs through libcrypto's
# EVP_CIPHER_CTX_reset, which dereferences the freed ctx BEFORE reaching the
# free() ASan intercepts. With an uninstrumented libcrypto that surfaces as a
# bare SEGV; instrumenting libcrypto turns it into a labelled
# heap-use-after-free with full alloc/free/use stacks (the PASS signature).
# WHY crypto-mdebug: openssl.c references CRYPTO_mem_leaks_fp() at teardown,
# which only exists when libcrypto is built with enable-crypto-mdebug.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
W="$HERE/.work"; mkdir -p "$W"

LUA54=$(cd -- "$HERE/../_toolchain/.work/lua54" 2>/dev/null && pwd || true)
[ -n "$LUA54" ] && [ -x "$LUA54/lua-shared" ] || {
  echo "BLOCKED: shared Lua 5.4 toolchain not found at ../_toolchain/.work/lua54/lua-shared" >&2
  exit 2; }
echo "using shared Lua 5.4 at: $LUA54"

SSL_VER=1.1.1w
SSL_TAR="$W/openssl-$SSL_VER.tar.gz"
SSL_SRC="$W/openssl-$SSL_VER"
SSL_PREFIX="$W/openssl-install"
LIBCRYPTO="$SSL_PREFIX/lib/libcrypto.so.1.1"

# Symbol check on a DUMPED FILE, not a pipe: `nm ... | grep -q` under
# `pipefail` can nondeterministically fail (grep -q exits early -> SIGPIPE kills
# nm -> pipeline non-zero) even when the symbol is present. grep on a file is
# deterministic. Needs both crypto-mdebug (CRYPTO_mem_leaks_fp) and ASan.
have_libcrypto() {
  [ -f "$LIBCRYPTO" ] || return 1
  nm -D "$LIBCRYPTO" > "$W/.libcrypto.syms" 2>/dev/null || return 1
  grep -q CRYPTO_mem_leaks_fp "$W/.libcrypto.syms" && grep -q __asan_init "$W/.libcrypto.syms"
}

# ---- 1. OpenSSL 1.1.1w (shared + crypto-mdebug + ASan) --------------------
if have_libcrypto; then
  echo "OpenSSL $SSL_VER (ASan + crypto-mdebug) already built at $SSL_PREFIX"
else
  echo "== fetching + building OpenSSL $SSL_VER (this takes a few minutes) =="
  if [ ! -f "$SSL_TAR" ]; then
    wget -q -O "$SSL_TAR" \
      "https://github.com/openssl/openssl/releases/download/OpenSSL_1_1_1w/openssl-$SSL_VER.tar.gz" \
      || curl -fsSL -o "$SSL_TAR" "https://www.openssl.org/source/openssl-$SSL_VER.tar.gz" \
      || { echo "BLOCKED: cannot download OpenSSL $SSL_VER (no network?)" >&2; exit 2; }
  fi
  rm -rf "$SSL_SRC"; tar -C "$W" -xzf "$SSL_TAR"
  ( cd "$SSL_SRC"
    ./config --prefix="$SSL_PREFIX" --openssldir="$SSL_PREFIX/ssl" \
      shared enable-crypto-mdebug \
      -fsanitize=address -fno-omit-frame-pointer -g -O1 >config.log 2>&1
    make -j"$(nproc)" >build.log 2>&1
    make install_sw >install.log 2>&1 )
  have_libcrypto \
    || { echo "BLOCKED: libcrypto missing CRYPTO_mem_leaks_fp or __asan_init after build" >&2; exit 1; }
  echo "built OpenSSL $SSL_VER at $SSL_PREFIX"
fi

# ---- 2. lua-openssl checkout (+ submodules) -------------------------------
if [ ! -d "$W/lua-openssl/.git" ]; then
  echo "== cloning lua-openssl =="
  git clone --quiet https://github.com/zhaozg/lua-openssl "$W/lua-openssl"
fi
git -C "$W/lua-openssl" submodule update --init --recursive --quiet
echo "lua-openssl at: $W/lua-openssl"

echo "OK: prerequisites ready. Now run ./run.sh"
