#!/usr/bin/env bash
# Build the prerequisites for luaossl #124 (cross-domain double-free of an
# X509_STORE owned by BOTH a Lua-GC store userdata and an SSL_CTX):
#   1. OpenSSL 1.1.1w from source, into ./.work/openssl-install, built
#      SHARED + AddressSanitizer.
#   2. a clone of luaossl into ./.work/luaossl.
# The reference Lua 5.4 comes from the shared toolchain (read-only), not here.
# Idempotent; everything lands under ./.work.
#
# WHY OpenSSL 1.1.1w from source (two independent reasons):
#  (a) The vulnerable luaossl commit (5be1b44, Mar 2018) is written for the
#      OpenSSL 1.0.2 / 1.1.x API. Against this box's OpenSSL 3.5 it does NOT
#      compile (SHLIB_VERSION_HISTORY, RSA_SSLV23_PADDING, ... were removed in
#      3.0). 1.1.1w is the newest release its 2018 code builds against.
#  (b) The #124 bug lives in luaossl's OWN compat shim for SSL_CTX_set1_cert_store.
#      At 5be1b44 luaossl never detects that OpenSSL provides set1_cert_store
#      natively (config.h.guess only probes C attributes; HAVE_SSL_CTX_set1_cert_store
#      is left undefined -> HAVE_SSL_CTX_SET1_CERT_STORE == 0), so it ALWAYS uses
#      its shim. On any OpenSSL >= 1.1.0 (X509_STORE opaque -> HAVE_X509_STORE_REFERENCES
#      == 0) that shim expands to the OWNERSHIP-taking SSL_CTX_set_cert_store()
#      (set0 semantics: no refcount bump) -> the store ends up owned twice.
#      1.1.1w satisfies ">= 1.1.0", so the buggy path is compiled and reached.
# WHY ASan *inside* libcrypto: the second free runs through libcrypto's
# X509_STORE_free, which dereferences the freed store (reads its refcount)
# BEFORE reaching the free() ASan intercepts. With an uninstrumented libcrypto
# that read is on poisoned-but-not-owned memory ASan can't see, so detection is
# nondeterministic; instrumenting libcrypto turns it into a labelled
# heap-use-after-free with full alloc/free/use stacks (the PASS signature).
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

# Symbol check on a DUMPED FILE, not a pipe: `nm ... | grep -q` under pipefail
# can nondeterministically fail (grep -q exits early -> SIGPIPE kills nm ->
# pipeline non-zero) even when the symbol is present. grep on a file is
# deterministic. Needs ASan (__asan_init) and the set0 primitive we exercise.
have_libcrypto() {
  [ -f "$LIBCRYPTO" ] || return 1
  nm -D "$LIBCRYPTO" > "$W/.libcrypto.syms" 2>/dev/null || return 1
  grep -q __asan_init "$W/.libcrypto.syms" && grep -q X509_STORE_free "$W/.libcrypto.syms"
}

# ---- 1. OpenSSL 1.1.1w (shared + ASan) ------------------------------------
if have_libcrypto; then
  echo "OpenSSL $SSL_VER (ASan) already built at $SSL_PREFIX"
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
      shared \
      -fsanitize=address -fno-omit-frame-pointer -g -O1 >config.log 2>&1
    make -j"$(nproc)" >build.log 2>&1
    make install_sw >install.log 2>&1 )
  have_libcrypto \
    || { echo "BLOCKED: libcrypto missing __asan_init or X509_STORE_free after build" >&2; exit 1; }
  echo "built OpenSSL $SSL_VER at $SSL_PREFIX"
fi

# ---- 2. luaossl checkout --------------------------------------------------
if [ ! -d "$W/luaossl/.git" ]; then
  echo "== cloning luaossl =="
  git clone --quiet https://github.com/wahern/luaossl "$W/luaossl"
fi
git -C "$W/luaossl" submodule update --init --recursive --quiet
echo "luaossl at: $W/luaossl"

echo "OK: prerequisites ready. Now run ./run.sh"
