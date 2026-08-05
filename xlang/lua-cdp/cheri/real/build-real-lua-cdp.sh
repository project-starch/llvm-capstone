#!/usr/bin/env bash
# Build a real-Lua-purecap CDP reproduction for CheriBSD -- the CHERI half of the
# fair real-Lua comparison (the Capstone half is the LUA_CDP_* domains in
# ../../capstone-lua/lua_domain.c). On CheriBSD real Lua is an ordinary purecap
# program, so a reproduction is just: real Lua + a minimal native-object stub + a
# Lua userdata/__gc trigger -- the same shape as the Capstone domain, no
# freestanding gymnastics. Under CheriBSD revocation the stale cross-domain access
# faults (CAUGHT); with revocation off it completes (MISS).
#
#   ./build-real-lua-cdp.sh [reproduction.c]     # default: cdp_x509.c (luaossl #124)
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CHERI_ROOT=${CHERI_ROOT:-$HOME/cheri}
SDK=${SDK:-$CHERI_ROOT/output/sdk}
ROOTFS=${ROOTFS:-$CHERI_ROOT/rootfs-purecap}
LUA=${LUA_SRC:-$HERE/../../_toolchain/.work/lua54}
OUT=${OUT:-$CHERI_ROOT/lua-cdp-real}
SRC=${1:-$HERE/cdp_x509.c}
for p in "$SDK/bin/clang" "$ROOTFS/usr/include" "$LUA/lua.h"; do
  [ -e "$p" ] || { echo "MISSING: $p" >&2; exit 2; }
done
mkdir -p "$OUT"
BUILTIN=$(echo "$SDK"/lib/clang/*/include | tr ' ' '\n' | head -1)
# -nostdinc + explicit -isystem: the default header search leaks to host glibc
# (bits/floatn.h -> unsupported __float128); force sysroot-only. The prelude no-ops
# the shared Lua source's diagnostic probes (Capstone defines them via
# capstone_lua_libc.h), keeping the interpreter source byte-identical across platforms.
flags=(--target=riscv64-unknown-freebsd -march=rv64gcxcheri -mabi=l64pc128d
       --sysroot="$ROOTFS" -mno-relax -O0 -w -ftls-model=initial-exec
       -include "$HERE/cheri-lua-prelude.h" -nostdinc -isystem "$BUILTIN"
       -isystem "$ROOTFS/usr/include" -I"$LUA")
luasrcs=(); for f in "$LUA"/*.c; do case "$(basename "$f")" in luac.c|onelua.c|lua.c) ;; *) luasrcs+=("$f");; esac; done
name=$(basename "$SRC" .c)
"$SDK/bin/clang" "${flags[@]}" -o "$OUT/$name" "$SRC" "${luasrcs[@]}" -lm
echo "built $OUT/$name  (real Lua 5.4.7 purecap + $name)"
