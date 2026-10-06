#!/bin/bash
# Cross-compile client-repros for CheriBSD riscv64-purecap.
#
# These cases are plain C over libc, so unlike mmgr-repros they need none of
# PostgreSQL's memory-context sources and none of ports/postgres' CMake. That
# is the point of the corpus: the objects come from malloc, so the build has
# nothing to stand up but the case, the driver and the verbatim upstream
# functions.
#
# Revocation note: these are SPATIAL defects, so unlike the SQLite temporal
# cases they do not depend on a quarantine sweep having run. A capability's
# BOUNDS are checked on every access whether or not anything was revoked.
set -euo pipefail
S=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$S/.." && pwd)
SDK=/home/zephyr/cheriBSD/cheri/output/sdk
SYSROOT=/home/zephyr/cheriBSD/cheri/output/rootfs-riscv64-purecap
CLANG=$SDK/bin/clang
OUT=${OUT:-$R/out-cheri}
mkdir -p "$OUT"

PURECAP=(--target=riscv64-unknown-freebsd --sysroot="$SYSROOT"
         -march=rv64imafdcxcheri -mabi=l64pc128d -mno-relax)

# -static, and the reason is worth recording because the failure is silent at
# build time and only shows up as a refusal to start:
#     ld-elf.so.1: ...: Traditional TLS not supported
# parseOidArray (upstream_pgdump.c) calls isdigit(), which on FreeBSD reaches
# the thread-local _ThreadRuneLocale, and the purecap rtld will not load a
# dynamic binary that uses the traditional TLS model. -ftls-model=local-exec
# does not help -- it fails at link time against libc.so.7's own definition.
# Static linking keeps libc's real allocator, which is all these cases need:
# they are SPATIAL defects, so what matters is that a malloc'd object carries
# its own capability bounds, not that any quarantine has swept.
# The SQLite arm already links one case (spellfixoom) this way.
STATIC=(-static)

built=0; failed=0
for d in "$R"/[0-9][0-9]_*/; do
  [ -f "$d/case.c" ] || continue
  tag=$(basename "$d")
  # No -g, and stripped. Static purecap with debug info is 8.6 MB a case; the
  # same binary stripped is 0.69 MB, and these have to cross an emulated
  # network to the guest. The host-ASan build (build-host.sh) keeps -g, which
  # is where a symbolised trace is actually wanted.
  if "$CLANG" "${PURECAP[@]}" "${STATIC[@]}" -O0 -I"$S" \
       "$d/case.c" "$S/driver.c" "$S"/upstream_*.c -o "$OUT/$tag" 2>"$OUT/$tag.err"; then
    "$SDK/bin/llvm-strip" "$OUT/$tag" 2>/dev/null || true
    printf '  %-52s OK  %s\n' "$tag" "$(stat -c %s "$OUT/$tag") bytes"; built=$((built+1))
  else
    printf '  %-52s FAIL\n' "$tag"; head -6 "$OUT/$tag.err" | sed 's/^/      /'; failed=$((failed+1))
  fi
done
echo "built=$built failed=$failed  ($OUT)"
[ "$failed" = 0 ]
