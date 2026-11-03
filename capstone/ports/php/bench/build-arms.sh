#!/usr/bin/env bash
# Build the PHP 5.0.0 CLI twice from one prepared tree, for the capprof comparison of PHP's own
# benchmark scripts (Zend/bench.php, Zend/micro_bench.php; see profile-php.sh):
#
#   capstone  the virtual-profile SDK's capstone-cc (CC_CAP), the stock (unpatched) zend_alloc.c,
#             every emalloc block from the SDK's virtual mallocng
#   plain     a plain rv64 driver (CC_PLAIN: llvm-capstone's RISCV target, pristine musl 1.2.5,
#             static, no PIE), the profiling kit's plain arm
#
# usage: build-arms.sh TREE OUT
#   TREE  a tree prepared by ../app/prepare-sources.sh (configure --disable-all, the tree patches)
#   OUT   new or existing directory; OUT/<arm>/php is the image, OUT/<arm>/build-record.txt its record
# env:  CC_CAP (default ~/prof-dev/cap3/sdk/capstone-cc), CC_PLAIN (default ~/prof-dev/rv/rv-cc),
#       OBJCOPY (default ~/prof-dev/cap3/build/bin/llvm-objcopy), OPT (default -O2), JOBS (default 8),
#       ARMS (default "capstone plain")
#
# Same translation units, overlay and flags as ../app/build-virtual.sh realphp with
# PHP_REALPHP_ALLOC=stock, except the optimization level (OPT, as the SPEC kit's -O2 in both arms,
# not the port's fault-attribution -O0) and the stack-object flag (the SDK default here, as the
# SPEC census builds; the port's comparison build passes -capstone-shrink-stack=false). Both arms
# get the same flags; the only per-arm difference is the compiler driver (and with it the libc).
set -euo pipefail
TREE=$(cd "${1:?usage: build-arms.sh TREE OUT}" && pwd -P); OUT=${2:?usage: build-arms.sh TREE OUT}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd -P)
CC_CAP=${CC_CAP:-$HOME/prof-dev/cap3/sdk/capstone-cc}
CC_PLAIN=${CC_PLAIN:-$HOME/prof-dev/rv/rv-cc}
OBJCOPY=${OBJCOPY:-$HOME/prof-dev/cap3/build/bin/llvm-objcopy}
OPT=${OPT:--O2} JOBS=${JOBS:-8}
export SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-1700000000}   # php -v's __DATE__/__TIME__ (see build-virtual.sh)
[ -f "$TREE/Makefile" ] && [ -f "$TREE/main/php_config.h" ] && [ -f "$TREE/PROVENANCE.txt" ] \
  || { echo "build-arms: $TREE is not a tree prepared by prepare-sources.sh" >&2; exit 2; }

for ARM in ${ARMS:-capstone plain}; do
  case $ARM in capstone) CC=$CC_CAP ;; plain) CC=$CC_PLAIN ;; *) echo "build-arms: arm $ARM?" >&2; exit 2 ;; esac
  B=$OUT/$ARM; rm -rf "$B"; mkdir -p "$B/obj" "$B/log"; T=$B/tree
  cp -a "$TREE" "$T"
  # The two configure answers glibc gives and musl (both arms' libc) does not: build-virtual.sh's delta.
  sed -i -e 's|^#define HAVE_RES_NMKQUERY 1$|#define HAVE_RES_NMKQUERY 0|' \
         -e 's|^#define HAVE_RES_NSEND 1$|#define HAVE_RES_NSEND 0|' "$T/main/php_config.h"
  n=$(diff "$TREE/main/php_config.h" "$T/main/php_config.h" | grep -c '^<' || true)
  [ "$n" -eq 8 ] || { echo "build-arms: php_config.h: $n defines corrected, expected 8" >&2; exit 2; }
  { sed -n '/^PHP_GLOBAL_OBJS *=/p' "$T/Makefile"; sed -n '/^PHP_CLI_OBJS *=/p' "$T/Makefile"; } \
    | grep -o '[A-Za-z_0-9/]*\.lo' | sed 's/\.lo$/.c/' | LC_ALL=C sort -u > "$B/tus.txt"
  [ "$(wc -l < "$B/tus.txt")" -ge 140 ] || { echo "build-arms: short TU list" >&2; exit 2; }
  CF=("$OPT" -std=gnu89 -fno-common -gline-tables-only -D__off64_t=off_t
      -Wno-implicit-int -Wno-implicit-function-declaration -Wno-format-security
      -I"$T" -I"$T/Zend" -I"$T/main" -I"$T/TSRM" -I"$T/ext" -I"$T/ext/standard" -I"$T/regex" -I"$T/sapi/cli")
  printf '%s\n' "${CF[@]}" > "$B/cflags.txt"
  echo "[$ARM] compiling $(wc -l < "$B/tus.txt") TUs with $CC $OPT (JOBS=$JOBS)"
  export CC B T
  # One compile per TU, JOBS at a time; the flags are read back from cflags.txt (one per line).
  # shellcheck disable=SC2016
  xargs -P "$JOBS" -n 1 bash -c 'mapfile -t cf < "$B/cflags.txt"; n=$(echo "$1" | tr / _)
    "$CC" "${cf[@]}" -c "$T/$1" -o "$B/obj/${n%.c}.o" 2> "$B/log/$n.log" || echo "FAILED $1"' _ \
    < "$B/tus.txt" | tee "$B/failed.txt"
  [ ! -s "$B/failed.txt" ] || { echo "[$ARM] $(wc -l < "$B/failed.txt") TUs failed; logs in $B/log" >&2; exit 1; }
  # flex's dead bare yytext in both scanners (build-virtual.sh explains); -fno-common in both arms.
  "$OBJCOPY" --localize-symbol=yytext "$B/obj/Zend_zend_ini_scanner.o"
  mapfile -t OBJS < <(sed 's|/|_|g; s|\.c$|.o|; s|^|'"$B"'/obj/|' "$B/tus.txt")
  "$CC" "${CF[@]}" "${OBJS[@]}" -lm -o "$B/php" 2> "$B/log/link.log" \
    || { echo "[$ARM] link failed:" >&2; grep -i error "$B/log/link.log" | head >&2; exit 1; }
  { echo "arm $ARM cc $CC opt $OPT tree $TREE"; sed -n '1,7p' "$TREE/PROVENANCE.txt"
    case $ARM in capstone) echo "sdk.json $(sha256sum "$(dirname "$CC")/sdk.json" | cut -c1-64)" ;; *) cat "$CC" | tail -1 ;; esac
    "$(dirname "$OBJCOPY")/llvm-readelf" -h "$B/php" | grep -E 'Machine|Type'
    sha256sum "$B/php"; } > "$B/build-record.txt"
  cat "$B/build-record.txt"
done
