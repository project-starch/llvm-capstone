#!/bin/bash
# probe-image.sh NAME [REPO_RELATIVE_OR_PHP_SOURCE=REPLACEMENT ...]   (run through ns-build.sh)
# A probe image: the measured image's own objects (the byte-identical rebuild in php-diag), with the named
# translation units recompiled from a replacement source and everything relinked in build-virtual.sh's order.
# Each replacement is bound over the original file's path for this process tree alone, so the compile commands
# are build-virtual.sh's, word for word, and no file outside this work directory changes.
#   key `nested:zend-cap-mode.c`  -> capstone/ports/php/zend-alloc/sublet/zend-cap-mode.c (object sup_zend-cap-mode.o)
#   key `php:Zend/zend_API.c`     -> the PHP tree's file (object Zend_zend_API.o)
set -euo pipefail
name=$1; shift
R=$CAPSTONE_REPO_ROOT; OUT=$PHP_VIRTUAL_OUT; P=$PHP_SRC
[ -e "$OUT/.fix-php-sentinel" ] || { echo "not inside the bound namespace" >&2; exit 3; }
export SOURCE_DATE_EPOCH=1700000000
CC="$OUT/sdk/capstone-cc"; B="$OUT/realphp/nested"; T="$B/tree"; NESTED="$R/capstone/ports/php/zend-alloc/sublet"
D="$W/probes/build-$name"; rm -rf "$D"; mkdir -p "$D/obj"
cp "$B"/obj/*.o "$D/obj/"
CF=(-O0 -std=gnu89 -fno-common -gline-tables-only -D__off64_t=off_t
    -Wno-implicit-int -Wno-implicit-function-declaration -Wno-format-security
    -I"$T" -I"$T/Zend" -I"$T/main" -I"$T/TSRM" -I"$T/ext" -I"$T/ext/standard" -I"$T/regex" -I"$T/sapi/cli"
    -DZEND_CAP_SUBLET -I"$NESTED" -I"$R/capstone" -I"$R/capstone/runtime/include"
    -mllvm -capstone-shrink-stack=false)
for word in "$@"; do
  key=${word%%=*}; repl=${word#*=}
  case "$key" in
    nested:*) src="$NESTED/${key#nested:}"; obj="$D/obj/sup_$(basename "${key#nested:}" .c).o"; comp="$src" ;;
    php:*)    f=${key#php:}; src="$P/$f"; obj="$D/obj/$(echo "$f" | tr '/' '_' | sed 's/\.c$/.o/')"; comp="$T/$f" ;;
    *) echo "unknown key $key" >&2; exit 2 ;;
  esac
  [ -f "$src" ] && [ -f "$obj" ] || { echo "no $src or $obj" >&2; exit 2; }
  mount --bind "$repl" "$src"
  cmp -s "$repl" "$comp" || { echo "bind of $repl over $src did not take" >&2; exit 3; }
  "$CC" "${CF[@]}" -c "$comp" -o "$obj"
  echo "  recompiled $key from $repl"
done
OBJS=()
while IFS= read -r f; do OBJS+=("$D/obj/$(echo "$f" | tr '/' '_' | sed 's/\.c$/.o/')"); done < "$B/tus.txt"
OBJS+=("$D/obj/sup_zend-cap-mode.o")
"$CC" "${CF[@]}" "${OBJS[@]}" -lm -o "$D/php-real-nested.dom"
sha256sum "$D/php-real-nested.dom"
rm -rf "$D/obj"
