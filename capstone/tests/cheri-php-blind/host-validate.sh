#!/usr/bin/env bash
# Validate the allocator-pattern rows on the HOST with ASAN, before spending any
# CHERI time on them.
#
# The rows claim to model the corpus's 31 ASAN-blind defects. That claim is
# testable with the detector we already understand: with PHP's size-class cache
# and REAL_SIZE rounding in place ASAN must be BLIND to them, and with the two
# corpus patches applied (-DPHP_PATCHED_ALLOC) it must CATCH them. Reproducing
# that flip here is what licenses using these rows as the CHERI baseline.
#
# This script does not touch CHERI and needs no vehicle.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# gcc + libasan is what the corpus itself pins (container/Containerfile: gcc 13 / libasan8),
# and the classifier matches ASAN's own wording, so use the same detector family here.
CC=${CC:-gcc}
OUT=${OUT:-$SCRIPT_DIR/host-build}
mkdir -p "$OUT"
command -v "$CC" >/dev/null || { echo "MISSING host compiler: $CC" >&2; exit 2; }

# -fno-sanitize-address-use-after-scope mirrors the corpus's asan variant flags
# (scripts/build-php-5.0.0-variant.sh:30).
COMMON="-O0 -g -fno-omit-frame-pointer -I$SCRIPT_DIR/php-alloc -I$SCRIPT_DIR/rows"
ASAN="-fsanitize=address -fno-sanitize-address-use-after-scope"
export ASAN_OPTIONS=${ASAN_OPTIONS:-detect_leaks=0:abort_on_error=0:exitcode=86:print_legend=0}

classify() {  # stdin = combined output, $1 = rc
  local blob rc; blob=$(cat); rc=$1
  local kind=""
  # Same kind list, in the same order, as the corpus's scripts/run-corpus.py:33.
  for t in heap-buffer-overflow stack-buffer-overflow global-buffer-overflow \
           heap-use-after-free stack-use-after-return double-free \
           "attempting free on address" SEGV negative-size-param use-after-poison; do
    case "$blob" in *"$t"*) kind=$t; break;; esac
  done
  if [ -n "$kind" ] && case "$blob" in *"ERROR: AddressSanitizer"*) true;; *) false;; esac; then
    echo "ASAN:$kind"
  elif case "$blob" in *"ERROR: AddressSanitizer"*) true;; *) false;; esac; then
    echo "ASAN"
  elif [ "$rc" -ge 128 ]; then echo "SIGNAL:$((rc-128))"
  elif [ "$rc" -ne 0 ]; then echo "EXIT:$rc"
  else echo "OK"; fi
}

printf '%-24s %-28s %-28s %s\n' ROW "stockalloc(real PHP)" "patched(corpus patches)" FLIP
printf '%s\n' "----------------------------------------------------------------------------------------------------"
blind=0; caught=0
for src in "$SCRIPT_DIR"/rows/*.c; do
  row=$(basename "$src" .c)
  declare -A res=()
  for arm in stockalloc patched; do
    extra=""; [ "$arm" = patched ] && extra="-DPHP_PATCHED_ALLOC"
    bin="$OUT/${row}.${arm}"
    if ! $CC $COMMON $ASAN $extra "$src" "$SCRIPT_DIR/php-alloc/php_alloc.c" -o "$bin" \
         2>"$OUT/${row}.${arm}.cc.log"; then
      res[$arm]="BUILD-FAIL"; continue
    fi
    out=$(timeout 20 "$bin" 2>&1); rc=$?
    res[$arm]=$(printf '%s' "$out" | classify "$rc")
    printf '%s\n' "$out" > "$OUT/${row}.${arm}.out"
  done
  flip=""
  case "${res[stockalloc]}" in
    ASAN*) ;;  # already caught with the real allocator: not an ASAN-blind shape
    *) case "${res[patched]}" in ASAN*) flip="BLIND->CAUGHT"; blind=$((blind+1));; esac;;
  esac
  case "${res[stockalloc]}" in ASAN*) caught=$((caught+1));; esac
  printf '%-24s %-28s %-28s %s\n' "$row" "${res[stockalloc]}" "${res[patched]}" "$flip"
done
printf '%s\n' "----------------------------------------------------------------------------------------------------"
echo "rows reproducing the ASAN-blind flip: $blind"
echo "rows ASAN already catches with the real allocator: $caught"
echo "logs: $OUT"
