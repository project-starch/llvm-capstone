#!/usr/bin/env bash
# Compile the PHP-allocator-pattern rows as CHERI-RISC-V *purecap* binaries, in
# two arms: the real PHP 5.0.0 allocator, and the same allocator with the two
# patches the corpus applies to make ASAN able to see the defects at all.
#
# Output: $OUT holds one ELF per (row, arm) named `<arm>__<row>`, plus
# cheri_status and run-in-guest.sh. Bake $OUT into the CheriBSD purecap image.
#
# This is the CHERI BASELINE column, not our system. Measurement only.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

CHERI_SDK=${CHERI_SDK:-$HOME/cheri/output/sdk}
SYSROOT=${SYSROOT:-$CHERI_SDK/sysroot-riscv64-purecap}
CC="$CHERI_SDK/bin/clang"
OUT=${OUT:-$HOME/cheri/rootfs-purecap/root/cheri-php-blind}

for p in "$CC" "$SYSROOT"; do
  [ -e "$p" ] || { echo "MISSING: $p" >&2; exit 2; }
done
mkdir -p "$OUT"

# --- probe the purecap target triple/arch this SDK actually accepts ----------
BASEFLAGS="--sysroot=$SYSROOT -mno-relax"
PURECAP=""
probe=$(mktemp "${TMPDIR:-/tmp}/cheri-probe-XXXX.c"); echo 'int main(void){return 0;}' > "$probe"
for spec in \
  "--target=riscv64-unknown-freebsd -march=rv64gcxcheri -mabi=l64pc128d" \
  "--target=riscv64-unknown-freebsd -march=rv64imafdcxcheri -mabi=l64pc128d" \
  "--target=riscv64-unknown-freebsd -march=rv64gc_xcheri -mabi=l64pc128d"; do
  if "$CC" $spec $BASEFLAGS "$probe" -o "$probe.out" 2>/dev/null; then PURECAP="$spec"; break; fi
done
rm -f "$probe" "$probe.out"
[ -n "$PURECAP" ] || { echo "no working purecap -march for $CC" >&2; exit 3; }
CFLAGS="$PURECAP $BASEFLAGS -I$SCRIPT_DIR/php-alloc"
# -O0 on purpose: a use-after-free or an out-of-bounds read is undefined
# behaviour, and at -O1+ the compiler may hoist the load before the free or
# elide the dangling access entirely, so the access we want CHERI to police is
# never emitted. The same reasoning the SQLite CHERI baseline records.
OPT=${OPT:--O0}
echo "[*] purecap flags: $CFLAGS"
echo "[*] rows built at $OPT"

# --- revocation-status probe (shared with the sqlite baseline) ---------------
STATUS_SRC="$SCRIPT_DIR/../cheri-baseline/cheri_status.c"
if [ -f "$STATUS_SRC" ]; then
  if "$CC" $CFLAGS -O1 "$STATUS_SRC" -o "$OUT/cheri_status" 2>"$OUT/cheri_status_cc.log"; then
    echo "  cheri_status: built"
  else
    echo "  cheri_status: BUILD-FAIL (see cheri_status_cc.log)" >&2
  fi
else
  echo "  cheri_status: source not found at $STATUS_SRC" >&2
fi

# --- every row, in both arms -------------------------------------------------
built=0; failed=0
while IFS=$'\t' read -r row models ncases klass es ep note; do
  case "$row" in ''|\#*) continue;; esac
  src="$SCRIPT_DIR/rows/$row.c"
  [ -f "$src" ] || { echo "  ROW $row: MISSING $src" >&2; failed=$((failed+1)); continue; }
  for arm in stockalloc patched; do
    extra=""; [ "$arm" = patched ] && extra="-DPHP_PATCHED_ALLOC"
    bin="$OUT/${arm}__${row}"
    if "$CC" $CFLAGS $OPT -g $extra "$src" "$SCRIPT_DIR/php-alloc/php_alloc.c" \
         -o "$bin" 2>"$OUT/${arm}__${row}.cc.log"; then
      built=$((built+1))
    else
      failed=$((failed+1)); echo "  ROW $row ($arm): BUILD-FAIL" >&2
    fi
  done
  echo "  ROW $row: both arms attempted"
done < "$SCRIPT_DIR/rows.tsv"

cp "$SCRIPT_DIR/rows.tsv" "$OUT/rows.tsv"
cp "$SCRIPT_DIR/run-in-guest.sh" "$OUT/run-in-guest.sh"
chmod +x "$OUT/run-in-guest.sh"
echo "[*] built=$built failed=$failed -> $OUT"
[ "$failed" -eq 0 ]
