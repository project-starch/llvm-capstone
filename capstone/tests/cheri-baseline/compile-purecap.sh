#!/usr/bin/env bash
# Compile the trimmed 15-row SQLite defect corpus as CHERI-RISC-V *purecap*
# binaries (plus one purecap sqlite3 amalgamation object), for the CHERI baseline
# of the paper's security table (agentB-015).
#
# Output: $OUT (default a rootfs overlay) holds one ELF per row named
# `<newrow>_<dir>` plus `run-in-guest.sh`. Bake $OUT into the CheriBSD disk image
# (--disk-image/extra-files) or 9p-share it, then run under CHERI-QEMU.
#
# This is the *baseline* column (CHERI), not our system. Measurement only.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
REPROS="$REPO_ROOT/capstone/benchmarks/sqlite/cve-repros"
source "$REPO_ROOT/capstone/tests/capstone-test-env.sh" 2>/dev/null || true

CHERI_SDK=${CHERI_SDK:-/home/alexey/cheri/output/sdk}
SYSROOT=${SYSROOT:-$CHERI_SDK/sysroot-riscv64-purecap}
CC="$CHERI_SDK/bin/clang"
OUT=${OUT:-/home/alexey/cheri-ws/rootfs-overlay/root/cheri-baseline}
# CHERI-CLEAN amalgamation: CheriBSD ships a purecap-patched sqlite3. The vanilla
# upstream amalgamation is NOT capability-clean (int->ptr casts, unaligned cap
# stores) and faults during *normal* operation under purecap, which would mask
# the injected defect. Default to the patched source; keep the vanilla one only
# to demonstrate that contrast (sanity_vanilla).
SQLITE_SRC=${SQLITE_SRC_DIR:-/home/alexey/cheri/cheribsd/contrib/sqlite3}
SQLITE_VANILLA=${SQLITE_VANILLA:-$(bash "$REPO_ROOT/capstone/benchmarks/sqlite/fetch-sqlite.sh" 2>/dev/null | tail -1)}

for p in "$CC" "$SYSROOT" "$SQLITE_SRC/sqlite3.c"; do
  [ -e "$p" ] || { echo "MISSING: $p" >&2; exit 2; }
done
mkdir -p "$OUT"

# --- probe the correct purecap target triple/arch for this SDK -----------------
BASEFLAGS="--sysroot=$SYSROOT -mno-relax"
PURECAP=""
probe=$(mktemp /tmp/cheri-probe-XXXX.c); echo 'int main(void){return 0;}' > "$probe"
for spec in \
  "--target=riscv64-unknown-freebsd -march=rv64gcxcheri -mabi=l64pc128d" \
  "--target=riscv64-unknown-freebsd -march=rv64imafdcxcheri -mabi=l64pc128d" \
  "--target=riscv64-unknown-freebsd -march=rv64gc_xcheri -mabi=l64pc128d"; do
  if "$CC" $spec $BASEFLAGS "$probe" -o "$probe.out" 2>/dev/null; then PURECAP="$spec"; break; fi
done
rm -f "$probe" "$probe.out"
[ -n "$PURECAP" ] || { echo "could not find a working purecap -march for $CC" >&2;
  echo "  (tried rv64gcxcheri / rv64imafdcxcheri / rv64gc_xcheri)" >&2; exit 3; }
CFLAGS="$PURECAP $BASEFLAGS"
# The measured shims/mock are built at -O0 on purpose: a use-after-free / null
# deref is undefined behaviour, and at -O1+ the compiler may hoist the handle
# load before the free or elide the dangling access entirely, so the access we
# want CHERI to police is never emitted. -O0 emits every load/store as written,
# which is the faithful condition for a catch/no-catch measurement. (Contrast the
# borrow-cost task, where -O2 realism was the goal.)
OPT=${OPT:--O0}
echo "[*] purecap flags: $CFLAGS ; measured binaries at $OPT"

# --- sqlite amalgamation (purecap), built once ---------------------------------
SQOBJ="$OUT/sqlite3_purecap.o"
if [ ! -f "$SQOBJ" ] || [ "$SQLITE_SRC/sqlite3.c" -nt "$SQOBJ" ]; then
  echo "[*] compiling SQLite amalgamation purecap (once) ..."
  "$CC" $CFLAGS -O1 -g -DSQLITE_THREADSAFE=0 -DSQLITE_OMIT_LOAD_EXTENSION \
    -I"$SQLITE_SRC" -c "$SQLITE_SRC/sqlite3.c" -o "$SQOBJ" \
    || { echo "sqlite purecap build FAILED" >&2; exit 4; }
fi
# --- mock SQLite lifecycle harness (what the corpus shims actually link against)-
# Upstream SQLite does not run purecap here (sanity_vanilla/_clean below fault);
# the mock reproduces each defect's alloc/free/callback lifecycle so the shims
# compile VERBATIM and the CHERI verdict reflects the defect, not a SQLite port.
MOCK="$SCRIPT_DIR/mock-sqlite"
MOCKOBJ="$OUT/mock_sqlite3.o"
"$CC" $CFLAGS $OPT -g -I"$MOCK" -c "$MOCK/mock_sqlite3.c" -o "$MOCKOBJ" \
  2>"$OUT/mock_cc.log" || { echo "mock build FAILED" >&2; cat "$OUT/mock_cc.log" >&2; exit 4; }
echo "[*] mock SQLite harness built"

# --- revocation-status probe ---------------------------------------------------
if "$CC" $CFLAGS -O1 "$SCRIPT_DIR/cheri_status.c" -o "$OUT/cheri_status" \
     2>"$OUT/cheri_status_cc.log"; then
  echo "  cheri_status: built"
else
  echo "  cheri_status: BUILD-FAIL (see cheri_status_cc.log)" >&2
fi

# --- sanity probes: purecap-clean SQLite must run to completion -----------------
# sanity_clean uses the patched (CHERI-clean) amalgamation SQOBJ (built above).
if "$CC" $CFLAGS -O1 -g -I"$SQLITE_SRC" "$SCRIPT_DIR/sanity_clean.c" "$SQOBJ" -lm \
     -o "$OUT/sanity_clean" 2>"$OUT/sanity_clean_cc.log"; then
  echo "  sanity_clean (patched sqlite): built"
else
  echo "  sanity_clean: BUILD-FAIL" >&2
fi
# sanity_vanilla uses the upstream amalgamation to show the contrast.
if [ -f "$SQLITE_VANILLA/sqlite3.c" ]; then
  VOBJ="$OUT/sqlite3_vanilla.o"
  [ -f "$VOBJ" ] || "$CC" $CFLAGS -O1 -g -DSQLITE_THREADSAFE=0 \
      -DSQLITE_OMIT_LOAD_EXTENSION -I"$SQLITE_VANILLA" -c "$SQLITE_VANILLA/sqlite3.c" \
      -o "$VOBJ" 2>"$OUT/sqlite3_vanilla_cc.log"
  if "$CC" $CFLAGS -O1 -g -I"$SQLITE_VANILLA" "$SCRIPT_DIR/sanity_clean.c" "$VOBJ" -lm \
       -o "$OUT/sanity_vanilla" 2>"$OUT/sanity_vanilla_cc.log"; then
    echo "  sanity_vanilla (upstream sqlite): built"
  fi
fi

# sanity_mock: the mock harness itself must run to completion under purecap.
if "$CC" $CFLAGS $OPT -g -I"$MOCK" "$SCRIPT_DIR/sanity_clean.c" "$MOCKOBJ" \
     -o "$OUT/sanity_mock" 2>"$OUT/sanity_mock_cc.log"; then
  echo "  sanity_mock (mock harness): built"
else
  echo "  sanity_mock: BUILD-FAIL" >&2
fi
# row3_reuse: faithful reuse-not-free variant of the diesel defect (headline).
if "$CC" $CFLAGS $OPT -g "$SCRIPT_DIR/row3_reuse.c" -o "$OUT/3r_row3_reuse" \
     2>"$OUT/row3_reuse_cc.log"; then
  echo "  row3_reuse (reuse-not-free): built"
fi

# --- one purecap ELF per in-scope row (shims compiled VERBATIM vs the mock) -----
built=0; failed=0
while IFS=$'\t' read -r newrow dir oracle klass predA predB; do
  case "$newrow" in ''|\#*) continue;; esac
  src="$REPROS/$dir/before.c"
  [ -f "$src" ] || { echo "  ROW $newrow: MISSING $src" >&2; failed=$((failed+1)); continue; }
  bin="$OUT/${newrow}_${dir}"
  if "$CC" $CFLAGS $OPT -g -I"$MOCK" "$src" "$MOCKOBJ" -lm -o "$bin" \
       2>"$OUT/${newrow}_cc.log"; then
    built=$((built+1)); echo "  ROW $newrow ($dir): built"
  else
    failed=$((failed+1)); echo "  ROW $newrow ($dir): BUILD-FAIL (see ${newrow}_cc.log)" >&2
  fi
done < "$SCRIPT_DIR/rows.tsv"

cp "$SCRIPT_DIR/rows.tsv" "$OUT/rows.tsv"
cp "$SCRIPT_DIR/run-in-guest.sh" "$OUT/run-in-guest.sh" 2>/dev/null || true
chmod +x "$OUT/run-in-guest.sh" 2>/dev/null || true
echo "[*] built=$built failed=$failed  -> $OUT"
[ "$failed" -eq 0 ]
