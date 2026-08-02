#!/usr/bin/env bash
# Bring up mruby as a CHERI-RISC-V purecap interpreter that actually RUNS.
#
# Takes an mruby source tree, applies the one source change, builds it with the
# purecap toolchain and the three config/ABI flags, then VERIFIES the result is
# genuinely capability-mode rather than trusting that it built.
#
# What it does NOT do: run it. mruby needs a CheriBSD guest for that — stage
# the binary into the image built by capstone/tests/cheri-baseline/provision-cheri-vehicle.sh
# and drive it with cheri-run.py. The final message prints the exact commands.
#
#   ./build-purecap-mruby.sh                    # uses $MRUBY_SRC or clones one
#   MRUBY_SRC=/path/to/mruby ./build-purecap-mruby.sh
#   ./build-purecap-mruby.sh --probe            # also build probe_run_ruby.c
#
# Prerequisites: the CHERI SDK and a purecap sysroot (both produced by
# provision-cheri-vehicle.sh), plus ruby and rake for mruby's own build system.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

CHERI_ROOT=${CHERI_ROOT:-$HOME/cheri}
SDK=${SDK:-$CHERI_ROOT/output/sdk}
SYSROOT=${SYSROOT:-$CHERI_ROOT/rootfs-purecap}
MRUBY_SRC=${MRUBY_SRC:-$CHERI_ROOT/mruby-purecap}
# The corpus pins per-row mruby versions; this is row 10's (CVE-2022-1106),
# the tree the port was developed against.
MRUBY_REPO=${MRUBY_REPO:-https://github.com/mruby/mruby}
MRUBY_REV=${MRUBY_REV:-bf5bbf0a4b7f19ea3960e59f32ec252b3aee2c1a}

for p in "$SDK/bin/clang" "$SYSROOT/usr/include"; do
  [ -e "$p" ] || { echo "MISSING: $p"; echo "  run capstone/tests/cheri-baseline/provision-cheri-vehicle.sh first"; exit 2; }
done
command -v rake >/dev/null || { echo "MISSING: rake (mruby builds with ruby+rake)"; exit 2; }

# --- source ------------------------------------------------------------------
if [ ! -d "$MRUBY_SRC/src" ]; then
  echo "== fetching mruby $MRUBY_REV =="
  git clone -q "$MRUBY_REPO" "$MRUBY_SRC" || { echo "clone failed"; exit 1; }
  git -C "$MRUBY_SRC" checkout -q "$MRUBY_REV" || { echo "checkout failed"; exit 1; }
fi

# --- the one source change ---------------------------------------------------
# RSTRING_EMBED_LEN_MAX is 4*sizeof(void*)-5: 27 at 8-byte pointers, 59 at
# 16-byte capabilities, which no longer fits the 5-bit length field. Fails as a
# _Static_assert, so the build stops before this is even reachable at runtime.
if grep -q '#define MRB_STR_EMBED_LEN_BIT 5' "$MRUBY_SRC/include/mruby/string.h"; then
  echo "== applying embed-length widening =="
  git -C "$MRUBY_SRC" apply "$HERE/mruby-purecap-embed-len.patch" \
    || sed -i 's/#define MRB_STR_EMBED_LEN_BIT 5/#define MRB_STR_EMBED_LEN_BIT 6/' \
         "$MRUBY_SRC/include/mruby/string.h"
fi

cp "$HERE/build_config_purecap.rb" "$MRUBY_SRC/"

# --- build -------------------------------------------------------------------
# The flags in build_config_purecap.rb, and what each one is for:
#   -ftls-model=initial-exec  purecap CheriBSD requires it
#   -cheri-tgot-tls           purecap uses capability TGOT TLS; without it the
#                             binary keeps R_RISCV_TLS_TPREL64 relocations and
#                             ld-elf.so.1 refuses it outright
#   -DMRB_USE_METHOD_T_STRUCT method pointers otherwise packed into an integer
#                             with flag bits; the shift clears the capability
#                             tag irreversibly -> PROT_CHERI_TAG at vm.c:561
#   -DPOOL_ALIGNMENT=16       parser pool hands out 8-aligned AST cons cells
#                             whose node* fields are 16-byte capabilities
#                             -> BUS_ADRALN at parse.y:125
echo "== building purecap mruby =="
( cd "$MRUBY_SRC" && rm -rf build/purecap \
  && CHERI_SDK="$SDK" CHERI_SYSROOT="$SYSROOT" \
     rake MRUBY_CONFIG=build_config_purecap.rb ) > "$MRUBY_SRC/purecap-build.log" 2>&1
rc=$?
BIN="$MRUBY_SRC/build/purecap/bin/mruby"
LIB="$MRUBY_SRC/build/purecap/lib/libmruby.a"
if [ "$rc" -ne 0 ] || [ ! -f "$BIN" ]; then
  echo "build failed (see $MRUBY_SRC/purecap-build.log)"
  grep -m5 -E 'error:' "$MRUBY_SRC/purecap-build.log" | cut -c1-120
  exit 1
fi

# --- verify: capability-mode, and no traditional TLS -------------------------
# Checked rather than assumed: a binary can build and still be hybrid, and the
# TLS relocations are what rtld rejects at load time.
echo "== verifying =="
fail=0
if "$SDK/bin/llvm-readelf" -h "$BIN" | grep -q 'cheriabi'; then
  echo "  [ok]   cheriabi, capability mode"
else
  echo "  [FAIL] not a purecap binary"; fail=1
fi
tprel=$("$SDK/bin/llvm-readelf" -r "$BIN" 2>/dev/null | grep -c TPREL)
if [ "$tprel" -eq 0 ]; then
  echo "  [ok]   no TPREL relocations (TGOT TLS in use)"
else
  echo "  [FAIL] $tprel TPREL relocations; rtld will refuse this"; fail=1
fi
[ "$fail" -eq 0 ] || exit 1

if [ "${1:-}" = "--probe" ]; then
  echo "== building probe_run_ruby =="
  "$SDK/bin/clang" --target=riscv64-unknown-freebsd -march=rv64gcxcheri \
    -mabi=l64pc128d --sysroot="$SYSROOT" -mno-relax -ftls-model=initial-exec \
    -cheri-tgot-tls -DMRB_USE_METHOD_T_STRUCT -O0 -g \
    -I"$MRUBY_SRC/include" -I"$MRUBY_SRC/build/purecap/include" \
    "$HERE/probe_run_ruby.c" "$LIB" -lm -o "$MRUBY_SRC/build/purecap/bin/probe_run_ruby" \
    && echo "  [ok]   probe_run_ruby"
fi

cat <<EOF

purecap mruby built:  $BIN
                      $LIB

It needs a CheriBSD guest to run. To try it:

  1. stage it into the image's overlay and rebuild the image
       cp $BIN \$ROOTFS/root/mruby-real/
       (rebuild with provision-cheri-vehicle.sh, or makefs directly)
  2. boot and drive it
       python3 capstone/tests/cheri-baseline/cheri-run.py \\
         \$CHERI_ROOT/xlang-run/qemu-argv.txt serial.log /root/mruby-real

probe_run_ruby.c (--probe) is the smoke test: it opens a VM, evaluates
arithmetic, puts, and blocks, then loads a corpus trigger, with SIGPROT and
SIGBUS handlers so any fault reports its cause and address.

NOTE: a corpus trigger running to exit 0 is NOT a CHERI verdict. See the
"not yet a MEASUREMENT" section of README.md before quoting any such run.
EOF
