#!/usr/bin/env bash
# Reproduce the CHERI temporal-safety overhead on the OFFICIAL Computer Language
# Benchmarks Game `binary-trees` Lua benchmark, run as a purecap CheriBSD process.
#
# It builds real Lua 5.4.7 purecap + a tiny rdinstret wrapper, bakes them + the
# benchmark into the CheriBSD image, boots CHERI-QEMU once, and runs three
# revocation configs (calibration-subtracted, n=3 / eager n=1):
#   spatial  : revocation OFF          -> baseline (spatial safety is ALWAYS on)
#   temporal : async quarantine sweep  -> the deployed default
#   eager    : revoke on every free    -> matches Capstone's security, not deployable
#
#   ./reproduce-cheri-lua-bench.sh          # N=6 (default; keeps eager tractable under TCG)
#   N=10 ./reproduce-cheri-lua-bench.sh     # bigger workload (drop eager for large N)
#
# Prereqs (all present on a machine provisioned by cheri-baseline): the CHERI SDK,
# the purecap sysroot, and a provisioned CheriBSD purecap image under $CHERI_ROOT.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CHERI_ROOT=${CHERI_ROOT:-$HOME/cheri}
SDK=${SDK:-$CHERI_ROOT/output/sdk}
ROOTFS=${ROOTFS:-$CHERI_ROOT/rootfs-purecap}
LUA=${LUA_SRC:-$HERE/../../_toolchain/.work/lua54}
PRELUDE=${PRELUDE:-$HERE/../real/cheri-lua-prelude.h}
STAGE=${STAGE:-$CHERI_ROOT/lua-bench}
RUNDIR=${RUNDIR:-$CHERI_ROOT/xlang-run}
BASELINE=${BASELINE:-$HERE/../../../../capstone/tests/cheri-baseline}
N=${N:-6}
mkdir -p "$STAGE"

for p in "$SDK/bin/clang" "$ROOTFS/usr/include" "$LUA/lua.h" "$PRELUDE" \
         "$CHERI_ROOT/output/cheribsd-riscv64-purecap.img"; do
  [ -e "$p" ] || { echo "MISSING: $p" >&2; exit 2; }
done

BUILTIN=$(echo "$SDK"/lib/clang/*/include | tr ' ' '\n' | head -1)
# -nostdinc + -isystem: default header search leaks to host glibc (bits/floatn.h ->
# __float128). -ftls-model=initial-exec -cheri-tgot-tls: CheriBSD's purecap rtld
# rejects traditional/general-dynamic TLS ("Traditional TLS not supported").
CFLAGS=(--target=riscv64-unknown-freebsd -march=rv64gcxcheri -mabi=l64pc128d
        --sysroot="$ROOTFS" -mno-relax -O0 -w -ftls-model=initial-exec -cheri-tgot-tls
        -nostdinc -isystem "$BUILTIN" -isystem "$ROOTFS/usr/include")

echo "== [1/5] build purecap Lua 5.4.7 interpreter =="
# whole interpreter incl. lua.c main; EXCLUDE luac.c/onelua.c (not the interp) and
# ltests.c (internal test lib: allocator hooks would skew the benchmark). The
# prelude no-ops the shared source's DBGP/DBGC diagnostic probes.
luasrcs=(); for f in "$LUA"/*.c; do
  case "$(basename "$f")" in luac.c|onelua.c|ltests.c) ;; *) luasrcs+=("$f");; esac
done
"$SDK/bin/clang" "${CFLAGS[@]}" -include "$PRELUDE" -I"$LUA" -DLUA_USE_POSIX \
  -o "$STAGE/lua" "${luasrcs[@]}" -lm
gd=$("$SDK/bin/llvm-readelf" -r "$STAGE/lua" 2>/dev/null | grep -ciE 'TLS_GD|DTPMOD' || true)
[ "$gd" = 0 ] || { echo "WARN: $gd general-dynamic TLS relocs remain (rtld may reject)"; }

echo "== [2/5] build the rdinstret wrapper (purecap) =="
"$SDK/bin/clang" "${CFLAGS[@]}" -o "$STAGE/runbench" "$HERE/runbench.c"

echo "== [3/5] stage benchmark + driver (N=$N) =="
cp -f "$HERE/binary-trees.lua" "$STAGE/binary-trees.lua"
sed "s/binary-trees\.lua [0-9][0-9]*/binary-trees.lua $N/" "$HERE/run-in-guest.sh" > "$STAGE/run-in-guest.sh"
chmod +x "$STAGE/run-in-guest.sh"

echo "== [4/5] bake into the CheriBSD image =="
OVERLAY_SRC="$STAGE" RUNDIR="$RUNDIR" bash "$BASELINE/provision-cheri-vehicle.sh" >/dev/null

echo "== [5/5] boot CHERI-QEMU once, run spatial/temporal/eager =="
python3 "$BASELINE/cheri-run.py" "$RUNDIR/qemu-argv.txt" "$RUNDIR/lua-bench.log" /root/lua-bench

echo "== results (calibrated: workload = mean(RUN) - mean(CAL)) =="
python3 "$HERE/parse-bench.py" "$RUNDIR/lua-bench.log"
echo "serial log: $RUNDIR/lua-bench.log"
