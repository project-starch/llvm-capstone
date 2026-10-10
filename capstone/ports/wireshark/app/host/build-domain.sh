#!/usr/bin/env bash
# Link full tshark, staged starts and allocator fixtures with the shared ABI-v2 SDK.
set -euo pipefail
APP=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$APP/deps/env.sh"
B=$TS_WORK/xbuild
HEAP=${TSAPP_HEAP:-level0}
case $HEAP in
  level0) OUT=$TS_WORK/domain HEAPF=(-DCAPSTONE_LEVEL0_OBJECT_BOUNDS=0) ;;
  shrink) OUT=$TS_WORK/domain-shrink HEAPF=() ;;
  sublet) OUT=$TS_WORK/domain-sublet HEAPF=() ;;
  *) echo "TSAPP_HEAP must be level0, shrink or sublet" >&2; exit 2 ;;
esac
ARENA=${TSAPP_ARENA_BYTES:-$((40 << 20))}
STACK=${TSAPP_STACK_BYTES:-$((1 << 20))}
LD=${CAPSTONE_LD_LLD:?}
mkdir -p "$OUT"; rm -rf "$OUT"/*.dom "$OUT"/*.o "$OUT/wmem-src"
[ -f "$B/build.ninja" ] || { echo "no cross build in $B; run host/cross-build.sh" >&2; exit 2; }

# tshark's link inputs, from ninja: objects and archives (the -l flags are musl's own).
mapfile -t INPUTS < <(ninja -C "$B" -t commands tshark | tail -1 | tr ' ' '\n' | grep -E '\.(o|a)$')
TSO=CMakeFiles/tshark.dir/tshark.c.o
printf '%s\n' "${INPUTS[@]}" | grep -qxF "$TSO" || { echo "tshark.c.o not among the link inputs" >&2; exit 1; }
grep -q 'TSAPP_STAGE(2)' "$TS_WORK/xsrc/tshark.c" || { echo "xsrc/tshark.c lacks patch 0006" >&2; exit 1; }

# All five tshark objects are compiled here, from the current source, with ninja's own compile
# command for tshark.c.o (another -o; the define for M1-M4). Not `ninja tshark.c.o`: every ninja
# invocation in this tree re-runs CMake (its glob check is always dirty) and then recompiles
# prefs.c, proto.c and manuf.c, and manuf.c alone takes about 90 minutes (2026-09-24, twice).
# Compiling M5 here too means no object can be older than the patched source.
CMD=$(ninja -C "$B" -t commands "$TSO" | tail -1)
for n in 1 2 3 4 5; do
  def="-DTSAPP_STOP_AT=$n"; [ "$n" = 5 ] && def=""
  ( cd "$B" && eval "${CMD/ -o $TSO / $def -o $OUT/tshark_m$n.o }" ) || { echo "compile failed: m$n" >&2; exit 1; }
done
# The stops are opaque to the compiler (patch 0006's volatile flag), so a staged object keeps the
# code after its stop. With a plain exit() M1's object was 7.7 KB against M5's 207 KB.
for n in 1 2 3 4; do
  a=$(stat -c %s "$OUT/tshark_m$n.o") b=$(stat -c %s "$OUT/tshark_m5.o")
  echo "tshark_m$n.o: $a bytes, M5's $b"
  [ $((a * 100)) -ge $((b * 95)) ] || { echo "STAGE GATE: tshark_m$n.o is much smaller than M5's; code after the stop was dropped" >&2; exit 1; }
done

SDK=$OUT/sdk
SDK_HEAP=$HEAP
[[ $HEAP == shrink ]] && SDK_HEAP=level0
bash "$CAPSTONE_REPO_ROOT/capstone/ports/common/application/build-sdk.sh" \
  "$SDK" "$TS_MUSL" "$TS_LIBC_ARCHIVE" \
  -DCAPSTONE_APPLICATION_HEAP="$SDK_HEAP" -DCAPSTONE_APPLICATION_HEAP_LOG=24 \
  -DCAPSTONE_APPLICATION_ARENA_BYTES="$ARENA" -DCAPSTONE_APPLICATION_STACK_BYTES="$STACK" \
  -DCMAKE_C_FLAGS_RELEASE="-O1 -DCAPSTONE_LEVEL0_STATS -DCAPSTONE_SUBLET_HEAP_STATS ${HEAPF[*]}"
export CAPSTONE_SDK=$SDK
PROFILEF=()
[[ ${CAPSTONE_APPLICATION_PROFILE:-physical} != virtual ]] || PROFILEF=(-DTSAPP_VIRTUAL_HEAP)
RT=() HEAPOBJ=()
if [[ $HEAP == sublet ]]; then
  "$SDK/capstone-cc" -O1 "${PROFILEF[@]}" -DTSAPP_SUBLET_HEAP -c "$APP/src/tsapp-heap.c" -o "$OUT/tsapp-heap.o"
  # wmem's block allocators from patch 0007, with their own ninja commands (less the dependency
  # files, which would overwrite ninja's record), on copies: the cross-built tree stays as it is.
  mkdir -p "$OUT/wmem-src/wsutil/wmem"
  for f in wmem_allocator_block wmem_allocator_block_fast; do
    cp "$TS_WORK/xsrc/wsutil/wmem/$f.c" "$OUT/wmem-src/wsutil/wmem/"
  done
  patch -s -d "$OUT/wmem-src" -p1 < "$APP/patches/0007-capstone-wmem-block-size-for-the-sublet-heap.patch"
  HEAPOBJ=()
  for f in wmem_allocator_block wmem_allocator_block_fast; do
    o=wsutil/CMakeFiles/wsutil.dir/wmem/$f.c.o
    c=$(ninja -C "$B" -t commands "$o" | tail -1)
    tail=" -MD -MT $o -MF $o.d -o $o -c $TS_WORK/xsrc/wsutil/wmem/$f.c"
    [[ $c == *"$tail" ]] || { echo "$o's compile command does not end as expected" >&2; exit 1; }
    ( cd "$B" && eval "${c%"$tail"} -I$TS_WORK/xsrc/wsutil/wmem -DCAPSTONE_WMEM_BLOCK_BYTES=1048576 -o $OUT/$f.o -c $OUT/wmem-src/wsutil/wmem/$f.c" )
    HEAPOBJ+=("$OUT/$f.o")
  done
else
  "$SDK/capstone-cc" -O1 "${PROFILEF[@]}" -c "$APP/src/tsapp-heap.c" -o "$OUT/tsapp-heap.o"
fi

link() {  # out-image tshark-object [runtime objects...]
  local out=$1 tso=$2; shift 2
  local ins=(); for i in "${INPUTS[@]}"; do
    if [ "$i" = "$TSO" ]; then ins+=("$tso"); elif [ "${i#/}" != "$i" ]; then ins+=("$i"); else ins+=("$B/$i"); fi
  done
  "$SDK/capstone-cc" -o "$out" "$@" "${HEAPOBJ[@]}" "$OUT/tsapp-heap.o" \
    "${ins[@]}" "$TS_LIBC_ARCHIVE"
}
gates() {  # image label
  local w o
  w=$("$CAPSTONE_LLVM_BIN/llvm-nm" "$1" | grep -cE ' [wv] ' || true)
  [ "$w" = 0 ] || { echo "LINK GATE: $2 has $w undefined weak symbols"; exit 1; }
  # Constructors and destructors run only from between link.ld's array markers, which the runtime
  # walks (hostcall.c, C-64). .ctors/.dtors, or a priority section link.ld does not place, would be
  # an orphan outside them and silently never run.
  o=$("$CAPSTONE_LLVM_BIN/llvm-readelf" -SW "$1" | grep -oE '\.(init_array|fini_array)\.[^ ]+|\.(ctors|dtors)[^ ]*' | sort -u | tr '\n' ' ' || true)
  [ -z "$o" ] || { echo "LINK GATE: $2 has constructor sections nothing runs: $o"; exit 1; }
}
for n in 1 2 3 4 5; do
  link "$OUT/tshark_m$n.dom" "$OUT/tshark_m$n.o" "${RT[@]}" > "$OUT/link-m$n.log" 2>&1 \
    || { echo "LINK FAILED: m$n"; grep -m5 -E 'error' "$OUT/link-m$n.log"; exit 1; }
  gates "$OUT/tshark_m$n.dom" "m$n"
done

# The safety fixtures. tshark.c.o's own compile command, less its dependency-file flags (they would
# overwrite ninja's record for tshark.c.o with the fixture's), for another source and object.
DEPF=" -MD -MT $TSO -MF $TSO.d -o $TSO -c $TS_WORK/xsrc/tshark.c"
[[ $CMD == *"$DEPF" ]] || { echo "tshark.c.o's compile command does not end as expected: ${CMD: -200}" >&2; exit 1; }
for n in $(seq 1 15); do
  ( cd "$B" && eval "${CMD%"$DEPF"} -DTSAPP_FIXTURE=$n -o $OUT/tsapp_fx$n.o -c $APP/src/tsapp-safety.c" ) \
    || { echo "compile failed: fixture $n" >&2; exit 1; }
  link "$OUT/tsapp_fx$n.dom" "$OUT/tsapp_fx$n.o" "${RT[@]}" > "$OUT/link-fx$n.log" 2>&1 \
    || { echo "LINK FAILED: fixture $n"; grep -m5 -E 'error' "$OUT/link-fx$n.log"; exit 1; }
  gates "$OUT/tsapp_fx$n.dom" "fixture $n"
done

for n in 1 2 3 4 5; do
  python3 - "$OUT/tshark_m$n.dom" 33554688 <<'PY'
import subprocess, sys
img, stack = sys.argv[1], int(sys.argv[2])
out = subprocess.run(['readelf', '-lW', img], capture_output=True, text=True).stdout
memsz = max(int(l.split()[5], 16) for l in out.splitlines() if l.split()[:1] == ['LOAD'])
pages = (memsz + 8192 + stack - 1) // 4096 + 1
order = (pages - 1).bit_length() if pages > 1 else 0
print(f"{img.rsplit('/', 1)[1]}  code_len={memsz} ({memsz / 2**20:.1f} MiB)  declared stack={stack}  "
      f"block=order {order} = {(1 << order) * 4096 >> 20} MiB")
PY
done
( cd "$OUT" && sha256sum tshark_m*.dom tsapp_fx*.dom ) > "$OUT/SHA256SUMS"
echo "heap arm ${TSAPP_HEAP:-level0}: $OUT"
