#!/usr/bin/env bash
# Link full tshark, staged starts and allocator fixtures with the shared ABI-v2 SDK.
#
# TSAPP_HEAP=chunks is the sublet arm with ONE difference: wmem's BLOCK allocator is the wmem
# port's chunk port (ports/wireshark/wmem, patches 0001+0002, the source its replay harness tests),
# so every chunk is a region of its own and a chunk free, a scope reset and a block's end are
# revokes. Its blocks come LINEAR from the Sublet heap (__capstone_sublet_malloc_linear), its
# headers live in heap records beside them (src/tsapp-wmem-chunks.c), and the port's own
# src/allocators/sublet/chunks.c carves and revokes. BLOCK_FAST is the sublet arm's, unchanged.
# Patch 0007's block size applies to both files; on the ported block allocator its one macro edit
# is re-anchored, since 0002 changed its context.
set -euo pipefail
APP=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$APP/deps/env.sh"
B=$TS_WORK/xbuild
HEAP=${TSAPP_HEAP:-level0}
case $HEAP in
  level0) OUT=$TS_WORK/domain HEAPF=(-DCAPSTONE_LEVEL0_OBJECT_BOUNDS=0) ;;
  shrink) OUT=$TS_WORK/domain-shrink HEAPF=() ;;
  sublet) OUT=$TS_WORK/domain-sublet HEAPF=() ;;
  chunks) OUT=$TS_WORK/domain-chunks HEAPF=() ;;
  *) echo "TSAPP_HEAP must be level0, shrink, sublet or chunks" >&2; exit 2 ;;
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
[[ $HEAP == chunks ]] && SDK_HEAP=sublet
bash "$CAPSTONE_REPO_ROOT/capstone/ports/common/application/build-sdk.sh" \
  "$SDK" "$TS_MUSL" "$TS_LIBC_ARCHIVE" \
  -DCAPSTONE_APPLICATION_HEAP="$SDK_HEAP" -DCAPSTONE_APPLICATION_HEAP_LOG=24 \
  -DCAPSTONE_APPLICATION_ARENA_BYTES="$ARENA" -DCAPSTONE_APPLICATION_STACK_BYTES="$STACK" \
  -DCMAKE_C_FLAGS_RELEASE="-O1 -DCAPSTONE_LEVEL0_STATS -DCAPSTONE_SUBLET_HEAP_STATS ${HEAPF[*]}"
export CAPSTONE_SDK=$SDK
RT=() HEAPOBJ=()
if [[ $HEAP == sublet || $HEAP == chunks ]]; then
  WMEMF=(); [[ $HEAP == chunks ]] && WMEMF=(-DTSAPP_WMEM_CHUNKS)
  "$SDK/capstone-cc" -O1 -DTSAPP_SUBLET_HEAP "${WMEMF[@]}" -c "$APP/src/tsapp-heap.c" -o "$OUT/tsapp-heap.o"
  # wmem's block allocators from patch 0007, with their own ninja commands (less the dependency
  # files, which would overwrite ninja's record), on copies: the cross-built tree stays as it is.
  mkdir -p "$OUT/wmem-src/wsutil/wmem"
  for f in wmem_allocator_block wmem_allocator_block_fast; do
    cp "$TS_WORK/xsrc/wsutil/wmem/$f.c" "$OUT/wmem-src/wsutil/wmem/"
  done
  patch -s -d "$OUT/wmem-src" -p1 < "$APP/patches/0007-capstone-wmem-block-size-for-the-sublet-heap.patch"
  HEAPOBJ=()
  WPORT=$CAPSTONE_REPO_ROOT/capstone/ports/wireshark/wmem
  WCF=()
  if [[ $HEAP == chunks ]]; then
    # The chunk port's block allocator: upstream + the wmem port's 0001 and 0002, exactly the
    # source its replay harness builds (0001 touches only the four allocator .c files), then
    # 0007's one macro edit. xsrc's copy must be upstream's, or this is not the tested source.
    WC=$OUT/wmem-chunks-src; rm -rf "$WC"; mkdir -p "$WC/wsutil/wmem"
    for f in wmem_allocator_block wmem_allocator_block_fast wmem_allocator_simple wmem_allocator_strict; do
      cp "$TS_WORK/xsrc/wsutil/wmem/$f.c" "$WC/wsutil/wmem/"
    done
    for p in "$WPORT"/patches/wireshark-4.6.8-0001-*.patch "$WPORT"/patches/wireshark-4.6.8-0002-*.patch; do
      patch -s -d "$WC" -p1 --batch --forward --fuzz=0 < "$p" || { echo "$(basename "$p") does not apply at fuzz 0" >&2; exit 1; }
    done
    python3 - "$WC/wsutil/wmem/wmem_allocator_block.c" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = "#define WMEM_BLOCK_SIZE (8 * 1024 * 1024)\n"
new = ("#ifdef CAPSTONE_WMEM_BLOCK_BYTES\n#define WMEM_BLOCK_SIZE (CAPSTONE_WMEM_BLOCK_BYTES)\n"
       "#else\n#define WMEM_BLOCK_SIZE (8 * 1024 * 1024)\n#endif\n")
if s.count(old) != 1: sys.exit("0007's block-size anchor is not unique in the ported block allocator")
open(p, "w").write(s.replace(old, new))
PY
    cp "$WC/wsutil/wmem/wmem_allocator_block.c" "$OUT/wmem-src/wsutil/wmem/wmem_allocator_block.c"
    WCF=(-DWMEM_PORT_HOOKS -DWMEM_PORT_CHUNKS -DWM_DOMAIN -I"$WPORT/src/shared"
         -I"$CAPSTONE_REPO_ROOT/capstone/runtime/include")
    # The level below: the port's own chunks.c, unchanged, and this app's backing for it.
    # TSAPP_WMEM_ABLATE=1: the chunk free's one give stubbed out (the harness's WM_P1_ABLATE), the
    # matched arm that attributes fixture 13 to that revoke and shows the counter identity can fail.
    ABL=(); [[ ${TSAPP_WMEM_ABLATE:-0} == 1 ]] && ABL=(-DWM_ABLATE_RETIRE_GIVE)
    for s in "$WPORT/src/allocators/sublet/chunks.c" "$APP/src/tsapp-wmem-chunks.c"; do
      "$SDK/capstone-cc" -O1 -std=c11 -DWM_DOMAIN "${ABL[@]}" -I"$WPORT/src/shared" \
        -I"$CAPSTONE_REPO_ROOT/capstone/runtime/include" -c "$s" -o "$OUT/$(basename "${s%.c}").o"
      HEAPOBJ+=("$OUT/$(basename "${s%.c}").o")
    done
  fi
  for f in wmem_allocator_block wmem_allocator_block_fast; do
    o=wsutil/CMakeFiles/wsutil.dir/wmem/$f.c.o
    c=$(ninja -C "$B" -t commands "$o" | tail -1)
    tail=" -MD -MT $o -MF $o.d -o $o -c $TS_WORK/xsrc/wsutil/wmem/$f.c"
    [[ $c == *"$tail" ]] || { echo "$o's compile command does not end as expected" >&2; exit 1; }
    WF=(); [[ $f == wmem_allocator_block ]] && WF=("${WCF[@]}")
    ( cd "$B" && eval "${c%"$tail"} -I$TS_WORK/xsrc/wsutil/wmem -DCAPSTONE_WMEM_BLOCK_BYTES=1048576 ${WF[*]} -o $OUT/$f.o -c $OUT/wmem-src/wsutil/wmem/$f.c" )
    HEAPOBJ+=("$OUT/$f.o")
  done
else
  "$SDK/capstone-cc" -O1 -c "$APP/src/tsapp-heap.c" -o "$OUT/tsapp-heap.o"
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
for n in $(seq 1 13); do
  ( cd "$B" && eval "${CMD%"$DEPF"} -DTSAPP_FIXTURE=$n -o $OUT/tsapp_fx$n.o -c $APP/src/tsapp-safety.c" ) \
    || { echo "compile failed: fixture $n" >&2; exit 1; }
  link "$OUT/tsapp_fx$n.dom" "$OUT/tsapp_fx$n.o" "${RT[@]}" > "$OUT/link-fx$n.log" 2>&1 \
    || { echo "LINK FAILED: fixture $n"; grep -m5 -E 'error' "$OUT/link-fx$n.log"; exit 1; }
  gates "$OUT/tsapp_fx$n.dom" "fixture $n"
done

# NEGATIVE CONTROL, chunks arm: M5 relinked without chunks.o comes back with EXACTLY the chunk
# port's twelve entry points undefined. It shows the LINK gate can fire, and pins that the ported
# block allocator and its backing (tsapp-wmem-chunks.o) reach the chunk port through every entry
# point chunks.o defines; a link that fell back to upstream's allocator would reference none. It is
# the ABI-v2 successor of the v0 control, "M5 without hostcall.o", which the chunks arm's T1 names
# (wmem/PREREGISTRATION-tshark-step2.md). That one has no v2 form on any arm: an ABI-v2 SDK links
# its runtime archive whole, so there is no runtime object to leave out; level0 and sublet
# therefore run no link control (README.md).
if [[ $HEAP == chunks ]]; then
  KEEP=("${HEAPOBJ[@]}") NOCH=()
  for o in "${HEAPOBJ[@]}"; do [ "$(basename "$o")" = chunks.o ] || NOCH+=("$o"); done
  [ ${#NOCH[@]} -lt ${#KEEP[@]} ] || { echo "CONTROL: chunks.o is not among the heap objects" >&2; exit 1; }
  HEAPOBJ=("${NOCH[@]}")
  set +e; ctl=$(link "$OUT/nochunks.dom" "$OUT/tshark_m5.o" 2>&1); set -e
  HEAPOBJ=("${KEEP[@]}")
  und=$(printf '%s\n' "$ctl" | grep -oE 'undefined symbol: [A-Za-z_][A-Za-z0-9_]*' | sed 's/undefined symbol: //' | LC_ALL=C sort -u | tr '\n' ' ')
  want="wm_block_close wm_block_forget wm_block_open wm_block_reset wm_chunk_adopt wm_chunk_bytes wm_chunk_forget wm_chunk_issue wm_chunk_report wm_chunk_retire wm_chunk_split wm_chunks_init "
  [ "$und" = "$want" ] \
    || { echo "CONTROL FAILED: without chunks.o expected exactly ${want}undefined, got: ${und:-<none>}"; exit 1; }
  rm -f "$OUT/nochunks.dom"
  echo "control fired: without chunks.o exactly the chunk port's 12 entry points are missing"
fi

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
