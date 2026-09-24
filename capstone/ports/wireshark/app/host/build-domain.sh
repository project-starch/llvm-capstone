#!/usr/bin/env bash
# M0 and the staged images: the cross-built minimal tshark (host/cross-build.sh) linked as
# Capstone domains, FFmpeg-app style (ports/ffmpeg/app/host/build-domain.sh).
#
#   build-domain.sh      -> $TS_WORK/domain/tshark_m{1,2,3,4,5}.dom
#
# M1-M4 are tshark.c recompiled with -DTSAPP_STOP_AT=<n> (patch 0006): at milestone n the image
# prints `TSAPP-STAGE n` and exits with 100 + n. M5 is the unmodified program. Everything else is the same objects and archives
# the cross build linked tshark from (ninja's own link command), so the images differ only in
# that one object.
#
# The link is this port's own, not capstone-cc's:
# - level0 with a tshark-sized arena (TSAPP_ARENA_BYTES, default 40 MiB). Native peak heap is
#   27.8 MB with patch 0005, of which 27.3 MB is fixed-size wmem arenas (plan, results item 4);
# - a declared stack (.capstone_domreq, TSAPP_STACK_BYTES, default 1 MiB): manuf.c's
#   capability-initialiser frame alone is 322,080 bytes (results item 5);
# - level0 built with CAPSTONE_LEVEL0_STATS, and src/tsapp-heap.c as the runtime's exit hook: every
#   image ends with one `TSAPP-HEAP ... peak_end=<bytes>` line on stderr, the arena it needed;
# - src/tsapp-init-fini.c: the constructors (GLib, libgpg-error) and the destructor (libxml2) run,
#   which nothing else in a domain does, and exit no longer walks .fini_array through integers.
#
# GATES, each failing the build:
# - LINK: no undefined symbols, and no undefined weak symbol (the runner refuses the image; C-56);
# - NEGATIVE CONTROL: M5 relinked without hostcall.o comes back with exactly __capstone_hostcall
#   (what the libc's syscalls call) and domain_main (what start-musl calls) undefined. It shows
#   the LINK gate can fire, and pins what hostcall.o supplies: dropping start-musl.o as well
#   fails it. It cannot see a missing libc OVERRIDE: dropping string_bounds_safe.o too still
#   passes, because musl's own memcpy then links in its place (tested on a stand-in, 2026-09-24);
# - SIZE: code_len (PT_LOAD memsz) and the block the kernel module sizes for it
#   (capstone.c: code_len + 8 KiB + declared data, rounded to a power of two), printed per image.
set -euo pipefail
APP=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$APP/deps/env.sh"
B=$TS_WORK/xbuild OUT=$TS_WORK/domain
ARENA=${TSAPP_ARENA_BYTES:-$((40 << 20))}
STACK=${TSAPP_STACK_BYTES:-$((1 << 20))}
LD=${CAPSTONE_LD_LLD:?}
mkdir -p "$OUT"; rm -f "$OUT"/*.dom "$OUT"/*.o
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

# Runtime: env.sh's objects, with this image's own level0 and the domain requirement.
RTF=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m -Xclang -target-feature -Xclang +a
     -ffreestanding -fno-builtin -fno-jump-tables -ffunction-sections -fdata-sections -std=c99 -O1 -w
     -Wno-int-conversion -D_XOPEN_SOURCE=700 -nostdinc
     -isystem "$TS_MUSL/arch/capstone64" -isystem "$TS_MUSL/arch/generic" -isystem "$TS_MUSL/obj/include"
     -isystem "$TS_MUSL/include" -I"$TS_MUSL/src/include" -I"$TS_MUSL/src/internal" -I"$TS_MUSL/obj/src/internal")
"$CAPSTONE_CLANG" "${RTF[@]}" -DCAPSTONE_LEVEL0_ARENA_BYTES="$ARENA" -DCAPSTONE_LEVEL0_STATS \
  -c "$CAPSTONE_REPO_ROOT/capstone/ports/musl-capstone/runtime/level0.c" -o "$OUT/level0.o"
"$CAPSTONE_CLANG" "${RTF[@]}" -c "$APP/src/tsapp-heap.c" -o "$OUT/tsapp-heap.o"
"$CAPSTONE_CLANG" "${RTF[@]}" -c "$APP/src/tsapp-init-fini.c" -o "$OUT/tsapp-init-fini.o"
"$CAPSTONE_CLANG" -target capstone64-unknown-elf -ffreestanding -O0 -DCAPSTONE_DOMREQ_DATA="$STACK" \
  -DCAPSTONE_DOMREQ_STACK="$STACK" -c "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/domreq.S" -o "$OUT/domreq.o"
RT=(); for o in "$TS_RUNTIME_DIR"/*.o; do [ "$(basename "$o")" = level0.o ] || RT+=("$o"); done

link() {  # out-image tshark-object [runtime objects...]
  local out=$1 tso=$2; shift 2
  local ins=(); for i in "${INPUTS[@]}"; do
    if [ "$i" = "$TSO" ]; then ins+=("$tso"); elif [ "${i#/}" != "$i" ]; then ins+=("$i"); else ins+=("$B/$i"); fi
  done
  "$LD" --gc-sections -T "$TS_LINKER_SCRIPT" -o "$out" "$@" "$OUT/level0.o" "$OUT/tsapp-heap.o" "$OUT/tsapp-init-fini.o" "$OUT/domreq.o" \
    "${ins[@]}" "$TS_LIBC_ARCHIVE"
}
for n in 1 2 3 4 5; do
  link "$OUT/tshark_m$n.dom" "$OUT/tshark_m$n.o" "${RT[@]}" > "$OUT/link-m$n.log" 2>&1 \
    || { echo "LINK FAILED: m$n"; grep -m5 -E 'error' "$OUT/link-m$n.log"; exit 1; }
  w=$("$CAPSTONE_LLVM_BIN/llvm-nm" "$OUT/tshark_m$n.dom" | grep -cE ' [wv] ' || true)
  [ "$w" = 0 ] || { echo "LINK GATE: m$n has $w undefined weak symbols"; exit 1; }
  # Constructors and destructors run only from link.ld's plain .init_array/.fini_array, which
  # src/tsapp-init-fini.c walks. A priority section (.init_array.NNN) or .ctors/.dtors is an orphan
  # outside those markers, and would silently never run.
  o=$("$CAPSTONE_LLVM_BIN/llvm-readelf" -SW "$OUT/tshark_m$n.dom" | grep -oE '\.(init_array|fini_array)\.[^ ]+|\.(ctors|dtors)[^ ]*' | sort -u | tr '\n' ' ' || true)
  [ -z "$o" ] || { echo "LINK GATE: m$n has constructor sections nothing runs: $o"; exit 1; }
done

# Negative control: M5 without hostcall.o.
NOHC=(); for o in "${RT[@]}"; do [ "$(basename "$o")" = hostcall.o ] || NOHC+=("$o"); done
set +e; ctl=$(link "$OUT/nohostcall.dom" "$OUT/tshark_m5.o" "${NOHC[@]}" 2>&1); set -e
und=$(printf '%s\n' "$ctl" | grep -oE 'undefined symbol: [A-Za-z_][A-Za-z0-9_]*' | sed 's/undefined symbol: //' | sort -u | tr '\n' ' ')
[ "$und" = "__capstone_hostcall domain_main " ] \
  || { echo "CONTROL FAILED: expected exactly __capstone_hostcall and domain_main undefined, got: ${und:-<none>}"; exit 1; }
rm -f "$OUT/nohostcall.dom"
echo "control fired: without hostcall.o exactly __capstone_hostcall (the libc's way out) and domain_main (start-musl's call) are missing"

for n in 1 2 3 4 5; do
  python3 - "$OUT/tshark_m$n.dom" "$STACK" <<'PY'
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
( cd "$OUT" && sha256sum tshark_m*.dom ) > "$OUT/SHA256SUMS"
