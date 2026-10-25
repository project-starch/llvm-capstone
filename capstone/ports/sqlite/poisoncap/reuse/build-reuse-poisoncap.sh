#!/usr/bin/env bash
# PoisonCap's speedtest1 with the reuse-distance wrapper linked in, and the wrapper's test programs,
# all against their sqlite3.c and with their compiler and flags. Their sources are not touched.
#
#   bash in-container.sh bash .../reuse/build-reuse-poisoncap.sh [VARIANT]   (memsys5 | memsys5-spatial)
#
# Starts from build-speedtest1.sh's VARIANT build (default memsys5: their protection on; memsys5-spatial:
# their NESTED_TEMPORAL switch off; it is rerun here, so the build directory is that variant's): cheribuild's configured build directory, whose Makefile gives the compiler, the flags,
# the generated sqlite3.c and the speedtest1 link rule. Then:
#   - reuse-wrap.o, compiled with the Makefile's own $(TCC) (compiler, flags, include paths and the
#     feature defines, -DSQLITE_ENABLE_MEMSYS5 among them), in the build directory;
#   - speedtest1 relinked by the Makefile's own rule, with TLIBS extended by reuse-wrap.o and
#     -Wl,--wrap=sqlite3_config,--wrap=sqlite3_step,--wrap=sqlite3_reset (the old binary is removed
#     first: the rule's prerequisites do not include TLIBS);
#   - test-wrapped, test-traced (-DREUSE_WRAP_TRACE) and test-plain from test-reuse-wrap.c and the
#     same sqlite3.c, for check-reuse-wrap.py on their platform.
# The --wrap routing is checked in the binaries' symbols: __wrap_sqlite3_config defined and the
# program's references to sqlite3_config, sqlite3_step and sqlite3_reset bound to the wrappers.
set -euo pipefail
P=${POISONCAP_PLATFORM:?run through in-container.sh}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VARIANT=${1:-memsys5}
OUT=$P/speedtest1/reuse-$VARIANT
bash "$HERE/../build-speedtest1.sh" "$VARIANT"
B=$P/work/build/sqlite-riscv64-purecap-build
rm -rf "$OUT"; mkdir -p "$OUT"
# The Makefile's own compile command (TCC) and link libraries (TLIBS), as make expands them.
mvar() { make -s -C "$B" -f Makefile -f <(printf 'print-%s:\n\t@echo $(%s)\n' "$1" "$1") "print-$1"; }
read -r -a TCCV <<< "$(mvar TCC)"
TLIBS0=$(mvar TLIBS)
read -r -a LTLINKV <<< "$(mvar LTLINK)"   # libtool link: TCC plus the target and linker flags
WRAP="-Wl,--wrap=sqlite3_config -Wl,--wrap=sqlite3_step -Wl,--wrap=sqlite3_reset"
printf '%s\n' "TCC: ${TCCV[*]}" "LTLINK: ${LTLINKV[*]}" "TLIBS: $TLIBS0" > "$OUT/toolchain.txt"
grep -q -- '-DSQLITE_ENABLE_MEMSYS5' "$OUT/toolchain.txt" || { echo "TCC lacks SQLITE_ENABLE_MEMSYS5" >&2; exit 2; }
cc() { (cd "$B" && "${TCCV[@]}" "$@"); }
link() { (cd "$B" && "${LTLINKV[@]}" "$@"); }
cc -c "$HERE/reuse-wrap.c" -o "$OUT/reuse-wrap.o"
cc -DREUSE_WRAP_TRACE -c "$HERE/reuse-wrap.c" -o "$OUT/reuse-wrap-trace.o"

rm -f "$B/speedtest1"
make -C "$B" speedtest1 TLIBS="$TLIBS0 $OUT/reuse-wrap.o $WRAP" > "$OUT/link.log" 2>&1
cp "$B/speedtest1" "$OUT/speedtest1"

cc -c "$B/sqlite3.c" -o "$OUT/sqlite3.o"
cc -c "$HERE/test-reuse-wrap.c" -o "$OUT/test.o"
cc -c "$HERE/snapshot-stub.c" -o "$OUT/snapshot-stub.o"
link "$OUT/test.o" "$OUT/reuse-wrap.o" "$OUT/sqlite3.o" $WRAP $TLIBS0 -o "$OUT/test-wrapped"
link "$OUT/test.o" "$OUT/reuse-wrap-trace.o" "$OUT/sqlite3.o" $WRAP $TLIBS0 -o "$OUT/test-traced"
link "$OUT/test.o" "$OUT/snapshot-stub.o" "$OUT/sqlite3.o" $TLIBS0 -o "$OUT/test-plain"

NM=$P/work/sdk/bin/llvm-nm
for b in speedtest1 test-wrapped; do
  syms=$("$NM" "$OUT/$b")
  for s in __wrap_sqlite3_config __wrap_sqlite3_step __wrap_sqlite3_reset; do
    grep -q " T $s\$" <<< "$syms" || { echo "$b: $s not defined: the wrap did not link" >&2; exit 3; }
  done
done
sha256sum "$OUT/speedtest1" "$OUT"/test-* "$HERE/reuse-wrap.c" "$HERE/test-reuse-wrap.c" > "$OUT/SHA256SUMS"
echo "reuse build: $(head -1 "$OUT/SHA256SUMS" | cut -c1-16) (speedtest1)"
