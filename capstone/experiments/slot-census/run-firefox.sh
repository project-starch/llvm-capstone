#!/bin/bash
# Slot census of a web browser: Firefox running Speedometer, the browser workload of Mesh (PLDI 2019, Speedometer 2.0)
# and the CHERI/allocator literature. Firefox allocates through its own mozjemalloc, so the LD_PRELOAD tracer cannot
# see it; ffcensus.c is a replace-malloc library (on Linux, Nightly's firefox-bin takes replace_init from a preloaded
# library: MOZ_REPLACE_MALLOC_LIB is the Windows and Android route; ffcensus exports replace_init only) that counts
# every malloc-family and moz_arena_* call in front of mozjemalloc, in every Firefox process. JavaScript objects live
# in SpiderMonkey's garbage-collected heap, outside malloc, so they are hidden from this census ("hidden" class).
#
# usage: run-firefox.sh OUT_DIR [VERSION...]   (default 2.1 3.1)
# Speedometer 2.1 is the bug-fix release of 2.0 (WebKit/Speedometer release/2.1); 3.1 is the current release. Each is
# cloned from WebKit's repository at a recorded commit and served locally. Firefox Nightly (download.mozilla.org's
# latest, its BuildID and source changeset recorded), geckodriver $GECKO and Selenium $SELENIUM drive it headless.
# ffcensus is built against the replace-malloc headers of that Nightly's exact mozilla-central changeset. Controls:
# ffcontrol runs control.c's sequence and an arena sequence through ffcensus (both must pass), and ffcontrol without
# FFCENSUS_DIR must leave no census (the dump must fail). A cell is valid when Speedometer reports its score and the
# census holds the parent and at least one content process.
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=$1; shift
VERSIONS=("$@"); [ ${#VERSIONS[@]} -gt 0 ] || VERSIONS=(2.1 3.1)
GECKO=0.36.0 SELENIUM=4.27.1
PORT=${SLOTCENSUS_SPEEDOMETER_PORT:-18090}
. "$HERE/tracer.sh"
start_pass

# Firefox Nightly and the replace-malloc headers of its exact source changeset
F_T=firefox-nightly.tar.xz G_T=geckodriver-v$GECKO-linux64.tar.gz
curl -fsSL -o "$TOOL/$F_T" "https://download.mozilla.org/?product=firefox-nightly-latest-ssl&os=linux64&lang=en-US" \
  && tar xJf "$TOOL/$F_T" -C "$TOOL" || { echo "run-firefox: firefox download failed" >&2; exit 1; }
[ -s "$CACHE/$G_T" ] || curl -fsSL -o "$CACHE/$G_T" \
  "https://github.com/mozilla/geckodriver/releases/download/v$GECKO/$G_T" || exit 1
tar xzf "$CACHE/$G_T" -C "$TOOL" || exit 1
REV=$(sed -n 's/^SourceStamp=//p' "$TOOL/firefox/platform.ini")
[ -n "$REV" ] || { echo "run-firefox: no SourceStamp in platform.ini" >&2; exit 1; }
I=$TOOL/include; mkdir -p "$I/mozilla"
for f in malloc_decls.h replace_malloc.h replace_malloc_bridge.h mozjemalloc_types.h; do
  curl -fsSL -o "$I/$f" "https://hg.mozilla.org/mozilla-central/raw-file/$REV/memory/build/$f" || exit 1
done
# stand-ins for two Mozilla headers: only the macros the replace-malloc headers use, compiling as C
cat > "$I/mozilla/Types.h" <<'EOF'
#ifndef mozilla_Types_h
#define mozilla_Types_h
#include <stddef.h>
#include <stdint.h>
#define MOZ_EXPORT __attribute__((visibility("default")))
#define MFBT_API MOZ_EXPORT
#define MOZ_BEGIN_EXTERN_C
#define MOZ_END_EXTERN_C
#endif
EOF
cat > "$I/mozilla/MacroArgs.h" <<'EOF'
#ifndef mozilla_MacroArgs_h
#define mozilla_MacroArgs_h
#define MOZ_CONCAT2(a, b) a##b
#define MOZ_CONCAT(a, b) MOZ_CONCAT2(a, b)
#endif
EOF
# Firefox compiles the function table as C++, where malloc_decls.h declares one more entry (moz_may_purge_now, inside
# #ifdef __cplusplus) than a C build sees: without it every later field, the moz_arena_* functions among them, sits one
# slot early and calls the wrong function. The C build keeps that entry, with an int return type (its C++ type is an
# enum without a C typedef), so it is a function pointer in the same slot.
RE='s/#    ifdef __cplusplus\nMALLOC_DECL\(moz_may_purge_now, may_purge_now_result_t,/'
RE+='#    if 1\nMALLOC_DECL(moz_may_purge_now, int,/'
perl -0pi -e "$RE" "$I/malloc_decls.h"
grep -q "MALLOC_DECL(moz_may_purge_now, int," "$I/malloc_decls.h" \
  || { echo "run-firefox: malloc_decls.h changed" >&2; exit 1; }
# and the layout is checked: the compiled table must have exactly one slot per declaration in the header
DECLS=$(grep -cE '^(NOTHROW_)?MALLOC_DECL\(' "$I/malloc_decls.h")
cat > "$TOOL/slots.c" <<'EOF'
#include "replace_malloc.h"
#include <stdio.h>
int main(void) { printf("%zu\n", sizeof(malloc_table_t) / sizeof(void *)); }
EOF
gcc -I"$I" -o "$TOOL/slots" "$TOOL/slots.c" && SLOTS=$("$TOOL/slots") || exit 1
[ "$SLOTS" = "$DECLS" ] || { echo "run-firefox: table has $SLOTS slots, header declares $DECLS" >&2; exit 1; }
echo "malloc_table_t: $SLOTS slots, as declared" > "$OUT/logs/table-layout.txt"
gcc -O2 -fPIC -shared -fvisibility=hidden -I"$I" -o "$TOOL/libffcensus.so" "$HERE/ffcensus.c" -lpthread &&
gcc -O2 -fno-builtin -I"$I" -I"$HERE" -o "$TOOL/ffcontrol" "$HERE/ffcontrol.c" "$HERE/ffcensus.c" -lpthread \
  || { echo "run-firefox: ffcensus build failed" >&2; exit 1; }

# ffcensus's controls
mkdir -p "$OUT/control/ff-c" "$OUT/control/ff-arena" "$OUT/control/ff-negative"
FFCENSUS_DIR=$OUT/control/ff-c "$TOOL/ffcontrol" > "$OUT/control/ff-c.expect"
FFCENSUS_DIR=$OUT/control/ff-arena "$TOOL/ffcontrol" arena > "$OUT/control/ff-arena.expect"
"$TOOL/ffcontrol" > "$OUT/control/ff-negative.expect"
for k in c arena; do
  python3 "$HERE/ffcensus-dump.py" "$OUT/control/ff-$k" > "$OUT/control/ff-$k.jsonl" &&
  python3 "$HERE/check-control.py" "$OUT/control/ff-$k.expect" "$OUT/control/ff-$k.jsonl" > "$OUT/control/ff-$k.check" \
    && echo "control ff-$k: as expected (pass)" \
    || { echo "control ff-$k: NOT as expected"; cat "$OUT/control/ff-$k.check"; exit 75; }
done
if python3 "$HERE/ffcensus-dump.py" "$OUT/control/ff-negative" > /dev/null 2>&1; then
  echo "control ff-negative: NOT as expected (a census without FFCENSUS_DIR)"; exit 75
fi
echo "control ff-negative: as expected (fail)"

# Speedometer, Selenium
python3 -m venv "$TOOL/sel" && "$TOOL/sel/bin/pip" -q install "selenium==$SELENIUM" > "$OUT/logs/pip.log" 2>&1 \
  || { echo "run-firefox: selenium install failed" >&2; exit 1; }
for v in "${VERSIONS[@]}"; do
  git -c advice.detachedHead=false clone -q --depth 1 -b "release/$v" https://github.com/WebKit/Speedometer \
    "$TOOL/speedometer-$v" || exit 1
done
{
  provenance_tracer
  (cd "$TOOL" && sha256sum "$F_T" libffcensus.so) && (cd "$CACHE" && sha256sum "$G_T")
  grep -E '^(Version|BuildID|SourceStamp)=' "$TOOL/firefox/application.ini" | sed 's/^/firefox /'
  echo "replace-malloc headers: mozilla-central $REV memory/build"
  echo "geckodriver $GECKO, selenium $SELENIUM, firefox -headless"
  for v in "${VERSIONS[@]}"; do echo "speedometer $v $(git -C "$TOOL/speedometer-$v" rev-parse HEAD)"; done
} > "$OUT/provenance.txt"

: > "$OUT/cells.tsv"
for v in "${VERSIONS[@]}"; do
  cell=speedometer-$v
  D=$OUT/census-$v; mkdir -p "$D"
  (cd "$TOOL/speedometer-$v" && exec python3 -m http.server "$PORT" --bind 127.0.0.1) > "$OUT/logs/$cell.http" 2>&1 &
  http=$!
  sleep 1
  url="http://127.0.0.1:$PORT/index.html"; [ "$v" = 2.1 ] || url="$url?startAutomatically"
  t=$(date +%s)
  LD_PRELOAD=$TOOL/libffcensus.so FFCENSUS_DIR=$D \
    "$TOOL/sel/bin/python" "$HERE/ff-speedometer.py" "$TOOL/firefox/firefox" "$TOOL/geckodriver" "$url" "$v" \
    "$OUT/logs/$cell" > "$OUT/$cell.out" 2>&1; rc=$?
  kill $http 2>/dev/null; wait $http 2>/dev/null
  python3 "$HERE/ffcensus-dump.py" "$D" > "$OUT/$cell.jsonl"; dump=$?
  procs=$(grep -c . "$OUT/$cell.jsonl"); content=$(grep -c -- "-contentproc" "$OUT/$cell.jsonl")
  v_=INVALID; [ $rc = 0 ] && [ $dump = 0 ] && [ "$content" -ge 1 ] && grep -q '^score ' "$OUT/$cell.out" && v_=valid
  record_cell "$cell" 're:firefox' "$v_" "rc=$rc processes=$procs content=$content $(( $(date +%s) - t ))s"
done
echo CENSUS_DONE >> "$OUT/cells.tsv"
