#!/usr/bin/env bash
# Build the LOOKASIDE lifetime controls (README.md here) for one arm.
# A copy of ../build-controls.sh (the memsys5 controls) with three differences: the controls are
# the ones in this directory, the images are named la-control-*, and the include path adds ..
# for control_common.h. Staging, flags and arms are unchanged.
#   usage: build-controls.sh <virtual SDK dir> <amalgamation dir (sqlite3.c, sqlite3.h)> <out dir> release|protected
#
# The engine source is staged by the security corpus's own builder,
# ports/sqlite/repro322/build-virtual.py (--memsys5 for release, --sublet for protected), unchanged:
# it copies the amalgamation to <out>/stage/src/sqlite3-capstone.c, applies fts5-azarg-patch.py
# and fts3-doclist-null-patch.py, and on the protected arm sublet-3220000-memsys5.patch (-F0).
# It is run with --only 18 (the corpus's memsys5 case), only for that staging.
#
# Each control is then ONE translation unit that #includes the staged sqlite3-capstone.c: the
# controls call sqlite3DbMallocRaw/sqlite3DbFree and read db->lookaside, all internal to the
# amalgamation. The flags are the corpus's for
# a case of the core group on that arm (build-virtual.py), plus -g so a fault pc resolves to a
# probe's file:line. -O0, as the corpus.
set -euo pipefail
SDK=${1:?usage: build-controls.sh <sdk> <amalgamation> <out> release|protected}
AMALG=${2:?amalgamation dir}
OUT=${3:?out dir}
ARM=${4:?release|protected}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../../../.." && pwd)
REPRO=$REPO/capstone/ports/sqlite/repro322
case $ARM in
  release)   ARMFLAG=--memsys5; ARMDEFS=(-DREPRO322_VIRTUAL_MEMSYS5) ;;
  protected) ARMFLAG=--sublet;  ARMDEFS=(-DREPRO322_VIRTUAL_MEMSYS5 -DREPRO322_VIRTUAL_SUBLET) ;;
  *) echo "arm is release or protected, not $ARM" >&2; exit 2 ;;
esac
mkdir -p "$OUT"
python3 "$REPRO/build-virtual.py" --sdk "$SDK" --amalgamation "$AMALG" --out "$OUT/stage" \
  $ARMFLAG --only 18 > "$OUT/stage.log" 2>&1 || { cat "$OUT/stage.log"; exit 1; }
ENGINE=$OUT/stage/src/sqlite3-capstone.c
[[ -f $ENGINE ]] || { echo "no staged engine $ENGINE" >&2; exit 1; }
# The core group's flags, read from the builder's own record of them, so they cannot drift.
mapfile -t CORE < <(python3 -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["groups"]["core"]))' \
  "$OUT/stage/build.json")
CONFIG=(-DHAVE_MALLOC_H=1 -DHAVE_MALLOC_USABLE_SIZE=1 -DSQLITE_ENABLE_MEMSYS5)
build() {  # name source [extra flags]
  local name=$1 src=$2; shift 2
  local cmd=("$SDK/capstone-cc" -O0 -g -DREPRO322_VIRTUAL "${ARMDEFS[@]}" "$@" "${CONFIG[@]}" "${CORE[@]}"
             -I"$REPO/capstone/runtime/include" -I"$HERE" -I"$HERE/.." -I"$REPRO" -I"$OUT/stage/src" -I"$AMALG"
             "$src" -lm -o "$OUT/$name.dom")
  printf '%q ' "${cmd[@]}" > "$OUT/$name.log"; echo >> "$OUT/$name.log"
  "${cmd[@]}" >> "$OUT/$name.log" 2>&1 || { echo "BUILD $name FAILED"; grep -i error "$OUT/$name.log" | head; return 1; }
  echo "BUILD $name OK"
}
build la-control-parent          "$HERE/control-parent/control.c"
build la-control-sibling         "$HERE/control-sibling/control.c"
build la-control-successor       "$HERE/control-successor/control.c"
build la-control-successor-stale "$HERE/control-successor/control.c" -DCONTROL_STALE
(cd "$OUT" && sha256sum stage/src/sqlite3-capstone.c la-control-*.dom > SHA256SUMS)
cat "$OUT/SHA256SUMS"
