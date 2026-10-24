#!/bin/bash
# Build one SPEC src.alt package from the kit's authoring files.
# usage: make-srcalt.sh SPEC_ROOT KIT_DIR BENCH OUTDIR
#   SPEC_ROOT  a pristine, verified SPEC17 tree (spec-extract.py); used only to
#              diff against, then left with the src.alt directory in it
#   KIT_DIR    this kit (srcalt/BENCH/{README,files/})
# Writes OUTDIR/<package> and OUTDIR/<package>.sha256; fails closed.
set -euo pipefail
mkdir -p "$4"
SPEC_ROOT=$(cd "$1" && pwd) KIT=$(cd "$2" && pwd) B=$3 OUT=$(cd "$4" && pwd) NAME=capstone
A=$SPEC_ROOT/benchspec/CPU/$B/src/src.alt/$NAME
[ -d "$KIT/srcalt/$B/files" ] && [ -f "$KIT/srcalt/$B/README" ] || { echo "make-srcalt: no authoring files for $B"; exit 1; }
[ -e "$A" ] && { echo "make-srcalt: $A exists"; exit 1; }
mkdir -p "$A" "$OUT"
cp -a "$KIT/srcalt/$B/files/." "$A/"
cp "$KIT/srcalt/$B/README" "$A/README"
# every authoring file must replace an existing source file (no additions)
(cd "$KIT/srcalt/$B/files" && find . -type f) | while read -r f; do
  [ -f "$SPEC_ROOT/benchspec/CPU/$B/src/$f" ] || { echo "make-srcalt: $f is not a $B source file"; exit 1; }
done
cd "$SPEC_ROOT"
set +u; source ./shrc > /dev/null; set -u
makesrcalt --usewith 1.0.2 "$B" "$NAME" > "$OUT/makesrcalt-$B.log" 2>&1
pkg=$(find "$SPEC_ROOT" -maxdepth 3 -name "$B.$NAME.*.tar.xz" -newer "$A/README" | head -1)
[ -n "$pkg" ] || { echo "make-srcalt: makesrcalt produced no package"; cat "$OUT/makesrcalt-$B.log"; exit 1; }
cp "$pkg" "$OUT/"
(cd "$OUT" && sha256sum "$(basename "$pkg")" > "$(basename "$pkg").sha256")
echo "make-srcalt: $B -> $OUT/$(basename "$pkg")"
