#!/usr/bin/env bash
# Build PoisonCap's own speedtest1 the way their toolchain does: cheribuild's `sqlite` target
# (pycheribuild/projects/cross/sqlite.py at the pinned cheribuild) with --sqlite/build-speedtest,
# over their source tree exactly as published. Nothing of theirs is edited.
#
#   bash in-container.sh bash .../build-speedtest1.sh VARIANT   (cheribuild | memsys5 | memsys5-all32)
#
# The source tree: the artifact's sqlite/ is their fork CTSRD-CHERI/sqlite at a879a48e (its manifest
# says so) with two files changed and not committed, src/mem5.c and test/speedtest1.c. So the tree is
# that commit, checked out, with the artifact's two files laid over it; the script checks that the
# result differs from the artifact's sqlite/ in nothing but the two manifest files cheribuild
# regenerates, and after the build that it differs in nothing at all. A git checkout is needed
# because cheribuild runs create-fossil-manifest, which requires one.
#
# Variants, each in its own output directory under $P/speedtest1/:
#   cheribuild  cheribuild's recipe exactly. It does not enable memsys5, and their main() needs it
#               (SQLITE_CONFIG_HEAP returns SQLITE_ERROR without it), so this one is expected to stop.
#   memsys5     the same plus --sqlite/configure-options=--enable-memsys5, configure's own switch,
#               the least that lets their main() run. Which way they enabled memsys5 is not published.
#   memsys5-all32  as memsys5, but test/speedtest1.c is their fork's own at a879a48e (all 32 main
#               tests active, the same 256 MiB memsys5 heap) instead of the artifact's, in which 12
#               tests are commented out. mem5.c is the artifact's. Both files are theirs, unedited.
set -euo pipefail
VARIANT=${1:?cheribuild, memsys5 or memsys5-all32}
P=${POISONCAP_PLATFORM:?run through in-container.sh}
W=$P/work
SRC=$W/source/sqlite
ART=$W/artifact.tar.gz
TOP=$(tar tzf "$ART" | head -1 | cut -d/ -f1)   # the archive's one top-level directory
COMMIT=a879a48eed98af51ee0719e52bf5ecc0f6baeced
OUT=$P/speedtest1/$VARIANT
case $VARIANT in
  cheribuild) EXTRA=() ;;
  memsys5|memsys5-all32) EXTRA=(--sqlite/configure-options=--enable-memsys5) ;;
  *) echo "unknown variant $VARIANT" >&2; exit 2 ;;
esac
rm -rf "$OUT"; mkdir -p "$OUT"

# The published tree, for reference.
REF=$(mktemp -d)
tar xzf "$ART" -C "$REF" --strip-components=2 "$TOP/sqlite"

# Their fork at the manifest's commit, the artifact's two changed files over it.
if [ "$(git -C "$SRC" rev-parse HEAD 2>/dev/null)" != "$COMMIT" ]; then
  rm -rf "$SRC"; mkdir -p "$SRC"
  git -C "$SRC" init -q
  git -C "$SRC" remote add origin https://github.com/CTSRD-CHERI/sqlite.git
  git -C "$SRC" -c fetch.writeCommitGraph=false fetch -q --depth 1 origin "$COMMIT"
  git -C "$SRC" checkout -q --detach FETCH_HEAD
fi
git -C "$SRC" checkout -q -- . && git -C "$SRC" clean -qfdx
cp "$REF/src/mem5.c" "$SRC/src/mem5.c"
[ "$VARIANT" = memsys5-all32 ] || cp "$REF/test/speedtest1.c" "$SRC/test/speedtest1.c"
before=$(diff -rq --exclude=.git "$SRC" "$REF" || true)
want=$(printf 'Only in %s: manifest\nOnly in %s: manifest.uuid' "$REF" "$REF")
[ "$VARIANT" != memsys5-all32 ] || want=$(printf 'Only in %s: manifest\nOnly in %s: manifest.uuid\nFiles %s/test/speedtest1.c and %s/test/speedtest1.c differ' "$REF" "$REF" "$SRC" "$REF")
[ "$before" = "$want" ] \
  || { echo "source tree differs from the artifact beyond the expected files:"; echo "$before"; exit 3; } >&2

python3 "$P/cheribuild/cheribuild.py" sqlite-riscv64-purecap --skip-update --sqlite/clean \
  --source-root "$W/source" --build-root "$W/build" --tools-root "$W" --output-root "$W/output" \
  --sqlite/source-directory "$SRC" --sqlite/install-directory "$OUT/install" \
  --sqlite/build-speedtest "${EXTRA[@]}" > "$OUT/cheribuild.log" 2>&1 || { tail -30 "$OUT/cheribuild.log"; exit 4; }

# After the build, cheribuild's create-fossil-manifest has written the manifests: the tree must now
# equal the published one, file for file.
after=$(diff -rq --exclude=.git "$SRC" "$REF" || true)
want=""
[ "$VARIANT" != memsys5-all32 ] || want=$(printf 'Files %s/test/speedtest1.c and %s/test/speedtest1.c differ' "$SRC" "$REF")
[ "$after" = "$want" ] || { echo "after the build the tree differs from the artifact:"; echo "$after"; exit 5; } >&2
rm -rf "$REF"

B=$W/build/sqlite-riscv64-purecap-build
cp "$B/speedtest1" "$OUT/speedtest1"
grep -m1 '^ *\$ .*configure' "$B/config.log" > "$OUT/configure-command.txt" || true
grep -E '^(OPT_FEATURE_FLAGS|TCC|CFLAGS|TCC_EXTRA|TLIBS) *=' "$B/Makefile" > "$OUT/makefile-vars.txt" || true
sha256sum "$OUT/speedtest1" "$SRC/src/mem5.c" "$SRC/test/speedtest1.c" > "$OUT/SHA256SUMS"
echo "built $VARIANT: $(cut -c1-16 "$OUT/SHA256SUMS" | head -1)"
