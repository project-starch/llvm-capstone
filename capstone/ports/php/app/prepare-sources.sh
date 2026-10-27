#!/usr/bin/env bash
# Prepare PHP 5.0.0 for the virtual build (build-virtual.sh realphp) under <root>:
#   <root>/cache/php-5.0.0.tar.gz   the release, pinned by SHA-256
#   <root>/php-5.0.0                the release unpacked, configured for the cli SAPI with every
#                                   extension off, with the TREE patches of patches/ applied
#   <root>/php-5.0.0/PROVENANCE.txt what this tree is, and patchset.sha256 which patches made it
#
# This stands where the collaborator's /corpus/build/php-5.0.0 stood: an already-configured
# tree whose Makefile is the one source of truth for the translation units (build-virtual.sh
# reads PHP_GLOBAL_OBJS and PHP_CLI_OBJS from it) and whose main/php_config.h holds the
# configure answers the build corrects. Configured here natively, as the corpus did; the build
# ignores the Makefile's compiler and flags and uses only its TU list.
#
# The result is deterministic in its inputs: tarball hash, configure line, patch set. A root
# prepared with a different patch set is refused rather than reused, as the PostgreSQL port
# does, so a stale tree cannot be built silently.
set -euo pipefail
ROOT=${1:?usage: prepare-sources.sh <root>}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERSION=5.0.0
URL=https://museum.php.net/php5/php-${VERSION}.tar.gz
# The museum publishes no checksum for this release. This value was taken 2026-10-27 from a
# download whose unpacked tree equals the php-src tag php-5.0.0 in every file the build compiles,
# up to CVS $Id$ keywords and re2c #line directives (the tarball is the release; the tag is the
# repository). The size is 5,595,997 bytes.
SHA256=5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919
CACHE=${PHP_500_CACHE_ROOT:-$ROOT/cache}
TARBALL=$CACHE/php-${VERSION}.tar.gz
TREE=$ROOT/php-${VERSION}
mkdir -p "$ROOT" "$CACHE"

# ---- the tree patches: everything in patches/ except the seam, which is a build MODE -------
# (applied by build-virtual.sh to a copy of zend_alloc.c, seam arms only; see patches/README.md)
TREE_PATCHES=()
for p in "$SCRIPT_DIR"/patches/php-${VERSION}-*.patch; do
  case "$p" in *-zend-alloc-sublet-hooks.patch) continue ;; esac
  TREE_PATCHES+=("$p")
done
[ "${#TREE_PATCHES[@]}" -gt 0 ] || { echo "no tree patches under $SCRIPT_DIR/patches" >&2; exit 2; }
PATCH_HASH=$({ for p in "${TREE_PATCHES[@]}"; do printf '%s ' "$(basename "$p")"; sha256sum < "$p"; done; } \
             | sha256sum | cut -d' ' -f1)

if [ -d "$TREE" ]; then
  if [ "$(cat "$TREE/patchset.sha256" 2>/dev/null)" = "$PATCH_HASH" ] && [ -f "$TREE/Makefile" ]; then
    echo "already prepared: $TREE" >&2
    echo "$TREE"
    exit 0
  fi
  echo "$TREE exists but was prepared with another patch set (or never finished); use a fresh root" >&2
  exit 2
fi

# ---- the release ------------------------------------------------------------------------
if [ ! -f "$TARBALL" ]; then
  echo "fetching $URL" >&2
  curl -fsSL "$URL" -o "$TARBALL.part" && mv "$TARBALL.part" "$TARBALL"
fi
echo "$SHA256  $TARBALL" | sha256sum -c - >/dev/null \
  || { echo "php-${VERSION}.tar.gz: SHA-256 mismatch (expected $SHA256)" >&2; exit 2; }
tar xzf "$TARBALL" -C "$ROOT"
[ -d "$TREE/Zend" ] || { echo "the archive did not unpack to $TREE" >&2; exit 2; }

# ---- the tree patches, in name order ------------------------------------------------------
for p in "${TREE_PATCHES[@]}"; do
  patch -d "$TREE" -p1 -N -s < "$p" || { echo "$(basename "$p") does not apply" >&2; exit 2; }
done

# ---- configure --------------------------------------------------------------------------
# --disable-all: the cli SAPI and ext/standard only, which is the 145-TU tree the port was
# measured on (ext/standard 65, Zend 39, main 23, main/streams 9, regex 4, TSRM 3, sapi/cli 2).
# -std=gnu89 is for configure's OWN conftests: a current gcc defaults to C23 and rejects the
# 2004 `main(){...}` probe as an implicit int, which reads as "C compiler cannot create
# executables". The flag reaches the Makefile's CFLAGS, which this port never uses.
(cd "$TREE" && CFLAGS="-std=gnu89" ./configure --disable-all > configure.log 2>&1) \
  || { echo "configure failed; see $TREE/configure.log" >&2; tail -5 "$TREE/configure.log" >&2; exit 2; }
for f in Makefile main/php_config.h main/build-defs.h Zend/zend_config.h TSRM/tsrm_config.h; do
  [ -f "$TREE/$f" ] || { echo "configure left no $f" >&2; exit 2; }
done
# The TU list is the whole point of the configured tree; a different count is a different port.
TUS=$(cd "$TREE" && make -s -p -n 2>/dev/null | grep -E '^(PHP_GLOBAL_OBJS|PHP_CLI_OBJS) =' \
      | tr ' ' '\n' | grep -c '\.lo$' || true)
[ "$TUS" = 145 ] || { echo "the configured tree lists $TUS translation units, expected 145" >&2; exit 2; }

# ---- the record -------------------------------------------------------------------------
echo "$PATCH_HASH" > "$TREE/patchset.sha256"
{
  echo "PHP ${VERSION}, prepared by capstone/ports/php/app/prepare-sources.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "tarball: $URL  sha256 $SHA256"
  echo "configure: CFLAGS=-std=gnu89 ./configure --disable-all  (145 translation units)"
  echo "tree patches, in order:"
  for p in "${TREE_PATCHES[@]}"; do echo "  $(basename "$p")  $(sha256sum < "$p" | cut -c1-16)"; done
  echo "patch set: $PATCH_HASH"
  echo "The seam patch (0003) is not in this tree; build-virtual.sh applies it to a copy of"
  echo "Zend/zend_alloc.c for the seam arms only."
} > "$TREE/PROVENANCE.txt"
echo "prepared: $TREE ($TUS translation units)" >&2
echo "$TREE"
