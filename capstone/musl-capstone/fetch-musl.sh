#!/usr/bin/env bash
# Download and verify the official musl release archive. musl is NOT vendored,
# same policy as fetch-sqlite.sh: the upstream tree stays immutable under
# $CAPSTONE_TMP_ROOT and our changes live in this directory as an overlay.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../tests/capstone-test-env.sh"

MUSL_VERSION=${MUSL_VERSION:-1.2.5}
# SHA-256 is what the musl ecosystem publishes for its releases, so this pin is
# checkable against a third party (distro package recipes), not just against us.
MUSL_ARCHIVE_SHA256=${MUSL_ARCHIVE_SHA256:-a9a118bbe84d8764da0ea0d28b3ab3fae8477fc7e4085d90102b8596fc7c75e4}
MUSL_BASENAME="musl-$MUSL_VERSION"
MUSL_CACHE_ROOT=${MUSL_CACHE_ROOT:-$CAPSTONE_TMP_ROOT/musl-src}
MUSL_ARCHIVE=${MUSL_ARCHIVE:-$MUSL_CACHE_ROOT/$MUSL_BASENAME.tar.gz}
MUSL_SRC_DIR=${MUSL_SRC_DIR:-$MUSL_CACHE_ROOT/$MUSL_BASENAME}
MUSL_URL=${MUSL_URL:-https://musl.libc.org/releases/$MUSL_BASENAME.tar.gz}

mkdir -p "$MUSL_CACHE_ROOT"

if [[ ! -f "$MUSL_ARCHIVE" ]]; then
  curl -L "$MUSL_URL" -o "$MUSL_ARCHIVE"
fi

python3 - "$MUSL_ARCHIVE" "$MUSL_ARCHIVE_SHA256" "$MUSL_CACHE_ROOT" "$MUSL_SRC_DIR" <<'PY'
import hashlib
import pathlib
import sys
import tarfile

archive = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
cache_root = pathlib.Path(sys.argv[3])
src_dir = pathlib.Path(sys.argv[4])

actual = hashlib.sha256(archive.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"musl archive SHA-256 mismatch: expected {expected}, got {actual}")

if not (src_dir / "Makefile").is_file():
    with tarfile.open(archive) as source:
        source.extractall(cache_root)
PY

for required in Makefile src/internal/syscall.h arch/riscv64/syscall_arch.h; do
  [[ -e "$MUSL_SRC_DIR/$required" ]] || {
    echo "missing musl source file: $MUSL_SRC_DIR/$required" >&2
    exit 1
  }
done

printf '%s\n' "$MUSL_SRC_DIR"
