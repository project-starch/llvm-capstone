#!/usr/bin/env bash
set -euo pipefail

# Build a first Capstone-compiled SQLite-facing VFS skeleton domain against
# the official SQLite 3.53.1 amalgamation header.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

REPO_ROOT=${CAPSTONE_REPO_ROOT}
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${1:-$TMP_ROOT/capstone-runtime-qemu-share}
OBJ_DIR=${OBJ_DIR:-$TMP_ROOT/sqlite-vfs-skeleton-obj}
SQLITE_VERSION=${SQLITE_VERSION:-3530100}
SQLITE_BASENAME=${SQLITE_BASENAME:-sqlite-amalgamation-$SQLITE_VERSION}
SQLITE_ARCHIVE_URL=${SQLITE_ARCHIVE_URL:-https://sqlite.org/2026/${SQLITE_BASENAME}.zip}
SQLITE_CACHE_ROOT=${SQLITE_CACHE_ROOT:-$TMP_ROOT/$SQLITE_BASENAME}
SQLITE_ARCHIVE=${SQLITE_ARCHIVE:-$TMP_ROOT/${SQLITE_BASENAME}.zip}
SQLITE_SRC_DIR=${SQLITE_SRC_DIR:-$SQLITE_CACHE_ROOT/$SQLITE_BASENAME}

CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
SQLITE_VFS_OPT_LEVEL=${SQLITE_VFS_OPT_LEVEL:--O2}
SQLITE_DOMAIN_OPT_LEVEL=${SQLITE_DOMAIN_OPT_LEVEL:--O2}
SKELETON_DIR="$SCRIPT_DIR/sqlite-vfs-skeleton"
DOMAIN_ENTRY_SRC=${DOMAIN_ENTRY_SRC:-$SKELETON_DIR/sqlite_vfs_skeleton_domain_entry.S}
OUT_DOM="$OUT_DIR/sqlite_vfs_skeleton.dom"

mkdir -p "$TMP_ROOT" "$OUT_DIR" "$OBJ_DIR"

if [[ ! -f "$SQLITE_SRC_DIR/sqlite3.h" ]]; then
  mkdir -p "$SQLITE_CACHE_ROOT"
  if [[ ! -f "$SQLITE_ARCHIVE" ]]; then
    curl -L "$SQLITE_ARCHIVE_URL" -o "$SQLITE_ARCHIVE"
  fi
  python3 - "$SQLITE_ARCHIVE" "$SQLITE_CACHE_ROOT" <<'PY'
import pathlib
import shutil
import sys
import zipfile

archive = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive) as zf:
    zf.extractall(out_dir)
PY
fi

if [[ ! -f "$SQLITE_SRC_DIR/sqlite3.h" ]]; then
  echo "missing sqlite3.h after extraction: $SQLITE_SRC_DIR/sqlite3.h" >&2
  exit 1
fi

"$CLANG" -target capstone64-unknown-elf -ffreestanding -O0 \
  -I"$SKELETON_DIR" \
  -I"$SQLITE_SRC_DIR" \
  -c "$START_SRC" \
  -o "$OBJ_DIR/sqlite_vfs_start.o"

"$CLANG" -target capstone64-unknown-elf -ffreestanding "$SQLITE_VFS_OPT_LEVEL" \
  -I"$SKELETON_DIR" \
  -I"$SQLITE_SRC_DIR" \
  -c "$SKELETON_DIR/capstone_sqlite_vfs.c" \
  -o "$OBJ_DIR/capstone_sqlite_vfs.o"

"$CLANG" -target capstone64-unknown-elf -ffreestanding "$SQLITE_DOMAIN_OPT_LEVEL" \
  -I"$SKELETON_DIR" \
  -I"$SQLITE_SRC_DIR" \
  -c "$SKELETON_DIR/sqlite_vfs_skeleton_domain.c" \
  -o "$OBJ_DIR/sqlite_vfs_skeleton_domain.o"

"$CLANG" -target capstone64-unknown-elf -ffreestanding \
  -I"$SKELETON_DIR" \
  -I"$SQLITE_SRC_DIR" \
  -c "$DOMAIN_ENTRY_SRC" \
  -o "$OBJ_DIR/sqlite_vfs_skeleton_domain_entry.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/sqlite_vfs_start.o" \
  "$OBJ_DIR/capstone_sqlite_vfs.o" \
  "$OBJ_DIR/sqlite_vfs_skeleton_domain.o" \
  "$OBJ_DIR/sqlite_vfs_skeleton_domain_entry.o"

"$LLVM_READOBJ" -h "$OUT_DOM"

echo "Built $OUT_DOM"



