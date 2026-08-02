#!/usr/bin/env bash
# Build Row 7 — RUSTSEC-2022-0070 (secp256k1 preallocated-context UAF).
#
# NO AddressSanitizer, deliberately. This row's instrument is valgrind, for the
# same structural reason as Row 3: the stale dereference executes inside
# libsecp256k1, C code compiled by the `cc` crate and therefore uninstrumented.
# ASan sees the Rust-side free but has nothing to check at the read, so an ASan
# build exits 0 and reads as "no defect". Verified, not assumed -- see
# evidence.txt.
#
# The version pin is exact and load-bearing: the advisory range is
# >=0.24.0,<0.24.2, and cargo silently resolves "0.24.1" to a PATCHED 0.24.3
# where the borrow checker rejects the reproducer outright. 0.24.1, 0.23.4 and
# 0.22.1 are yanked, so 0.24.0 is the only fetchable vulnerable release.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

command -v cargo >/dev/null || { echo "error: cargo not found" >&2; exit 1; }
command -v valgrind >/dev/null || {
  echo "error: valgrind not found -- it is this row's instrument, not optional" >&2
  exit 1; }

echo "=== Building (debug, no sanitizer; valgrind is the instrument) ==="
RUSTFLAGS="-g" cargo build

echo "=== Build Complete ==="
