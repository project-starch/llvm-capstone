#!/usr/bin/env bash
# Fetch nginx at a pinned version, verified by hash. Same shape as the MicroPython and SQLite
# ports: the tree lives outside the repository, what belongs to this port lives inside it.
#
# The version is the one the paper's A1 survey measured, so the blindspot figure that motivates
# this port and the source this port reads are the same nginx and not two of them.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAPSTONE_TMP_ROOT=${CAPSTONE_TMP_ROOT:-/tmp/capstone}
NGX_VERSION=${NGX_VERSION:-1.28.0}
NGX_SHA256=${NGX_SHA256:-c6b5c6b086c0df9d3ca3ff5e084c1d0ef909e6038279c71c1c3e985f576ff76a}
NGX_SRC_DIR=${NGX_SRC_DIR:-$CAPSTONE_TMP_ROOT/nginx-$NGX_VERSION}

mkdir -p "$CAPSTONE_TMP_ROOT"
TARBALL=$CAPSTONE_TMP_ROOT/nginx-$NGX_VERSION.tar.gz

if [[ ! -f $TARBALL ]]; then
  curl -sS -L -o "$TARBALL" "https://nginx.org/download/nginx-$NGX_VERSION.tar.gz"
fi

# Verified rather than trusted: a tarball that changed under the pin would otherwise turn into a
# port of a different allocator without anything saying so.
got=$(sha256sum "$TARBALL" | cut -d' ' -f1)
if [[ $got != "$NGX_SHA256" ]]; then
  echo "nginx-$NGX_VERSION.tar.gz hashes $got, expected $NGX_SHA256" >&2
  exit 1
fi

[[ -d $NGX_SRC_DIR ]] || tar -C "$CAPSTONE_TMP_ROOT" -xzf "$TARBALL"
echo "nginx at $NGX_SRC_DIR ($NGX_VERSION, hash verified)"
