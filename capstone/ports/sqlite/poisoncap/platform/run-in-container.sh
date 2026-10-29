#!/usr/bin/env bash
# Run a command in the platform's build container (Ubuntu 24.04, the Dockerfile beside this script) as
# the invoking user, with the platform directory and this repository mounted at their own paths.
# PoisonCap's toolchain links against the container's libraries (its ld.lld needs Ubuntu 24.04's
# libxml2.so.2), so it runs there, not on the host.
#
#   bash run-in-container.sh [--rebuild] [COMMAND [ARG ...]]     (working directory: POISONCAP_PLATFORM)
#
# The image is built when it is missing; --rebuild builds it again first and prints its id.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
P=${POISONCAP_PLATFORM:-$HOME/poisoncap-platform}
P=$(cd "$P" && pwd)
TAG=poisoncap-build:ubuntu24.04
if [ "${1:-}" = --rebuild ]; then
  shift
  docker build -q -t "$TAG" -f "$HERE/Dockerfile" "$HERE"
elif ! docker image inspect "$TAG" > /dev/null 2>&1; then
  docker build -q -t "$TAG" -f "$HERE/Dockerfile" "$HERE" > /dev/null
fi
[ $# -gt 0 ] || exit 0
exec docker run --rm -u "$(id -u):$(id -g)" -v "$P:$P" -v "$REPO:$REPO" -w "$P" \
  -e HOME="$P/home" -e POISONCAP_PLATFORM="$P" "$TAG" "$@"
