#!/usr/bin/env bash
# Run build-platform.sh inside the pinned Ubuntu 24.04 build container (Dockerfile beside this
# script), as the invoking user, with the platform directory mounted at the same path.
#
#   bash build-platform-docker.sh [stage ...]      (stages as build-platform.sh)
#
# POISONCAP_PLATFORM (default ~/poisoncap-platform) is the directory build-platform.sh uses; the
# repository checkout this script lives in must be inside it or mounted too, so it is mounted at
# its own path as well.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
P=${POISONCAP_PLATFORM:-$HOME/poisoncap-platform}
mkdir -p "$P/logs" "$P/home"
P=$(cd "$P" && pwd)
TAG=poisoncap-build:ubuntu24.04
docker build -q -t "$TAG" -f "$HERE/Dockerfile" "$HERE" > "$P/logs/docker-image.txt"
docker run --rm -u "$(id -u):$(id -g)" -v "$P:$P" -v "$REPO:$REPO" -e HOME="$P/home" \
  "$TAG" dpkg -l > "$P/logs/container-packages.txt"
exec docker run --rm -u "$(id -u):$(id -g)" -v "$P:$P" -v "$REPO:$REPO" -w "$P" \
  -e HOME="$P/home" -e POISONCAP_PLATFORM="$P" -e POISONCAP_HOST_COMPAT=0 \
  -e POISONCAP_REAL_CC=clang-18 -e POISONCAP_REAL_CXX=clang++-18 \
  -e BUILD_JOBS="${BUILD_JOBS:-64}" \
  "$TAG" bash "$HERE/build-platform.sh" "$@"
