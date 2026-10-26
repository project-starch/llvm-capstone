#!/usr/bin/env bash
# Run a command in the platform's build container (the image build-platform-docker.sh builds from
# the Dockerfile beside this script), as the invoking user, with the platform directory and this
# repository mounted at their own paths. PoisonCap's toolchain links against the container's
# libraries (its ld.lld needs Ubuntu 24.04's libxml2.so.2), so it runs there, not on the host.
#
#   bash in-container.sh COMMAND [ARG ...]        (working directory: POISONCAP_PLATFORM)
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
P=${POISONCAP_PLATFORM:-$HOME/poisoncap-platform}
P=$(cd "$P" && pwd)
TAG=poisoncap-build:ubuntu24.04
docker image inspect "$TAG" > /dev/null || docker build -q -t "$TAG" -f "$HERE/Dockerfile" "$HERE" > /dev/null
exec docker run --rm -u "$(id -u):$(id -g)" -v "$P:$P" -v "$REPO:$REPO" -w "$P" \
  -e HOME="$P/home" -e POISONCAP_PLATFORM="$P" "$TAG" "$@"
