#!/usr/bin/env bash
# Run build-platform.sh in the pinned Ubuntu 24.04 build container (in-container.sh), after rebuilding
# the image from the Dockerfile beside this script and recording its packages.
#
#   bash build-platform-docker.sh [stage ...]      (stages as build-platform.sh)
#
# POISONCAP_PLATFORM (default ~/poisoncap-platform) is the directory build-platform.sh uses.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
P=${POISONCAP_PLATFORM:-$HOME/poisoncap-platform}
mkdir -p "$P/logs" "$P/home"
docker build -q -t poisoncap-build:ubuntu24.04 -f "$HERE/Dockerfile" "$HERE" > "$P/logs/docker-image.txt"
bash "$HERE/in-container.sh" dpkg -l > "$P/logs/container-packages.txt"
exec bash "$HERE/in-container.sh" env BUILD_JOBS="${BUILD_JOBS:-64}" bash "$HERE/build-platform.sh" "$@"
