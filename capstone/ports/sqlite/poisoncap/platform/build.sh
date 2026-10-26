#!/usr/bin/env bash
# Build the published PoisonCap platform: rebuild the container image, record its id and packages, then
# run build-stages.sh inside it.
#
#   bash build.sh [stage ...]      stages: prepare verify llvm qemu cheribsd image (default: all)
#
# POISONCAP_PLATFORM (default ~/poisoncap-platform) holds work/, logs/ and the pinned cheribuild.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
P=${POISONCAP_PLATFORM:-$HOME/poisoncap-platform}
mkdir -p "$P/logs" "$P/home"
bash "$HERE/run-in-container.sh" --rebuild > "$P/logs/docker-image.txt"
bash "$HERE/run-in-container.sh" dpkg -l > "$P/logs/container-packages.txt"
exec bash "$HERE/run-in-container.sh" env BUILD_JOBS="${BUILD_JOBS:-64}" bash "$HERE/build-stages.sh" "$@"
