#!/usr/bin/env bash
# Build the dependency image. The image holds NO project source -- llvm-capstone is
# bind-mounted at run time by run.sh, so builds land in the host working copy.
#
#   ./build-image.sh
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${CAPSTONE_IMAGE:-capstone-build}"

echo "==> building image $IMAGE"
"$HERE/pod.sh" build -t "$IMAGE" -f "$HERE/Containerfile" "$HERE"

echo "==> toolchain in the image"
"$HERE/run.sh" bash -c '
  gcc --version | head -1
  cmake --version | head -1
  echo "ninja $(ninja --version)"
  python3 --version
  ld.lld --version | head -1
  ccache --version | head -1
'
echo "==> done. Next: ./run.sh capstone/container/setup.sh"
