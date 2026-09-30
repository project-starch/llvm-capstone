#!/usr/bin/env bash
# podman wrapper.
#
# Why this exists: this workspace is often driven from a terminal inside the VSCode snap,
# where XDG_DATA_HOME points at ~/snap/code/<revision>/.local/share. Snap carries that
# directory forward across revisions, so the libpod database ends up recording a static
# dir from an older revision and every podman call dies with:
#
#   Error: database static dir "/home/jason/snap/code/255/..." does not match our static
#   dir ".../code/264/...": database configuration mismatch
#
# Pinning XDG_DATA_HOME to the real (non-snap) home store sidesteps it without resetting
# or deleting anything. It is a no-op when XDG_DATA_HOME already points there.
set -euo pipefail
export XDG_DATA_HOME="${CAPSTONE_PODMAN_DATA_HOME:-$HOME/.local/share}"
exec podman "$@"
