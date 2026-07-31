#!/usr/bin/env bash
# Minimal runner: transfer the controller + ONE image to the board target, run it.
# One image per clean boot -- a second domain at the same entry VA within one boot
# hangs regardless of this bug.
set -euo pipefail
IMG=${1:?usage: run.sh <delin|nop>}
case "$IMG" in delin|nop) ;; *) echo "arg must be 'delin' or 'nop'" >&2; exit 1;; esac
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
echo "transfer these two files to the target, then run the controller:"
echo "  $D/images/ladder_perf_ctl"
echo "  $D/images/$IMG.dom"
echo
echo "on the target:  ./ladder_perf_ctl $IMG.dom"
echo "expected: nop -> a result line with retval=9 ; delin -> no output (wedge)"
