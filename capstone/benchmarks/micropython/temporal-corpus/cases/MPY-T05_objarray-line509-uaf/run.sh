#!/usr/bin/env bash
# MPY-T05: measured by building the fix commit's PARENT with our port, because the
# defect is fixed at the pin and cannot be shown in a domain built from it.
#
# The work is shared with the other row that the same commit fixes, so the runner
# lives in one place rather than being duplicated here.
exec bash "$(git rev-parse --show-toplevel)/capstone/benchmarks/micropython/temporal-corpus/backport-2024/run.sh" "$@"
