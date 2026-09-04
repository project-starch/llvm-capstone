#!/usr/bin/env bash
# Forwarder. The implementation lives under the UNDERSCORE name because
# run-beebs-simple-common.sh:20-21 computes it from BEEBS_BENCHMARK, which is the
# BEEBS source-directory name (janne_complex). This hyphenated name is what the
# docs and the sibling scripts use, so both spellings must keep working.
exec bash "$(dirname "$0")/build-beebs-janne_complex-host.sh" "$@"
