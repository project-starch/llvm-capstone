#!/usr/bin/env bash
# Two spellings, one script.  The shared runner (run-beebs-simple-common.sh)
# derives build-script names from BEEBS_BENCHMARK, which is the BEEBS source
# directory name `janne_complex` (underscore); the documentation and
# run-all-beebs.sh use the hyphenated form every other benchmark has.  The
# implementation lives under the underscore name; this file only forwards.
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/build-beebs-janne_complex-capstone.sh" "$@"
