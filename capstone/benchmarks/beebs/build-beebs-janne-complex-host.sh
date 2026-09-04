#!/usr/bin/env bash
# Forwarder: see build-beebs-janne-complex-capstone.sh for why two spellings
# of this benchmark's scripts exist.  The implementation is the underscore one.
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/build-beebs-janne_complex-host.sh" "$@"
