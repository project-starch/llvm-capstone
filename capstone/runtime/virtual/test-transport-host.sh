#!/usr/bin/env bash
# Native Linux contract against the actual exec.c (test-transport-host.c); an optional argument
# selects another exec.c (the negative control: origin/dev's, which reports overlap=1).
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
src=$(realpath "${1:-$here/exec.c}")
out=$(mktemp -d "${TMPDIR:-/tmp}/transport-host.XXXXXX")
trap 'rm -rf "$out"' EXIT
${CC:-gcc} -O1 -g -pthread -I"$here" -I"$here/../include" -DEXEC_C="\"$src\"" \
    "$here/test-transport-host.c" \
    "$here/../linux/"{application-image,image-hash,delegate-service,spawner,signals,park,domain-fault}.c \
    "$here/../common/"{launch,delegate,spawn,msghdr}.c -o "$out/test"
rc=0
"$out/test" || rc=1
"$out/test" legacy || rc=1
exit $rc
