# Shared release pin for the original scripts and the defect corpus.
# CMake reads the same manifest directly. Source this file; do not execute it.
PG_PIN_FIELDS=$(python3 - "$(dirname -- "${BASH_SOURCE[0]}")/memory-contexts/upstream.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    pin = json.load(stream)
print(pin["version"], pin["url"], pin["sha256"])
PY
) || return 1
read -r PG_PIN_VERSION PG_URL PG_SHA256 <<< "$PG_PIN_FIELDS"
unset PG_PIN_FIELDS
if [[ -n ${PG_VERSION:-} && $PG_VERSION != "$PG_PIN_VERSION" ]]; then
    echo "PostgreSQL version mismatch: PG_VERSION=$PG_VERSION, pinned $PG_PIN_VERSION; update upstream.json and its patches together" >&2
    return 1
fi
PG_VERSION=$PG_PIN_VERSION

pg_verify_archive() {
    local actual
    actual=$(sha256sum -- "$1") || return 1
    if [[ ${actual%% *} != "$PG_SHA256" ]]; then
        echo "PostgreSQL archive checksum mismatch: $1" >&2
        return 1
    fi
}
