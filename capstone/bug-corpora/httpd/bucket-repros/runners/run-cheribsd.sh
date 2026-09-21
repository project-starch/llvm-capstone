#!/usr/bin/env bash
# The same eight cases under stock CheriBSD purecap with libc heap revocation
# turned on. This asks one question: does the mechanism that protects malloc
# see anything here? The guest default is left alone -- unlike the PoisonCap
# runs, nothing is disabled to make room for a measurement.
#
# The ABI probe runs first in the same guest and prints CheriBSD's own
# malloc_revoke_enabled(), so "the revoker was on" is read off the run rather
# than assumed from the flag.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$HERE/..
REPO=$(cd -- "$ROOT/../../../.." && pwd)
CENSUS=${APR_BUCKETS_CENSUS_DIR:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/apr-buckets-census}
BUILD=${1:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/apr-buckets-cheri}
OUT=${2:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/apr-buckets-cheri-run}
: "${CHERI_SDK:?set CHERI_SDK to the CHERI SDK directory}"
: "${CHERI_SYSROOT:?set CHERI_SYSROOT to the purecap rootfs}"
: "${CHERI_IMAGE:?set CHERI_IMAGE to the purecap disk image}"

if [ ! -f "$CENSUS/apr_buckets_alloc.c" ]; then
  echo "CONTROL-FAILED no census output at $CENSUS (bash capstone/ports/apr/build-buckets-census.sh)" >&2
  exit 75
fi
mkdir -p "$BUILD"
CC=$CHERI_SDK/bin/clang
TARGET=(--target=riscv64-unknown-freebsd13 -march=rv64imafdcxcheri -mabi=l64pc128d
        -mno-relax --sysroot="$CHERI_SYSROOT")

# The probe is built here from the shared source rather than borrowed from
# another component's build tree, so this runner depends on no other port.
"$CC" "${TARGET[@]}" -O1 -o "$BUILD/cheribsd-abi-probe" \
  "$REPO/capstone/ports/common/host/cheribsd/abi-probe.c" \
  || { echo "CONTROL-FAILED build abi-probe" >&2; exit 75; }

for dir in "$ROOT"/[0-9][0-9]_*; do
  name=$(basename "$dir")
  "$CC" "${TARGET[@]}" -O1 -g -o "$BUILD/$name" "$dir/case.c" \
    "$ROOT/shared/driver.c" "$ROOT/shared/apr-stubs.c" "$ROOT/shared/count-free.c" \
    "$CENSUS/apr_buckets_alloc.c" "$CENSUS/apr_pools.c" \
    -I"$ROOT/shared" -I"$CENSUS" || { echo "CONTROL-FAILED build $name" >&2; exit 75; }
done

# The shared runner matches a case's `expect` against whole lines. Each verdict
# line carries its reason after the marker, so the marker goes in as a regex and
# the arm line is a second condition: a case that printed the right verdict for
# the wrong arm does not pass.
"${PYTHON:-python3}" - "$ROOT" "$BUILD" <<'PY'
import json, pathlib, sys
root, build = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
cases = []
for d in sorted(root.glob("[0-9][0-9]_*")):
    n = int(d.name.split("_", 1)[0])
    program = str(build / d.name)
    cases.append(dict(name=f"bucket-{n}-fixed", program=program, args=["fixed", str(n)],
                      expect_regex="VERDICT FIXED .*", also_expect=[f"case={n} arm=fixed"]))
    cases.append(dict(name=f"bucket-{n}-buggy", program=program, args=[str(n)],
                      expect_regex="VERDICT DEFECT-REPRODUCED .*", also_expect=[f"case={n} arm=buggy"]))
(build / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
PY
[ $? -eq 0 ] || { echo "CONTROL-FAILED could not write cases.json" >&2; exit 75; }

exec "${PYTHON:-python3}" "$REPO/capstone/ports/common/host/cheribsd/run.py" "$OUT" \
  --sdk "$CHERI_SDK" --rootfs "$CHERI_SYSROOT" --image "$CHERI_IMAGE" \
  --cases "$BUILD/cases.json" \
  --abi-probe "$BUILD/cheribsd-abi-probe" \
  --runtime-revocation on
