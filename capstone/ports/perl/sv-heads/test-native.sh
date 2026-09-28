#!/usr/bin/env bash
# Native gates for the SV-head adapter, on the host, with AddressSanitizer.
#
#   bash test-native.sh [--suite]
#
# Builds the pinned Perl 5.36.3 with the port's patches, patch 0001 of this
# directory and -DPERL_SV_HEAD_ADAPTER, linked with native.c; with ASan, a
# released head is poisoned while its reuse is delayed (PERL_SVH_NATIVE_MODE=1),
# so any read of it that bypasses the adapter's record is reported.
#
# Gates, each of which must fire in its positive control:
#   1. the core's unit test (test-sv-heads.c) in both modes;
#   2. the study workload, records.pl 512 3 0, reproduces its oracle in both
#      modes with no ASan report;
#   3. positive control: `$x = *foo; *x = $x` reads a freed head in upstream
#      S_glob_assign_glob; mode 1 must report it, mode 0 must run it.
# --suite also runs upstream's complete test suite in both modes (about 10
# minutes each on 10 jobs) and lists, per mode, the files that failed and the
# first frame of every ASan report.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PORT=$HERE/..
REPO=$(cd -- "$HERE/../../../.." && pwd)
ROOT=${PERL_SVH_NATIVE_ROOT:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/perl-svh-native-gate}
ARCHIVE=${PERL_ARCHIVE:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/perl-src/perl-5.36.3.tar.gz}
ARCHIVE_SHA=f2a1ad88116391a176262dd42dfc52ef22afb40f4c0e9810f15d561e6f1c726a
SUITE=0
[[ ${1:-} == --suite ]] && SUITE=1
[[ ! -e $ROOT ]] || { echo "gate root already exists: $ROOT" >&2; exit 2; }
[[ -f $ARCHIVE ]] || { echo "missing $ARCHIVE" >&2; exit 2; }
[[ $(sha256sum "$ARCHIVE" | cut -d' ' -f1) == "$ARCHIVE_SHA" ]] \
  || { echo 'Perl archive SHA-256 mismatch' >&2; exit 2; }
mkdir -p "$ROOT"
export ASAN_OPTIONS=detect_leaks=0

cc -g -Wall -Wextra -Werror -fsanitize=address,undefined -fno-sanitize-recover=all \
  "$HERE/test-sv-heads.c" -o "$ROOT/test-sv-heads"
for mode in 0 1; do
  PERL_SVH_REPORT=0 PERL_SVH_NATIVE_MODE=$mode "$ROOT/test-sv-heads"
done

cc -O1 -g -Wall -Wextra -Werror -fsanitize=address -fno-omit-frame-pointer \
  -c "$HERE/native.c" -o "$ROOT/native.o"
tar -xzf "$ARCHIVE" -C "$ROOT"
S=$ROOT/perl-5.36.3
chmod -R u+w "$S"
for p in "$PORT"/musl/patches/5.36.3/*.patch "$HERE"/patches/5.36.3/*.patch; do
  patch -d "$S" --batch --fuzz=0 -p1 -s < "$p"
done
(
  cd "$S"
  export PERL_SVH_NATIVE_MODE=1 PERL_SVH_REPORT=0
  ./Configure -des -Dprefix="$ROOT/install" -Uusedl -Uusethreads -Uusemymalloc \
    -Dusenm=false -Dinc_version_list=none -Doptimize='-O1 -g' \
    -Accflags='-DPERL_SV_HEAD_ADAPTER -fsanitize=address -fno-omit-frame-pointer' \
    -Aldflags="-fsanitize=address $ROOT/native.o" > "$ROOT/configure.log" 2>&1
  make -j"${JOBS:-16}" > "$ROOT/make.log" 2>&1
)
echo "built $S/perl (the build itself ran miniperl under mode 1)"

WORKLOAD=$REPO/capstone/experiments/applications/workloads/records.pl
for mode in 0 1; do
  out=$(cd "$S" && PERL_SVH_REPORT=1 PERL_SVH_NATIVE_MODE=$mode ./perl -Ilib "$WORKLOAD" 512 3 0 2> "$ROOT/records.$mode.err")
  [[ $out == 'EXP-OK perl 2357760' ]] || { echo "records.pl mode $mode: $out" >&2; exit 1; }
  ! grep -q AddressSanitizer "$ROOT/records.$mode.err" \
    || { echo "records.pl mode $mode: ASan report" >&2; exit 1; }
  grep '^PERL_REUSE_GAP' "$ROOT/records.$mode.err" | cut -c1-120
done
echo "records.pl 512 3 0: oracle in both modes, no ASan report"

control='$x = *foo; *x = $x; print "done\n"'
out=$(cd "$S" && PERL_SVH_REPORT=0 PERL_SVH_NATIVE_MODE=0 ./perl -e "$control" 2>&1)
[[ $out == done ]] || { echo "positive control, mode 0: $out" >&2; exit 1; }
if (cd "$S" && PERL_SVH_REPORT=0 PERL_SVH_NATIVE_MODE=1 ./perl -e "$control" \
      > "$ROOT/control.out" 2> "$ROOT/control.err"); then
  echo "positive control, mode 1: no report" >&2; exit 1
fi
grep -q 'AddressSanitizer: use-after-poison' "$ROOT/control.err" \
  && grep -q 'S_glob_assign_glob' "$ROOT/control.err" \
  || { echo "positive control, mode 1: wrong report" >&2; exit 1; }
echo "positive control: S_glob_assign_glob's freed-head read reported in mode 1, runs in mode 0"

if [[ $SUITE == 1 ]]; then
  for mode in 1 0; do
    mkdir -p "$ROOT/asan-$mode"
    (
      cd "$S"
      export PERL_SVH_NATIVE_MODE=$mode PERL_SVH_REPORT=0 TEST_JOBS=${TEST_JOBS:-10}
      export ASAN_OPTIONS="detect_leaks=0:log_path=$ROOT/asan-$mode/asan"
      make test_harness > "$ROOT/suite-$mode.log" 2>&1 || true
    )
    python3 - "$ROOT" "$mode" <<'PY'
import sys
from pathlib import Path
root, mode = Path(sys.argv[1]), sys.argv[2]
lines = (root / f'suite-{mode}.log').read_text(errors='replace').splitlines()
start = max(i for i, line in enumerate(lines) if line.startswith('Test Summary Report'))
print(f'== mode {mode}')
print('\n'.join(lines[start:]))
sites = {}
for report in sorted((root / f'asan-{mode}').glob('asan.*')):
    text = report.read_text(errors='replace').splitlines()
    frame = next(line.split(' in ', 1)[1] for line in text if line.strip().startswith('#0 '))
    sites[frame] = sites.get(frame, 0) + 1
for frame, count in sorted(sites.items()):
    print(f'ASan {count}x {frame}')
PY
  done
fi
