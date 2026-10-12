#!/usr/bin/env bash
# Build one arm of virtual Perl 5.36.3 with the reuse-distance hooks
# (docs/plans/2026-10-11-nested-reuse-distance.md), without editing the port:
#
#   build.sh stock|protected ROOT
#
# Environment as for ports/perl/musl/build-perl-domain.sh (CAPSTONE_LLVM_BUILD_DIR,
# CAPSTONE_REPO_ROOT, CAPSTONE_TMP_ROOT, PERL_MIRROR, PERL_CROSS_MIRROR, JOBS).
#
# 1. The port's own build, unchanged (PERLD_PROFILE=virtual, PERLD_SUBLET=0|1,
#    default -O2): it unpacks, runs perl-cross's crosspatch, applies the port's
#    patches (and sublet 0001 on the protected arm) and builds. Its image is
#    kept as ROOT/uninstrumented.dom, to compare with the images of earlier
#    runs built the same way.
# 2. make clean, then the hook patch for this arm (sv-heads-reuse-gap-ARM.patch,
#    fuzz 0) and the two headers into the source root, then the port's build
#    again from its perl stage with PERLD_OPT="-O2 -DEXP_REUSE_GAP". The
#    define goes to the target compile only (perl-cross's -Doptimize; the host
#    miniperl uses its own optimize). The image is ROOT/perl.dom.
# 3. Checks that fail the build: every object in libperl.a and lib/auto/*.a
#    that refers to PL_sv_root (each new_SV expansion does) also refers to
#    Perl_reuse_gap_sv_issue after step 2, and none does after step 1; the
#    image defines Perl_reuse_gap_sv_issue and Perl_reuse_gap_sv_release.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ $# == 2 ]] || { echo "usage: build.sh stock|protected ROOT" >&2; exit 2; }
ARM=$1 ROOT=$(realpath -m "$2")
case $ARM in stock) SUB=0 ;; protected) SUB=1 ;; *) echo "arm: stock or protected" >&2; exit 2 ;; esac
R=${CAPSTONE_REPO_ROOT:?}
STUDY=$R/capstone/experiments/study
PORT=$R/capstone/ports/perl/musl/build-perl-domain.sh
S=$ROOT/src/perl-5.36.3
NM=${CAPSTONE_LLVM_BUILD_DIR:?}/bin/llvm-nm
[[ ! -e $ROOT ]] || { echo "use a fresh ROOT: $ROOT" >&2; exit 2; }
mkdir -p "$ROOT"
export PERLD_PROFILE=virtual PERLD_SUBLET=$SUB PERLD_ROOT=$ROOT

# hooked_objects LABEL: per archive member, does it use PL_sv_root (an undefined
# reference; every new_SV expansion makes one, globals.o only defines it)
# and does it call (or, sv.o, define) the issue hook?
hooked_objects() {
  local line
  line=$(find "$S/libperl.a" "$S/lib/auto" -name '*.a' -print0 | xargs -0 "$NM" -A 2>/dev/null | python3 -c '
import re, sys, collections
seen = collections.defaultdict(lambda: [0, 0])
for line in sys.stdin:
    m = re.match(r"^(\S+?\.a:[^:\s]+):\s+(?:[0-9a-f]+\s+)?([A-Za-z])\s+(\S+)$", line.rstrip("\n"))
    if not m: continue
    member, kind, name = m.groups()
    if name == "PL_sv_root" and kind == "U": seen[member][0] = 1
    if name == "Perl_reuse_gap_sv_issue" and kind in "UT": seen[member][1] = 1
uses = sum(a for a, b in seen.values())
hooked = sum(b for a, b in seen.values())
missing = sorted(k for k, (a, b) in seen.items() if a and not b)
print(uses, hooked, len(missing), " ".join(missing[:5]))')
  read -r CHECK_USES CHECK_HOOKED CHECK_MISSING CHECK_EXAMPLES <<< "$line"
  echo "$1 objects_using_PL_sv_root=$CHECK_USES objects_calling_issue_hook=$CHECK_HOOKED using_without_hook=$CHECK_MISSING ${CHECK_EXAMPLES:-}"
}

s=$(date +%s)
bash "$PORT" > "$ROOT/build-1.log" 2>&1
cp "$S/perl" "$ROOT/uninstrumented.dom"
hooked_objects uninstrumented
[[ $CHECK_USES -gt 0 && $CHECK_HOOKED == 0 ]] \
  || { echo "control: the uninstrumented build already calls the hook, or nothing uses PL_sv_root" >&2; exit 1; }
echo "STEP1 $ARM $(( $(date +%s) - s ))s uninstrumented sha256=$(sha256sum "$ROOT/uninstrumented.dom" | cut -d' ' -f1)"

s=$(date +%s)
(cd "$S" && make clean > "$ROOT/clean.log" 2>&1)
cp "$STUDY/reuse-gap-observer.h" "$HERE/reuse-gap-hook.h" "$S/"
patch -d "$S" -p1 --batch --forward --fuzz=0 < "$HERE/sv-heads-reuse-gap-$ARM.patch" > "$ROOT/hook-patch.log"
PERLD_FROM=perl PERLD_OPT="-O2 -DEXP_REUSE_GAP" bash "$PORT" > "$ROOT/build-2.log" 2>&1
grep -q -- '-DEXP_REUSE_GAP' "$S/Makefile.config" \
  || { echo "EXP_REUSE_GAP is not in the target CFLAGS" >&2; exit 1; }
if grep -q -- 'EXP_REUSE_GAP' <(grep '^HOSTCFLAGS' "$S/Makefile.config"); then
  echo "EXP_REUSE_GAP reached the host CFLAGS" >&2; exit 1
fi
cp "$S/perl" "$ROOT/perl.dom"
hooked_objects instrumented
[[ $CHECK_USES -gt 0 && $CHECK_MISSING == 0 ]] \
  || { echo "an object hands out SV heads without the hook" >&2; exit 1; }
# nm to a file first: under pipefail, `nm | grep -q` fails when grep exits early.
"$NM" "$ROOT/perl.dom" > "$ROOT/perl.nm"
for sym in Perl_reuse_gap_sv_issue Perl_reuse_gap_sv_release; do
  grep -q " T $sym\$" "$ROOT/perl.nm" || { echo "image lacks $sym" >&2; exit 1; }
done
echo "STEP2 $ARM $(( $(date +%s) - s ))s perl.dom sha256=$(sha256sum "$ROOT/perl.dom" | cut -d' ' -f1)"
