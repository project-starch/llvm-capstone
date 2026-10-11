#!/usr/bin/env bash
# c-repros as Capstone domain programs, one per case, built by an application
# SDK's capstone-cc.
#
# WHY THIS EXISTS. These five are the corpus's non-nested cases: the defective
# object comes straight from malloc, not from an allocator carved out of a
# bigger block. Until now they ran only on CheriBSD, and the reason given was
# that a Capstone domain cannot host a non-nested object. That reason was
# wrong. It pointed at memory-contexts/src/allocators, which is the replay
# study's level 0 and is in neither arm image. What an arm image actually
# links is ports/musl-capstone/runtime/level0.c, where
# CAPSTONE_LEVEL0_OBJECT_BOUNDS has been on by default since 2026-09-30:
# malloc narrows what it returns to the bytes asked for, so an object from it
# carries its own bounds exactly as a CheriBSD malloc'd object does. Nothing
# had to be relaxed for these cases to run. They had no domain build, and this
# is it.
#
# WHAT THE SUBLET ARM MEANS HERE, and why nothing about it is weakened. Sublet
# protects sub-allocations an application allocator carves out of a block it
# owns. These cases have no application allocator, so there is nothing to
# sublet and the arm has nothing extra to enforce; the refusing malloc of
# memory-contexts/src/allocators/sublet/unsupported-allocators.c, whose job is
# to catch a PostgreSQL memory manager reaching an unadapted libc path, is not
# in this build and is untouched by it. Running the sublet SDK here measures
# whether its runtime changes what happens to a plain malloc'd object, which
# is a result worth having and is not a claim that these cases exercise
# sublet.
#
#   usage: build-domain.sh <sdk dir> [out dir]
#
# The SDK is the one an arm was built with, so the cases run on exactly the
# runtime the rest of that arm ran on -- for the two PostgreSQL arms,
# $PG_SU_ROOT/runtime from ports/postgres/app/build-domain.sh.
set -euo pipefail
SDK=${1:?usage: build-domain.sh <sdk dir> [out dir]}
S=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
R=$(cd -- "$S/.." && pwd)
OUT=${2:-$R/out-domain}
CC=$SDK/capstone-cc
[[ -x $CC ]] || { echo "no capstone-cc in $SDK" >&2; exit 2; }

# -O0, as the CheriBSD arm uses: these cases turn on a specific read or write
# running off a specific object, and an optimiser that hoists or merges them
# moves the fault away from the line the case names.
OPT=${PGCLIENT_OPT_LEVEL:--O0}

mkdir -p "$OUT"
built=0
failed=0
for d in "$R"/[0-9][0-9]_*/; do
  [[ -f $d/case.c ]] || continue
  tag=$(basename "$d")
  # Each case builds TWICE: the case, and its own negative control with
  # -DPGCLIENT_NEGATIVE_CONTROL, which moves one value to the safe side of the
  # boundary the defect crosses and must therefore COMPLETE. This is the
  # per-case control the judge calls attribution by `control`, and it is a
  # different question from controls.dom below: that one asks whether this
  # configuration reports at all, this one asks whether THIS fault depends on
  # THIS defect. The suffix keeps it out of case discovery, so no runner can
  # pick a control up as a measurement.
  for variant in "" "-control"; do
    defs=()
    [[ -n $variant ]] && defs=(-DPGCLIENT_NEGATIVE_CONTROL)
    if "$CC" "$OPT" -I"$S" "${defs[@]}" "$d/case.c" "$S/driver.c" "$S"/upstream_*.c \
         -o "$OUT/$tag$variant.dom" 2> "$OUT/$tag$variant.err"; then
      printf '  %-52s OK  %s bytes\n' "$tag$variant" "$(stat -c %s "$OUT/$tag$variant.dom")"
      built=$((built + 1))
    else
      printf '  %-52s FAIL\n' "$tag$variant"
      sed 's/^/      /' < "$OUT/$tag$variant.err" | head -8
      failed=$((failed + 1))
    fi
  done
done
# The arm's controls, from the same SDK (shared/controls.c), and what this build IS: the runner
# reads the heap from here and refuses an arm whose configuration needs another one.
if "$CC" "$OPT" "$S/controls.c" -o "$OUT/controls.dom" 2> "$OUT/controls.err"; then
  echo "  controls.dom OK"
else
  echo "  controls.dom FAIL"; sed 's/^/      /' < "$OUT/controls.err" | head -8
  failed=$((failed + 1))
fi
heap=$(sed -n 's/^CAPSTONE_APPLICATION_HEAP:STRING=//p' "$SDK/CMakeCache.txt")
virtual=$(sed -n 's/^CAPSTONE_APPLICATION_VIRTUAL:BOOL=//p' "$SDK/CMakeCache.txt")
printf '{"sdk": "%s", "heap": "%s", "virtual": "%s", "runtime_sha256": "%s", "opt": "%s"}\n' \
  "$(cd "$SDK" && pwd)" "${heap:-unknown}" "${virtual:-OFF}" \
  "$(sha256sum "$SDK/libapplication-runtime.a" 2>/dev/null | cut -d' ' -f1)" "$OPT" > "$OUT/build.json"
echo "built=$built failed=$failed heap=${heap:-unknown}  ($OUT)"
[[ $failed -eq 0 ]]
