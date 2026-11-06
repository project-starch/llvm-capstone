#!/usr/bin/env bash
# One set end to end on CheriBSD, in the documented order:
#   select -> build -> run (default; and off, every-free for the temporal set) -> seeded build + run -> judge
#   -> build --twins / run (twins) for the MISSED cases of the twin families -> judge (final)
#
#   run-set.sh <temporal|spatial> <work-dir> <juliet C dir> <CheriBSD sdk> <purecap rootfs> <disk image> [guests]
#
# Results land in <work-dir>/<set>/results/: <set>-verdicts.tsv and <set>-summary.json. Exit status is judge.py's.
# Needs a python3 with pexpect on PATH (python 3.9 or later); never start it from ../ (select.py shadows `select`).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
set_=${1:?set}; work=${2:?work dir}; juliet=${3:?juliet C dir}; sdk=${4:?sdk}; rootfs=${5:?rootfs}; image=${6:?image}
guests=${7:-1}
d=$work/$set_
[ ! -e "$d" ] || { echo "$d exists; a set runs into a fresh directory" >&2; exit 1; }
mkdir -p "$d"
cd /   # not ../: see above
python3 "$here/../select.py" --juliet "$juliet" --set "$set_" --out "$d/manifest.json"
python3 "$here/build.py" --manifest "$d/manifest.json" --juliet "$juliet" --sdk "$sdk" --out "$d/build"
vm=(--sdk "$sdk" --rootfs "$rootfs" --image "$image" --guests "$guests")
python3 "$here/run.py" --build "$d/build" "${vm[@]}" --revocation default --out "$d/run"
extra=()
if [ "$set_" = temporal ]; then
    python3 "$here/run.py" --build "$d/build" "${vm[@]}" --revocation off --out "$d/run-off"
    python3 "$here/run.py" --build "$d/build" "${vm[@]}" --revocation every-free --out "$d/run-every-free"
    extra=(--off-run "$d/run-off" --every-free-run "$d/run-every-free")
fi
python3 "$here/build.py" --manifest "$d/manifest.json" --juliet "$juliet" --sdk "$sdk" --out "$d/build-seeded" \
    --seeded --seeds "$d/run/observations.json"
python3 "$here/run.py" --build "$d/build-seeded" "${vm[@]}" --revocation default --out "$d/run-seeded"
j=(--manifest "$d/manifest.json" --juliet "$juliet" --build "$d/build" --run "$d/run"
   --seeded-build "$d/build-seeded" --seeded-run "$d/run-seeded" --llvm-bin "$sdk/bin" --rootfs "$rootfs"
   --out "$d/results" "${extra[@]}")
python3 "$here/judge.py" "${j[@]}"
python3 "$here/build.py" --manifest "$d/manifest.json" --juliet "$juliet" --sdk "$sdk" --out "$d/build-twins" --twins \
    --verdicts "$d/results/$set_-verdicts.tsv" --seeds "$d/run/observations.json"
python3 "$here/run.py" --build "$d/build-twins" "${vm[@]}" --revocation default --out "$d/run-twins"
python3 "$here/judge.py" "${j[@]}" --twin-build "$d/build-twins" --twin-run "$d/run-twins"
