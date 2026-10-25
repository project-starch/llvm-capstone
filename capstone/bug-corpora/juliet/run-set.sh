#!/usr/bin/env bash
# One set end to end, in the documented order:
#   select -> build (census, seeded) -> run (census, seeded) -> probe -> judge
#   -> build --twins / run (twins) for the MISSED cases of the twin families -> asan -> judge (final)
#
#   run-set.sh <temporal|spatial> <work-dir> <juliet C dir> <kit dir> <virtual SDK dir> <llvm bin dir> <juliet-timeout>
#
# Results land in <work-dir>/<set>/results/: <set>-verdicts.tsv and <set>-summary.json. Exit status is judge.py's.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
set_=${1:?set}; work=${2:?work dir}; juliet=${3:?juliet C dir}; kit=${4:?kit}; sdk=${5:?sdk}; llvm=${6:?llvm bin}
helper=${7:?juliet-timeout}
d=$work/$set_
[ ! -e "$d" ] || { echo "$d exists; a set runs into a fresh directory" >&2; exit 1; }
mkdir -p "$d"
python3 "$here/select.py" --juliet "$juliet" --set "$set_" --out "$d/manifest.json"
python3 "$here/build.py" --manifest "$d/manifest.json" --juliet "$juliet" --sdk "$sdk" --out "$d/build"
python3 "$here/build.py" --manifest "$d/manifest.json" --juliet "$juliet" --sdk "$sdk" --out "$d/build-seeded" --seeded
python3 "$here/run.py" --build "$d/build" --kit "$kit" --helper "$helper" --out "$d/run"
python3 "$here/run.py" --build "$d/build-seeded" --kit "$kit" --helper "$helper" --out "$d/run-seeded"
python3 "$here/probe.py" --sdk "$sdk" --kit "$kit" --helper "$helper" --out "$d/probe"
j=(--manifest "$d/manifest.json" --juliet "$juliet" --build "$d/build" --run "$d/run"
   --seeded-build "$d/build-seeded" --seeded-run "$d/run-seeded" --llvm-bin "$llvm" --out "$d/results")
python3 "$here/judge.py" "${j[@]}" --probe "$d/probe/probe.json"
python3 "$here/build.py" --manifest "$d/manifest.json" --juliet "$juliet" --sdk "$sdk" --out "$d/build-twins" --twins \
    --verdicts "$d/results/$set_-verdicts.tsv"
python3 "$here/run.py" --build "$d/build-twins" --kit "$kit" --helper "$helper" --out "$d/run-twins"
python3 "$here/asan.py" --manifest "$d/manifest.json" --juliet "$juliet" --verdicts "$d/results/$set_-verdicts.tsv" \
    --out "$d/asan.json"
python3 "$here/judge.py" "${j[@]}" --probe "$d/probe/probe.json" --twin-build "$d/build-twins" --twin-run "$d/run-twins" \
    --asan "$d/asan.json"
