#!/bin/bash
# rejudge.sh TAG: judge run19's stored runs (a copy of the first pass's per-arm run records and platform.json)
# with the runner as it stands in the tree, both scorings, then merge the two populations. No boot.
# Same working-directory layout and relative paths as the first pass's engine-judge.sh, so that a path in the
# evidence cannot differ for a reason other than the judge.
source $HOME/crenew-step4/corpus/work-fix-php/env.sh
set -e
tag=$1; cd $W/rejudge
R=$TREE/capstone
rm -rf out-$tag; mkdir -p out-$tag/run out-$tag/sens logs
common=(--corpus $R/bug-corpora/php/engine-repros --image php-meas/realphp/php-real-nested.dom --probe php-meas/probe-nested.dom
        --llvm-bin llvm/bin --qemu kit/qemu-system-riscv64 --images kit/images --adapter kit/adapter
        --platform-probe putenv=php-meas/probe-putenv.dom --from-record)
for pop in asan-blind temporal-asan-visible; do
  for s in run sens; do
    mkdir -p out-$tag/$s/$pop
    cp stored/run19/$pop/php-real-*-run.json stored/run19/$pop/platform.json out-$tag/$s/$pop/
    cp -r stored/run19/$pop/stage out-$tag/$s/$pop/
  done
  python3 $R/ports/php/app/run-realphp.py "${common[@]}" --population $pop --out out-$tag/run/$pop > logs/$tag-judge-$pop.log 2>&1
  python3 $R/ports/php/app/run-realphp.py "${common[@]}" --population $pop --out out-$tag/sens/$pop --score-oracle-crashes > logs/$tag-sens-$pop.log 2>&1
done
python3 $R/bug-corpora/php/engine-repros/tools/merge-population-bundles.py out-$tag/run out-$tag/merged
python3 $R/bug-corpora/php/engine-repros/tools/merge-population-bundles.py out-$tag/sens out-$tag/merged-sens
# The first pass recorded the symbolizer's source paths relative to its work directory (the run19 README says
# so): the absolute prefix is removed from the per-population bundles' rows, then the populations are merged again.
for s in run sens; do for f in out-$tag/$s/*/bundle-*/verdicts.jsonl; do sed -i "s|$A/||g" "$f"; done; done
rm -rf out-$tag/merged out-$tag/merged-sens
python3 $R/bug-corpora/php/engine-repros/tools/merge-population-bundles.py out-$tag/run out-$tag/merged
python3 $R/bug-corpora/php/engine-repros/tools/merge-population-bundles.py out-$tag/sens out-$tag/merged-sens
! grep -rl "$HOME" out-$tag/merged out-$tag/merged-sens
