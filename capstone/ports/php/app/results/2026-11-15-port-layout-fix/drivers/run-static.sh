#!/usr/bin/env bash
# run-static.sh: the static layout check (layout-static.dom of both arms) in one boot, into run-6/layout/php-static.
E=$HOME/asr-work/eval F=$HOME/asr-work/fix-php
cd $F/tree && source capstone/runtime/virtual/tests/trap-platform/eval-p13.env
export CAPSTONE_QEMU_DOMAIN_LOCK=$F/lock-static CAPSTONE_TMP_ROOT=$F/tmp PATH=$HOME/lanes/gatefix/llvm/bin:/usr/local/bin:/usr/bin:/bin
OUT=$E/run-6/layout/php-static
python3 capstone/runtime/virtual/trap_platform.py run --kit $E/kit-run-3 --work $OUT --step-timeout 300 \
  --file stock.dom=$F/prebuilt/trace-stock/layout-static.dom --file protected.dom=$F/prebuilt/trace-protected/layout-static.dom \
  --step "static-stock=cd /mnt/vm; ./capstone-vexec ./stock.dom" --step "static-protected=cd /mnt/vm; ./capstone-vexec ./protected.dom"
rc=$?
{ echo "static layout check, tree fix/php-port-layout $(cat $F/tree.commit), trap_platform rc=$rc"; sha256sum $F/prebuilt/trace-stock/layout-static.dom $F/prebuilt/trace-protected/layout-static.dom; } > $OUT/provenance.txt
python3 -c "
import json
for r in json.load(open(\"$OUT/results.json\")): print(r[\"name\"], r[\"kind\"], r[\"value\"], [l for l in r[\"output\"].splitlines() if l.startswith(\"LAYOUT-STATIC\")])
"
