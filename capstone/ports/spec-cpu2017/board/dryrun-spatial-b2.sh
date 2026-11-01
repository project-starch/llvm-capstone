#!/bin/bash
# dryrun-spatial-b2.sh <work dir>: spatial-b2.sh with every board call stubbed, over each branch of its gates.
# The real pieces: the bitstream and kit hash checks, the stage list, spatial-gate.py on the recorded lbm MapLife result.
set -u
W=${1:?work dir}; rm -rf "$W"; mkdir -p "$W"
S=/mnt/hdd/$USER/capstone/spec/board; REF=$S/out-hfl/519.lbm_r-train-capstone-spatial/result.json
cat > "$W/chain.sh" <<'EOF'
#!/bin/bash
# stub chain-board: makes each stage's out dir; the pilot's result.json is $STUB_PILOT (a copy of a recorded result)
echo "STUB CHAIN board $1 list $(basename $2) SB_KIT=${SB_KIT:-unset} SB_BITSTREAM=${SB_BITSTREAM:-unset}"
while read -r st _; do case $st in ''|'#'*) continue;; esac; n=$(basename $st); [ -e $3/$n ] && continue
  mkdir -p $3/$n; [ "${n#519.lbm}" != "$n" ] && cp "$STUB_PILOT" $3/$n/result.json; echo "  stub ran $n"; done < $2
touch $3/.chain-b$1-done
EOF
cat > "$W/store.sh" <<'EOF'
#!/bin/bash
case $1 in no-such-*) exit ${STUB_STORE_CTL:-1};; *) exit 0;; esac
EOF
cat > "$W/bare.sh" <<'EOF'
#!/bin/bash
mkdir -p $3; echo "STUB bare board $1 $2 -> ${STUB_BARE:-0}" | tee $3/grade.txt; exit ${STUB_BARE:-0}
EOF
printf '#!/bin/bash\nexit 3\n' > "$W/nvrb.sh"; printf '#!/bin/bash\nexit ${STUB_BACK:-0}\n' > "$W/back.sh"
chmod +x "$W"/*.sh
python3 - "$REF" "$W/pilot-bad.json" <<'EOF'
import json,sys; d=json.load(open(sys.argv[1])); d['arms']['capstone-spatial']['hashes']['lbm.out']='0'*64; json.dump(d,open(sys.argv[2],'w'))
EOF
case_run() {  # <name> <env...>
  local n=$1; shift; local D=$W/$n; mkdir -p $D/hfl
  cp -r "$(dirname $REF)" $D/hfl/; touch $D/hfl/.chain-b2-done
  ( : ) & local dead=$!; wait $dead
  env SP_PID=$dead SP_HFL=$D/hfl SP_OUT=$D/out SP_SLEEP=0 SP_CHAIN="$W/chain.sh" SP_BUSY=true SP_STORE="$W/store.sh" \
    SP_BOOKER=echo SP_FLASH=true SP_NVRB="$W/nvrb.sh" SP_BARE="$W/bare.sh" SP_AFTER="echo STUB after-train" \
    SP_BACK="$W/back.sh" STUB_PILOT=$REF "$@" bash $S/spatial-b2.sh
  echo "===== $n (exit $?)"; grep -E '^(SP|STUB|  stub|  G|  GATE)' $D/out/spatial-b2.log
}
case_run normal
case_run no-done-marker SP_HFL=/nonexistent
case_run store-control STUB_STORE_CTL=0
case_run bad-bit-hash SP_BIT=$S/spatial-gate.py
case_run flash-fails SP_FLASH=false
case_run bare-not-confirmed STUB_BARE=4
case_run pilot-gate-miss STUB_PILOT=$W/pilot-bad.json
case_run back-fails STUB_BACK=1
