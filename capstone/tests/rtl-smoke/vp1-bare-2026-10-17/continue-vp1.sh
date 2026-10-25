#!/bin/bash
# vp1's first session, continued after an INFRA non-run (2026-10-17).
# At 20:03 the 19:56:58 session's S-16 arm s16st-esc-n8 never started: JTAG load_image timed out ("waiting for busy to
# go low") before `continue`. It produced no reading, the S-16 comparison read 17/18 identical, and the session's gate
# stops before corpus set 7 on that missing reading. Rule and precedent: the board-run skill ("JTAG errors | infra --
# retry"), and board-36a641e0b.result-lines.txt:7-9 (the same event, re-run once with the same image).
# This script acts only when that is the ONLY thing between the session and its gate:
#   - every other gate line passed (FLOOR, S1GAPF, all four VP1ARM);
#   - every S-16 arm not identical to d93a8d66b never started (no "] continue", no reading).
# Then, for each such arm:
#   - re-run it ONCE, same image and record address;
#   - re-compare S-16;
#   - write a new SEQ section carrying the session's other gate lines verbatim (each marked) beside the new S16 line;
#   - apply after-flash.sh's gate to that section, unchanged.
# Pass: corpus set 7 exactly as after-flash.sh runs it. C7 identical: vp2's chain (the lead: "if vp1 as expected, go
# with vp2"). Anything else: HOLD for the lead. Never edits the running scripts; waits for them to end.
main() {
  set -u
  local V=${CONT_V:-/tmp/capstone/vp1-session} A=/tmp/capstone/accept-sublet C=${CONT_C:-/tmp/capstone/vp1-cont}
  local L=$V/SEQ G=$C/CONT.log S0="AFTER-FLASH START 19:56:58" BIT=caplifive_vp1_9ed9a2594.bit
  local F=/tmp/capstone/wt-b0/capstone/tests/rtl-smoke/sup-resume-2026-10-03
  local CH=${CONT_CHAIN:-/tmp/capstone/vp2-session/chain-vp2.sh} CL=${CONT_CHAINLOG:-/tmp/capstone/vp2-session/CHAIN.log}
  local RUNNER=${CONT_RUNNER:-$F/run_sup_bare_wedge.py} WAIT=${CONT_WAIT:-/tmp/capstone/vp2-session/board-free-wait.sh}
  local end=$(( $(date +%s) + 3*3600 ))
  echo "CONT START $(date +%T): waiting for rerun-then-vp2.sh to end" >> $G
  until grep -q '^CHAIN-VP2 rc=' $V/RERUN.log 2>/dev/null || [ $(date +%s) -ge $end ]; do sleep ${CONT_POLL_S:-30}; done
  grep -q '^CHAIN-VP2 rc=' $V/RERUN.log || { echo "HOLD $(date +%T): rerun-then-vp2.sh still running after 3 h" >> $G; return 1; }
  # vp2's chain must have HELD on this session, not run.
  local ch; ch=$(awk '/^CHAIN-VP2 START/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' $CL)
  printf '%s' "$ch" | grep -q "did not end clean; vp2 waits for the lead" \
    || { echo "HOLD $(date +%T): vp2's chain did not hold as expected: $(printf '%s' "$ch" | tail -2 | tr '\n' ' ')" >> $G; return 1; }
  # The session's own section: exactly one AFTER-FLASH section from 19:56:58 on, and it STOPPED at the gate.
  local cur; cur=$(awk -v s="$S0" 'index($0, s) == 1 {f=1} f' $L)
  [ "$(printf '%s\n' "$cur" | grep -c '^AFTER-FLASH START')" -eq 1 ] || { echo "HOLD: not exactly one section from $S0" >> $G; return 1; }
  printf '%s\n' "$cur" | grep -q '^STOP before corpus set 7' || { echo "HOLD: the session did not stop at its gate" >> $G; return 1; }
  local k
  for k in "FLOOR vs d93a8d66b rc=0" "S1GAPF vs d93a8d66b rc=0" "S16 vs d93a8d66b rc=1"; do
    printf '%s\n' "$cur" | grep -q "^$k " || { echo "HOLD $(date +%T): the session has no '$k' line" >> $G; return 1; }; done
  [ "$(printf '%s\n' "$cur" | grep -c '^VP1ARM .* rc=0 ::.*AS PRE-REGISTERED')" -eq 4 ] || { echo "HOLD: not all four VP1ARM as pre-registered" >> $G; return 1; }
  [ "$(printf '%s\n' "$cur" | grep -c '^GATE:')" -eq 1 ] || { echo "HOLD: more than the S16 gate line failed" >> $G; return 1; }
  # Every non-identical S-16 arm must be a non-start; at most two (more says the board, not an arm).
  local infra=() line a
  while IFS= read -r line; do
    case "$line" in
      "SAME "*|[0-9]*" arms, "*" not identical") ;;
      "DIFF "*) a=${line#DIFF }; a=${a%%:*}
         if [[ "$line" == *" new []" ]] && [ -f $V/accept/runs-$a.log ] && ! grep -q -E '\] continue\s*$|reading +[0-9]+: 0x' $V/accept/runs-$a.log
         then infra+=("$a"); else echo "HOLD $(date +%T): S-16 arm $a differs and it ran: $line" >> $G; return 1; fi ;;
      *) echo "HOLD $(date +%T): comparator line not covered by the rule: $line" >> $G; return 1 ;;
    esac
  done < $V/accept-cmp.txt
  [ ${#infra[@]} -ge 1 ] && [ ${#infra[@]} -le 2 ] || { echo "HOLD: ${#infra[@]} non-started arms (the rule covers 1-2)" >> $G; return 1; }
  echo "CONT $(date +%T): the session's only failure is the non-start of: ${infra[*]}" >> $G
  bash $WAIT 3600 || { echo "HOLD $(date +%T): the board was still busy after 1 h" >> $G; return 1; }
  export NEWBIT=$BIT FPGA_BITSTREAM=$BIT
  export FPGA_URL="$(cat ${CONT_URL_FILE:-$HOME/.claude-kisp/secrets/fpga-console-url})"
  mkdir -p $C/infra-fail-1; cp $V/accept-cmp.txt $C/accept-cmp.first.txt
  local img rec
  for a in "${infra[@]}"; do
    img=$F/images/$a.bin; rec=$(cat $F/images/$a.rec)
    grep -q "image $img -> " $V/accept/runs-$a.log || { echo "HOLD: $a's failed log does not name $img" >> $G; return 1; }
    mv $V/accept/runs-$a.log $C/infra-fail-1/runs-$a.log
    [ -d $V/accept/runs/$a ] && mv $V/accept/runs/$a $C/infra-fail-1/run-$a
    echo "$a RETRY START $(date +%T) $(sha256sum $img | cut -c1-16) (infra: the 19:56:58 attempt never started)" >> $V/accept/SEQ
    SUP_IMG=$img SUP_OUT=$V/accept/runs/$a SUP_REC_ADDR=$rec SUP_TIMEOUT=90 nice -n 10 python3 $RUNNER > $V/accept/runs-$a.log 2>&1
    echo "$a RETRY END rc=$? $(date +%T)" >> $V/accept/SEQ; sleep ${CONT_GAP_S:-8}
  done
  python3 $A/cmp-readings.py $A/accept $V/accept > $V/accept-cmp.txt 2>&1; local rc=$?
  { echo "AFTER-FLASH START $(date +%T) $BIT CONTINUATION of 19:56:58: infra retry of ${infra[*]} (never started: JTAG load_image busy timeout), one retry each, same image"
    echo "S16 vs d93a8d66b rc=$rc $(tail -1 $V/accept-cmp.txt) $(date +%T) [after the infra retry; the first comparison is vp1-cont/accept-cmp.first.txt]"
    printf '%s\n' "$cur" | grep -E '^(FLOOR|S1GAP|S1GAPF) vs d93a8d66b |^VP1ARM ' | sed 's/$/ [carried from 19:56:58]/'
  } >> $L
  # after-flash.sh's gate, unchanged, on this section.
  local ok=1 sec; sec=$(awk '/^AFTER-FLASH START/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' $L)
  for k in "S16 vs d93a8d66b rc=0" "FLOOR vs d93a8d66b rc=0" "S1GAPF vs d93a8d66b rc=0"; do
    printf '%b' "$sec" | grep -q "^$k " || { ok=0; echo "GATE: missing '$k'" >> $L; }
  done
  [ "$(printf '%b' "$sec" | grep -c '^VP1ARM .* rc=0 ::.*AS PRE-REGISTERED')" -eq 4 ] || { ok=0; echo "GATE: not all four VP1ARM as pre-registered" >> $L; }
  if [ $ok -ne 1 ]; then echo "STOP before corpus set 7: a reading differs (PREREG) $(date +%T)" >> $L; echo "HOLD $(date +%T): the gate stops after the retry" >> $G; return 1; fi
  bash ${CONT_C7:-$V/run-vp1-c7.sh} 1d2bbbf37f63f435 > $V/c7.out 2>&1
  python3 $V/compare-corpus.py /tmp/capstone/corpus-sil/board-c7/run/board-corpus.txt $V/c7/board-corpus.txt > $V/c7-cmp.txt 2>&1; rc=$?
  echo "C7 vs d93a8d66b rc=$rc $(tail -1 $V/c7-cmp.txt) $(date +%T)" >> $L
  # Stricter than the armed chain, which judged only DONE/STOP: a set-7 difference is "vp1 not as expected".
  if [ $rc -ne 0 ]; then echo "STOP after corpus set 7: it differs from d93a8d66b $(date +%T)" >> $L; echo "HOLD: C7 differs" >> $G; return 1; fi
  echo "AFTER-FLASH DONE $(date +%T)" >> $L
  echo "CONT $(date +%T): vp1 as expected; vp2's chain" >> $G
  VP1_CHAIN_PID=999999 bash $CH; echo "CHAIN-VP2 rc=$? $(date +%T)" >> $G
}
main "$@"; exit $?
