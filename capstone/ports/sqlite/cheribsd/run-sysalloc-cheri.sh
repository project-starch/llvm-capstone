#!/bin/bash
# Run ONLY the _sys binaries, with NOMEM5=1, so the case runs on the SYSTEM
# allocator instead of memsys5. Sibling of run-corpus-cheri.sh, which runs the
# plain binaries on memsys5.
#
# WHY A SEPARATE BINARY *AND* A SEPARATE ENV VAR, both of which are required:
#   - the plain build sets SQLITE_ZERO_MALLOC, which makes the default allocator
#     a stub that always fails. With NOMEM5=1 it would have no allocator at all.
#     Hence the _sys build, which drops ZERO_MALLOC.
#   - repro322_common.h:122 calls sqlite3_config(SQLITE_CONFIG_HEAP, ...)
#     unconditionally unless NOMEM5 is in the environment. SQLITE_CONFIG_HEAP
#     INSTALLS memsys5, so a _sys binary run WITHOUT NOMEM5 is still a memsys5
#     run. Both halves are needed or the arm silently measures memsys5 twice.
#
# repro_init() prints "repro allocator=system" when the switch took effect. This
# script treats a missing marker as NO-SWITCH rather than as a result, because a
# silently-unswitched run is indistinguishable from the memsys5 arm.
set -uo pipefail
# Paths. C is this directory, inside the repository, and holds the sources:
# cases/, poscontrol.c, repro322_common.h and the sibling scripts. WORK is
# where build output and run logs go, which must NOT be in the repository; it
# defaults to the out-of-tree directory these scripts were developed in, so
# behaviour is unchanged unless it is set. The toolchain and the pinned SQLite
# amalgamation are machine-specific and are overridable for the same reason.
C=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
WORK=${SQLITE_CHERI_WORK:-$HOME/arms/sqlite/cheribsd}
mkdir -p "$WORK"
SSHK="-n -i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10"
G() { ssh $SSHK -p 10086 root@localhost "$@" 2>/dev/null; }
mkdir -p "$WORK/results"
STAMP=$(date +%Y%m%d-%H%M%S)
TSV="$WORK/results/sysalloc-$STAMP.tsv"
printf 'tag\tverdict\texit\tsignal\tswitched\trevoke\tlast_line\n' > "$TSV"

echo "revocation_default=$(G 'sysctl -n security.cheri.runtime_revocation_default')"
echo "kernel=$(G 'uname -r')"
echo
printf '%-22s %-10s %-9s %-14s %s\n' CASE VERDICT SWITCHED REVOKE DETAIL

for tag in $(ls "$WORK/out" | grep '_sys$'); do
  raw=$(G "cd /root/corpus && NOMEM5=1 timeout 300 ./$tag 2>&1; echo __EXIT=\$?")
  ex=$(printf '%s' "$raw" | grep -o '__EXIT=[0-9]*' | tail -1 | cut -d= -f2)
  body=$(printf '%s' "$raw" | grep -v '__EXIT=')
  last=$(printf '%s' "$body" | tr -d '\r' | grep -v '^$' | tail -1 | cut -c1-80)
  if printf '%s' "$body" | grep -q 'allocator=system'; then sw=yes; else sw=NO; fi
  # repro_init() prints which revocation regime is in force. "stock-quarantine"
  # means no sweep is forced, so a PASS is use-after-REALLOCATION rather than a
  # statement about what revocation can see -- it is not a result for this arm.
  if printf '%s' "$body" | grep -q 'revoke=on-every-free'; then rv=every-free
  elif printf '%s' "$body" | grep -q 'revoke=stock-quarantine'; then rv=stock
  else rv=UNKNOWN; fi
  sig=''
  case "${ex:-none}" in
    0)   if printf '%s' "$body" | grep -q 'RETURNED rc=0'; then v=PASS
         elif printf '%s' "$body" | grep -q 'RETURNED rc='; then v=ERR
         else v=NORUN; fi ;;
    124) v=TIMEOUT ;;
    '')  v=NOCONN ;;
    *)   if [ "$ex" -gt 128 ]; then v=FAULT; sig=$((ex-128)); else v=ERR; fi ;;
  esac
  # A run that did not switch allocator, or did not force revocation, is not a
  # result for this arm. Both are recorded as such rather than as a verdict.
  [ "$rv" = UNKNOWN ] && v="NO-REVOKE"
  [ "$sw" = NO ] && v="NO-SWITCH"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$v" "${ex:-}" "$sig" "$sw" "$rv" "$last" >> "$TSV"
  printf '%-22s %-10s %-9s %-14s %s\n' "$tag" "$v" "$sw" "$rv" "${sig:+sig=$sig }$last"
done
echo
echo "results: $TSV"
awk -F'\t' 'NR>1{c[$2]++} END{for(k in c) printf "  %-10s %d\n", k, c[k]}' "$TSV"
