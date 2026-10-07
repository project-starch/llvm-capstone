#!/bin/bash
# One-command reproduction of the CheriBSD arm result (Dc = 0 of 24).
#
#   bash capstone/ports/sqlite/cheribsd/verify-cheribsd-arm.sh
#
# Assumes the guest is already booted and reachable on localhost:10086.
# If it is not, start it first (takes ~6 min to reach sshd):
#   tmux new-session -d -s cheriqemu 'bash ~/cheriBSD/run-qemu.sh'
#
# Per-process revocation knobs are libc env vars. Do NOT use the system sysctl
# security.cheri.runtime_revocation_every_free_default: it is global, sshd's own
# malloc/free traffic then makes the guest unreachable, and only a VM restart recovers it.
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
K="-n -i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
G() { ssh $K -p 10086 root@localhost "$@" 2>/dev/null; }

[ "$(G 'echo UP')" = UP ] || { echo "guest unreachable on localhost:10086 -- boot it first (see header)"; exit 1; }

echo "### environment"
G "uname -r; sysctl -n kern.features.cheri_revoke security.cheri.runtime_revocation_default security.cheri.runtime_revocation_every_free_default" \
  | paste -sd' ' - | sed 's/^/  kernel+knobs: /'

one() { # $1=tag $2=env-prefix $3=timeout
  local e l
  e=$(G "cd /root/corpus && env ${2:-X=1} timeout ${3:-300} ./$1 >/tmp/v.txt 2>&1; echo \$?")
  l=$(G "grep -v '^\$' /tmp/v.txt | tail -1 | cut -c1-46")
  case "${e:-?}" in
    0)   v=PASS ;;
    162) v='FAULT SIGPROT' ;;
    139) v='FAULT SIGSEGV' ;;
    138) v='FAULT SIGBUS <- PORT BROKEN' ;;
    124) v=TIMEOUT ;;
    *)   v="ERR($e)" ;;
  esac
  printf '  %-17s %-26s %s\n' "$1" "$v" "$l"
}

echo
echo "### gate: positive control must FAULT with revocation on, and PASS with it off"
one poscontrol
one poscontrol _RUNTIME_REVOCATION_DISABLE=1

echo
echo "### main run: all 24 corpus domains + 5 baseline probes (default config)"
for t in $(ls "$WORK/out" | grep -vE '^poscontrol|_static$'); do one "$t"; done
one spellfixoom_static '' 600   # statically linked: purecap ld-elf.so.1 rejects its traditional TLS

echo
echo "### control B2: revocation off, per process -- the 4 faults must be unchanged"
for t in backupattach detachtrig rtreecursor rtreeinode0; do one "$t" _RUNTIME_REVOCATION_DISABLE=1; done

echo
echo "### control C: revoke on EVERY free, per process -- strictest available"
one poscontrol _RUNTIME_REVOCATION_EVERY_FREE_ENABLE=1 600
for t in backupattach rtreecursor mem5design wschema fts5vocabeof staticbind; do
  one "$t" _RUNTIME_REVOCATION_EVERY_FREE_ENABLE=1 900
done

echo
echo "### expected"
cat <<'X'
  poscontrol             FAULT SIGPROT  (revocation on / every-free)   PASS (revocation off)
  backupattach, detachtrig, rtreecursor, rtreeinode0
                         FAULT SIGPROT in ALL configurations -> revocation-independent,
                         i.e. incidental memsys5 in-band-metadata tag clobber, NOT a detection
  every other domain     PASS in ALL configurations
  => Dc = 0 of 24.  Any SIGBUS means the purecap source adaptation is missing.
X
