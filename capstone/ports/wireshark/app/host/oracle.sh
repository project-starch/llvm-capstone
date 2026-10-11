#!/usr/bin/env bash
# The tshark app port's native oracle, with its configuration pinned and its own controls.
#
#   oracle.sh <stock-tshark> <candidate-tshark> <captures-dir> <work-dir>
#
# Output under comparison: `tshark -r <capture> -V -n` with TZ=UTC and an EMPTY configuration
# (HOME and WIRESHARK_CONFIG_DIR point at a fresh empty directory for every run), so neither build
# can read a user profile the other does not. -n matters: without it MAC names are resolved.
#
# PASS only if all of these hold:
#   workload   dhcp, dns_port, http, arp: candidate == stock, byte for byte
#   flip       each workload capture with one byte flipped (8 from the end): stock output
#              CHANGES, and the candidate still equals stock on the flipped input
#   negative   ntp.pcap: candidate DIFFERS from stock (NTP is not whitelisted). A comparison that
#              cannot fail proves nothing; this is the harness showing it can
#   pair       dns-ooo.pcap: candidate == stock (a capture the whitelist covers, not in the workload)
# Anything missing is an ERROR, never a pass.
set -uo pipefail
STOCK=${1:?stock tshark}; CAND=${2:?candidate tshark}; CAPS=${3:?captures dir}; WORK=${4:?work dir}
mkdir -p "$WORK"
for f in "$STOCK" "$CAND"; do [ -x "$f" ] || { echo "ERROR: $f is not an executable" >&2; exit 2; }; done

run() {   # binary capture out
  local cfg; cfg=$(mktemp -d)
  HOME="$cfg" WIRESHARK_CONFIG_DIR="$cfg" TZ=UTC "$1" -r "$2" -V -n > "$3" 2> "$3.err"
  local rc=$?; rm -rf "$cfg"; return $rc
}

fail=0
say() { printf '%-9s %-14s %s\n' "$1" "$2" "$3"; }
for c in dhcp dns_port http arp; do
  src="$CAPS/$c.pcap"; [ -f "$src" ] || { say ERROR "$c" "missing capture $src"; fail=1; continue; }
  run "$STOCK" "$src" "$WORK/$c.stock"; run "$CAND" "$src" "$WORK/$c.cand"
  [ -s "$WORK/$c.stock" ] || { say ERROR "$c" "stock produced no output"; fail=1; continue; }
  if cmp -s "$WORK/$c.stock" "$WORK/$c.cand"; then say workload "$c" MATCH; else say workload "$c" DIFFERS; fail=1; fi
  python3 -c 'import sys; b=bytearray(open(sys.argv[1],"rb").read()); b[len(b)-8]^=1; open(sys.argv[2],"wb").write(b)' \
    "$src" "$WORK/$c.flip.pcap"
  run "$STOCK" "$WORK/$c.flip.pcap" "$WORK/$c.flip.stock"; run "$CAND" "$WORK/$c.flip.pcap" "$WORK/$c.flip.cand"
  changed=$(diff "$WORK/$c.stock" "$WORK/$c.flip.stock" | grep -c '^[<>]')
  if [ "$changed" -gt 0 ] && cmp -s "$WORK/$c.flip.stock" "$WORK/$c.flip.cand"; then
    say flip "$c" "FIRES ($changed diff lines) and candidate still MATCHES"
  else
    say flip "$c" "FAILED (changed=$changed, candidate-vs-stock on flip: $(cmp -s "$WORK/$c.flip.stock" "$WORK/$c.flip.cand" && echo match || echo differs))"; fail=1
  fi
done
for spec in "negative ntp DIFFERS" "pair dns-ooo MATCH"; do
  set -- $spec; src="$CAPS/$2.pcap"
  [ -f "$src" ] || { say ERROR "$2" "missing capture $src"; fail=1; continue; }
  run "$STOCK" "$src" "$WORK/$2.stock"; run "$CAND" "$src" "$WORK/$2.cand"
  got=MATCH; cmp -s "$WORK/$2.stock" "$WORK/$2.cand" || got=DIFFERS
  if [ "$got" = "$3" ]; then say "$1" "$2" "$got (as required)"; else say "$1" "$2" "$got (required $3)"; fail=1; fi
done
[ "$fail" = 0 ] && echo "ORACLE PASS" || echo "ORACLE FAIL"
exit $fail
