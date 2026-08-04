#!/usr/bin/env bash
# Toolchain for Wireshark #16807. No from-source Wireshark build: the stock
# apt tshark (4.6.4) is compiled with Lua 5.4 (libwireshark depends on
# liblua5.4-0) and can load a Lua dissector headlessly, which is all we need.
# We detect the UAF with valgrind since the apt binary is not ASan-built.
# Idempotent. Reproduced on 2026-08-03.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

need=()
command -v tshark   >/dev/null 2>&1 || need+=(tshark)
command -v valgrind >/dev/null 2>&1 || need+=(valgrind)

if [ "${#need[@]}" -gt 0 ]; then
  echo "== installing: ${need[*]} =="
  # preseed so tshark's debconf (setuid dumpcap) does not prompt
  echo "wireshark-common wireshark-common/install-setuid boolean false" | sudo debconf-set-selections
  sudo apt-get update -q
  sudo apt-get install -y -q "${need[@]}"
fi

echo "-- tshark: $(tshark --version 2>/dev/null | head -1)"
echo "-- lua in libwireshark: $(dpkg -s libwireshark19 2>/dev/null | grep -io 'liblua5.4-0' | head -1 || echo '(check)')"
echo "-- valgrind: $(valgrind --version 2>/dev/null)"
echo "build.sh: toolchain ready (stock tshark + valgrind; no from-source build)"
