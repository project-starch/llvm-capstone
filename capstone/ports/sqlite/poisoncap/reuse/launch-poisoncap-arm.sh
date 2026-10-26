#!/usr/bin/env bash
# launch-poisoncap-arm.sh VARIANT PORT WORK: PoisonCap's wrapped speedtest1 (no arguments), E3c protocol.
# The E3c protocol: kernel with their FPGA kernel options, libc revocation default 0 at boot (so sshd starts
# and the binary is copied in), set to 1 before the run; stdout to a file, no polling.
cd ~/poisoncap-platform/repo/capstone/ports/sqlite/poisoncap/speedtest1
S=~/poisoncap-platform/speedtest1
exec python3 speedtest1-guest.py "$S/reuse-size100/$3" --binary "$S/reuse-$1/speedtest1" --port "$2" \
  --kernel-root ~/poisoncap-platform/kernels/poison-nodebug --revocation-default-off --timeout 28800 \
  --pre "sysctl security.cheri.runtime_revocation_default=1"
