#!/bin/bash
# Build the guest's SSH server for capstone_vm (tools/run.sh passes it with --ssh-server): a static
# dropbear multi-call binary (dropbear + dropbearkey) from the manifest's source tarball.
# usage: ssh-server.sh KIT CROSS_COMPILE OUT   (OUT new: dropbearmulti, ssh-server-record.txt)
set -euo pipefail
KIT=$(cd "$1" && pwd) CROSS=$2 OUT=$3
[ -e "$OUT" ] && { echo "ssh-server: $OUT exists" >&2; exit 2; }
M=$KIT/manifest.lock.json
m() { python3 -c "import json,sys;d=json.load(open(sys.argv[1]));[d:=d[k] for k in sys.argv[2].split('.')];print(d)" "$M" "$1"; }
SRC=$KIT/artifacts/sources/$(m platform.ssh_server.archive.file)
got=$(sha256sum "$SRC" | cut -c1-64); [ "$got" = "$(m platform.ssh_server.archive.sha256)" ] || { echo "ssh-server: archive is $got" >&2; exit 1; }
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
tar -C "$W" -xjf "$SRC"; cd "$W"/dropbear-*
CC=${CROSS}gcc CFLAGS="-O2 -ffile-prefix-map=$W/=" ./configure --host=riscv64-buildroot-linux-gnu --disable-zlib --enable-static \
  --disable-lastlog --disable-utmp --disable-utmpx --disable-wtmp --disable-wtmpx > "$OUT/configure.log" 2>&1
make -j8 PROGRAMS="dropbear dropbearkey" MULTI=1 STATIC=1 > "$OUT/make.log" 2>&1
cp dropbearmulti "$OUT/dropbearmulti"
{ echo "dropbear $(m platform.ssh_server.archive.sha256) $(basename "$SRC")"; echo "cross $("${CROSS}gcc" --version | head -1)"
  echo "dropbearmulti $(sha256sum "$OUT/dropbearmulti" | cut -c1-64)"; } | tee "$OUT/ssh-server-record.txt"
