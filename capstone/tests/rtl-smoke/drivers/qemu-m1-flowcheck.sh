#!/bin/bash
# qemu-m1-flowcheck.sh -- the emulator flow check of the prepared M1 series (--series m1), four retention
# patterns at C=64: every arm must reach stop=target, minted = 31 + one per allocation, retained 0/16/640/0
# (release: 768 allocations), the snapshot lines carrying take_cyc/give_cyc/n/init_n, and stale_alias_type 7
# on the emulator (Q-11; 2 on silicon). run-r1-qemu.sh takes the QEMU lock itself -- never wrap this in flock --
# and writes the qemu-pass record for the image on R1_RC=0. Its boot.log for the drop arm is the M1_QEMU_LOG
# chain-m1.sh's gate reads ('R1 m1 end' >= 1).
#
#   R1_DOM=<r1_slots_pools.dom> R1_HOST=<sqlite_host_rr.user> [OUT_BASE=${CAPSTONE_ARTIFACTS}/qemu-m1] bash qemu-m1-flowcheck.sh
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
: "${R1_DOM:?}" "${R1_HOST:?}"
OUT_BASE=${OUT_BASE:-${CAPSTONE_ARTIFACTS:-$HOME/capstone-artifacts}/qemu-m1}; mkdir -p "$OUT_BASE"
cd "$R"
rm -f "$OUT_BASE/done"
for arm in drop ring pressure release; do
  R1_DOM=$R1_DOM R1_HOST=$R1_HOST OUT=$OUT_BASE/qemu-$arm bash capstone/sublet/r1/run-r1-qemu.sh "--arm $arm --series m1 --cap 64 --budget 60000" 2097152 > "$OUT_BASE/qemu-$arm.log" 2>&1
  echo "$arm rc=$?" >> "$OUT_BASE/done"
done
echo QEMU_M1_DONE >> "$OUT_BASE/done"
python3 - "$OUT_BASE" <<'EOF'
import sys, re, pathlib
b = pathlib.Path(sys.argv[1])
for arm in ["drop", "ring", "pressure", "release"]:
    t = (b / f"qemu-{arm}" / "boot.log").read_text(errors="replace")
    end = re.findall(r"R1 m1 end [^\r\n]*", t); snaps = len(re.findall(r"R1 m1 snap ", t))
    print(arm, "snapshots", snaps, "|", end[-1][:140] if end else "NO END LINE")
EOF
