#!/usr/bin/env bash
# R-18: an ordinary `sw` whose DATA REGISTER carries capability metadata also writes its data into
# the same byte lanes of the OTHER bank of its 16-byte row, silently overwriting an unrelated
# scalar 8 bytes away.
#
#   ./run.sh sim     RTL simulation, ~13 s, NO BOARD  <- start here
#   ./run.sh trigger board, the four-way control that establishes the trigger
#   ./run.sh geom    board, the original row-offset pair plus the sentinel arm
#
# The images are FROZEN and checksummed on purpose. The effect depends on where the -O0 allocator
# places a scalar, so rebuilding can move it and cure the fault -- which is how several instruments
# in this investigation destroyed what they were measuring. Every board mode below runs the
# known-good control k800 FIRST; a boot whose control fails carries no verdict about anything.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
D=capstone/tests/fpga-repros/R18-scalar-store-metadata-clobber
MODE=${1:-sim}

echo "== verifying frozen artifacts =="
( cd "$D" && sha256sum -c SHA256SUMS >/dev/null ) \
  || { echo "CHECKSUM MISMATCH -- do not trust a run"; exit 1; }
echo "   ok, $(ls "$D"/src/*.dom | wc -l) images + $(ls "$D"/sim | wc -l) simulation artifacts"

# ---------------------------------------------------------------- simulation
if [[ "$MODE" == "sim" ]]; then
  cat <<'EOF'

== RTL SIMULATION -- this is the fastest and most direct evidence ==

Three directed tests, identical geometry and loop, differing ONLY in what produced the store's
data register. Copy them into the RTL tree and run each (~13 s):

    cp capstone/tests/fpga-repros/R18-.../sim/*.S \
       capstone/capstone-ariane/verif/tests/custom/capstone/
    # add a testlist entry per test, copying the `stc` entry in
    #   capstone/capstone-ariane/verif/tests/testlist_capstone.yaml
    cd capstone/capstone-ariane
    docker run --rm -v "$(pwd)":/workdir --user "$(id -u):$(id -g)" --entrypoint bash \
      -e HOME=/tmp -e RISCV=/toolchain -e CVA6_REPO_DIR=/workdir cva6-build-rv -c '
      cd /workdir
      source verif/regress/install-verilator.sh >/dev/null 2>&1
      source verif/regress/install-spike.sh     >/dev/null 2>&1
      source verif/sim/setup-env.sh             >/dev/null 2>&1
      cd verif/sim && python3 cva6.py --testlist=../tests/testlist_capstone.yaml \
        --test scalar-store-movc-zero --iss_yaml cva6.yaml \
        --target capstone_cv64a6_imafdc_sv39 --iss=veri-testharness \
        --issrun_opts=+debug_disable=1+UVM_VERBOSITY=UVM_NONE --issrun_opts=+time_out=2000000'

Expected:
    scalar-store-movc-zero.S          FAILED  tohost=3   witness at +0x10 zeroed
    scalar-store-realcap-samegeom.S   FAILED  tohost=3   same
    scalar-store-addi-zero.S          SUCCESS            <- the workaround

All three finish in ~12810 cycles. A `SUCCESS` printed AT the +time_out value is NOT a pass --
compare the cycle count. Delete stale artifacts before each run:
    rm -f verif/sim/out_*/veri-testharness_sim/<name>*

sim/movc-zero-rvfi-trace.dasm is the trace from the movc-zero run. It shows only TWO architectural
accesses to the corrupted slot in the entire run -- the seed writing 0x0a0a0a0a, and the readback
returning 0x00000000. Nothing wrote zero to it.
EOF
  exit 0
fi

# ---------------------------------------------------------------- board modes
: "${FPGA_URL:?set FPGA_URL from the secrets file (never commit or echo it)}"
: "${FPGA_FW:?set FPGA_FW to the built fw_payload.bin}"
: "${FPGA_BITSTREAM:=caplifive_65536_nodes.bit}"; export FPGA_BITSTREAM

O=capstone/caplifive-system/sw/buildroot/overlay/test-domains
T=capstone/caplifive-system/sw/buildroot/build/target/test-domains
export LADDER_FPGA_DIR=${LADDER_FPGA_DIR:-/tmp/capstone/ladder-fpga}
mkdir -p "$LADDER_FPGA_DIR" "$T"

case "$MODE" in
  trigger)
    RUNGS="k800 c8 gz0 gzs"
    cat <<'EOF'

== THE TRIGGER, on the board ==
   k800  control, returns 4
   c8    anchor, returns 67699255 (qc=567) -- has done so on 14 consecutive boots
   gz0   row-mate reset by `movc a0, zero; sw`   -> victim 9   DAMAGED
   gzs   row-mate reset by `lui; addi; sw`       -> victim 576 clean
   The full four-way control adds gzn (movc + nops, damaged) and gzl (stores the VALUE
   zero from a load, clean); the 4-slot budget takes them one boot at a time.
EOF
    for r in gz0 gzs; do echo 590400 > "$LADDER_FPGA_DIR/$r.oracle"; done
    echo 67699264 > "$LADDER_FPGA_DIR/c8.oracle"
    ;;
  geom)
    RUNGS="k800 c0 c8 sn8"
    cat <<'EOF'

== THE GEOMETRY PAIR, on the board ==
   c0   0x04090240 (p=64 k=9 qc=576) CORRECT  -- accumulator in the LOWER half
   c8   0x04090237 (p=64 k=9 qc=567) DEFECT   -- accumulator in the UPPER half
   sn8  567                          DEFECT   -- sentinel start, upper half
   sn8 starts at 1,000,000. Lost increments would give 1000567; 567 means the slot was
   OVERWRITTEN and counted up -- which the splash mechanism predicts, since the splash
   carries the store's DATA (zero), not its metadata.
EOF
    python3 -c "print((64<<20)|(9<<16)|576)" > "$LADDER_FPGA_DIR/c0.oracle"
    python3 -c "print((64<<20)|(9<<16)|576)" > "$LADDER_FPGA_DIR/c8.oracle"
    echo 1000576 > "$LADDER_FPGA_DIR/sn0.oracle"
    echo 1000576 > "$LADDER_FPGA_DIR/sn8.oracle"
    ;;
  *) echo "usage: $0 [sim|trigger|geom]"; exit 2 ;;
esac

echo
echo "== staging: $RUNGS =="
for r in $RUNGS; do
  [[ "$r" == "k800" ]] && continue          # k800 is package-installed, not ours to stage
  cp -f "$D/src/$r.dom" "$O/" && cp -f "$D/src/$r.dom" "$T/" || exit 1
done

cat <<'EOF'

REBUILD THE FIRMWARE BEFORE RUNNING, in this order, or the OLD initramfs ships and the runner
will misreport the missing domains as R-16 entry stalls:
  make build LINUX_PAYLOAD=1 A=linux-rebuild  ...   # FIRST
  make build LINUX_PAYLOAD=1 A=opensbi-rebuild ...   # THEN
then confirm the firmware HASH changed -- never the size; buildroot pads in 2 MiB steps.
`capstone/tests/preflight-board-run.sh` checks initramfs membership and the recorded QEMU pass.
EOF

echo
echo "== board run: control first =="
( cd capstone/tests/rtl-smoke \
  && BAKED_RUNGS="$RUNGS" BAKED_TIMEOUT=150 python3 -m fpga_driver.run_baked_rungs_fpga )
