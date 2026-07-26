# gp-free domain — silicon smoke runbook (Experiment A on the board)

Goal: run the gp-free globals-using domain on the **real CVA6/Capstone RTL** to
(1) empirically confirm the cscratch `gp` delivery works on hardware (settling the
cscratch-vs-ctvec question), and (2) get the Experiment-A cycle number. The whole
chain is already proven functionally on QEMU
(`history/22-07-2026_16-09-12_gp-free-domain-bringup-qemu-proof.md`); this is the
silicon port.

## Prereqs (state)

- Branch `capstone-gp-free`: gp-free compiler committed (`88054a14`).
- **FPGA monitor change applied (uncommitted, local experiment):**
  `caplifive-system/sw/buildroot/components/opensbi/lib/sbi/capstone-sbi/sbi_capstone.c`
  `create_domain` mints `gp` with `C_GEN_CAP` and stashes it at the cscratch
  region top slot (identical to the QEMU/buildroot copy already validated).
- Board access per `ref/fpga-borrow-cost-reproduction.md` (browser/websocket +
  the board-URL token in the `FPGA_URL` env var for one run only — never committed,
  never persisted to disk).

## Step 1 — rebuild the FPGA firmware (fw_payload) with the monitor change

The `.c.S` is compiled by capstone-c; its pattern rule watches the wrapper, not
the `#include`, so force-regen. FPGA platform + Linux payload:

```bash
cd capstone/caplifive-system/sw/buildroot
rm -f components/opensbi/lib/sbi/sbi_capstone_dom.c.S \
      components/opensbi/lib/sbi/capstone_int_handler.c.S
make build A=opensbi-rebuild CAPSTONE_CC_PATH="$(realpath <capstone-c>)"
# If the payload comes out kernel-less (~2MB), force the OpenSBI relink for fpga:
make -C build/build/opensbi-custom PLATFORM=fpga/ariane \
     CROSS_COMPILE=<...>/riscv64-buildroot-linux-gnu- LINUX_PAYLOAD=1
# a correct fw_payload is ~15MB (see fpga-borrow-cost-reproduction.md).
```

## Step 2 — build the gp-free domain for the board

Reuse the committed proof domain (globals + call graph), gp-free glue + flag:

```bash
source capstone/tests/capstone-test-env.sh
START_SRC=capstone/tests/runtime-qemu/gp-free-domain/start-gpfree-cscratch.S \
EXTRA_CLANG_FLAGS="-mllvm -capstone-gp-free" DOMAIN_OPT_LEVEL=-O0 \
  bash capstone/tests/runtime-qemu/build-domain.sh \
    capstone/tests/runtime-qemu/gp-free-domain/gpfree_app.c /tmp/gpfree_app.dom
# static gate: cjalr=0, cincoffset-gp=0, scc-gp>=1
```

A board controller (`.user`) is also needed to create+call the domain and read
the result back — adapt the freestanding soft-float controller pattern from
`tests/rtl-smoke/borrow_cost_fpga_nogp_ctl.c` (raw Linux syscalls, integer-only,
DPI create/call, prints the retval). The domain writes its result to the passed
region cap; the controller prints it.

## Step 3 — onto the board + run

Transfer via the reconnect-resilient UART driver (gzip+base64, per-chunk sha) —
see `/tmp/capstone/board_run_nogp.py` pattern and `ref/fpga-borrow-cost-*`. Run
`<ctl> <gpfree_app.dom>`. gdb-boot; the built-in image already exposes
`/dev/capstone`.

**Expected:** the domain creates, enters (`gp` delivered via cscratch — no
fabrication, no ctvec), runs the globals + call-graph workload, writes the result,
`domreturn`s, and the controller prints `retval = 554745961` (`0x2110C069`). That
is the silicon confirmation of cscratch `gp` delivery. For the cycle number, wrap
the workload in `mcycle` reads (see `tests/rtl-smoke/fpga_instrument.h`).

## Notes / risks (from prior board sessions)

- `domreturn` reset at high revoke counts is a temporal-op issue and irrelevant to
  Experiment A (spatial `SHRINK` allocates no rev-nodes); a plain integer domain
  should exit cleanly.
- Board is flaky (websocket drops, ~10 min/cycle, no HW breakpoints). Use the
  UART-banner stall-probe method, not `hbreak`.
- The FPGA monitor edit stays a local experiment (no submodule-source commit);
  reproduce from the snippet in the gp-free-domain README.
