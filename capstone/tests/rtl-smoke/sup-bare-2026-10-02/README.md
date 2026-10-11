# Supervised CALL on silicon: the RTL lane's directed tests, run bare on the board

The RTL lane's supervised-CALL tests (`capstone-ariane` `verif/tests/custom/capstone/`, commit `36a641e0b`) run on the
FPGA as bare M-mode programs, with no OpenSBI and no Linux. Each image is JTAG-loaded at `0x80000000` after
`monitor reset halt`, which is the same entry the simulation testbench uses.

- `tests/`: `sup-quantum.S`, `sup-escape.S`, `call-retpc.S`, byte-identical to `36a641e0b`.
- `inc/`: the only board-specific part.
  - `asm_insn.h` is found before the original (kept as `asm_insn_orig.h`, byte-identical) and redefines two testbench hooks:
    - `CAPPRINT(r)` appends `r` to `board_rec[]` through a fresh `CAPCREATE`d capability;
    - `CAP_PASS` prints `board_rec[]` over the 16550 UART (`SR <count>`, `SV <value>`, `SUPTEST END`).
  - Both expand INLINE: `CAPENTER(_start, _end_of_text)` bounds the PC capability to the test's code, so out-of-line
    harness code would not be fetchable.
  - `board_prologue.S` runs first, in integer mode. It sets up the UART (25 MHz, 57600 8N1, reg shift 2) and prints
    `SUPTEST BEGIN <id>`. It then rewrites `.data` with core stores, so a stale DDR tag cannot survive
    (`wt_dcache_mem.sv:300` at `36a641e0b`: "An entry covering this granule and carrying ctag=0 clears the
    granule's tag"), zeroes `.bss`, prints `SB1` and jumps to `_start`.
  - No node setup is needed: the rev-node unit's `INIT_STAGE` writes nodes 0..2 at every reset
    (`capstone_rev_node.anvil:325-336`, `:383`). According to an rtl-oracle reading, `monitor reset halt` reaches
    that reset through the debug module's ndmreset: `riscv-dbg dm_csrs.sv` forces hartreset to 0, and
    `ariane_xilinx.sv` feeds ndmreset into the core's rst_ni.
- Results: `results/board-36a641e0b.result-lines.txt`, one run each. All three board vectors are bit-identical to
  the RTL lane's simulation of the same tree. `sup-escape` shows fault-event delivery, NOT that the trap is stripped
  (see the result lines).
- `build.sh` rebuilds `images/` byte for byte (checked: `sha256sum -c images/SHA256SUMS` after a rebuild). It is
  self-contained: `env/` holds the riscv-tests headers the run used. These are capstone-ariane's riscv-tests
  checkout, env `2f75dc2`, whose `p/riscv_test.h` carries local modifications. Only the compiler is external
  (`CAPSTONE_LLVM_BIN`, default `<repo>/llvm/cmake-build-debug/bin`).
- `run_sup_bare.py` runs one image per board session (env: `FPGA_URL`, `FPGA_BITSTREAM`, `SUP_IMG`, `SUP_REC_ADDR`,
  `SUP_OUT`).
- Pre-registration: `PREREG.md`, mtime-attested before the first boot but not pushed before it. Its "simulation
  read N = 52" was stale; the tree's simulation reads 53, as the board did.
