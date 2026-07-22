# gp-free domain — real app in a pure-capability domain, silicon-shaped ABI

A real integer app (`gpfree_app.c`: `.rodata`/`.data` globals + a non-inlined
`helper()` call graph) built and run as a **gp-free / cjalr-free** domain — the
enabling work for running a real globals-using program on silicon (Experiment A,
`agent-handoff/plans/compatibility-eval-silicon-app.md` §2).

## Why gp-free

Our LLVM backend normally reaches every global via `cincoffset gp, <abs>` (needs
`gp = PCC(cursor 0)`) and calls/returns via `cjalr`. On the RTL:

- `gp` is **not** hardware-restored on `cscall`, and an image-bounded cap forced
  to cursor 0 is **unrepresentable** under capability compression (board owner,
  2026-07-22). So `gp = PCC(cursor 0)` cannot exist on silicon — our QEMU fork
  only makes it work by *fabricating* it (patch `7aca0540`).
- `cjalr` needs a code capability, unformable gp-free on a fresh domain entry.

`-capstone-gp-free` (LLVM, default off, byte-identical when off) fixes both:

- **calls/returns** → plain `jal`/`jalr` within PCC (bounds-checked on fetch);
- **global data** → `scc gp, &g` (set cursor to the object's absolute VA — an
  *in-bounds*, representable cursor) + the usual per-object `SHRINK`. `gp` is an
  image-covering data cap the **monitor delivers**, never fabricated.

## gp delivery (cscratch — the board owner's confirmed channel)

The monitor's `create_domain` mints an image-covering data cap with `C_GEN_CAP`
(in-bounds cursor) and stores it in the **top-16 slot of the domain's stack
(cscratch) region**; the entry glue (`start-gpfree-cscratch.S`) loads it
(`ldc gp, END-16`) and `delin`s it. This matches the reference compiler
`capstone/capstone-c`, which also delivers via cscratch. No dependence on `ctvec`.

The `create_domain` change is a submodule (caplifive-buildroot / caplifive-system
OpenSBI) edit — kept as a local experiment, not committed to submodule source.
To reproduce the QEMU monitor:

```c
// in create_domain(), after C_SET_CURSOR(dom_code,...), before sealing:
{
    unsigned d_end = base_addr + tot_size;
    __linear void *dom_gp;
    C_GEN_CAP(dom_gp, base_addr, base_addr + code_size);
    __linear void *gp_slot;
    C_GEN_CAP(gp_slot, base_addr + code_size + DOMAIN_DATA_SIZE, d_end);
    C_SET_CURSOR(gp_slot, gp_slot, d_end - 16);
    *(__linear void **)gp_slot = dom_gp;
}
```

Rebuild (buildroot; the `.c.S` is compiled by capstone-c, and its pattern rule
watches the wrapper not the `#include`, so force-regen):

```bash
cd capstone/caplifive-buildroot
rm -f components/opensbi/lib/sbi/sbi_capstone_dom.c.S \
      components/opensbi/lib/sbi/capstone_int_handler.c.S
make build A=opensbi-rebuild CAPSTONE_CC_PATH="$(realpath ../capstone-c)"
```

The QEMU fabrication guard lives in `capstone-qemu` `op_helper.c`
(`CAPSTONE_GP_FABRICATE=0` disables it; `CAPSTONE_GP_STANDIN=1` makes QEMU deliver
a representable gp as a monitor stand-in when the image is not rebuilt).

## Run

```bash
# With the monitor image rebuilt (real cscratch delivery, hack off):
bash build-and-run.sh
# Without rebuilding the monitor (QEMU delivers a representable gp):
CAPSTONE_GP_STANDIN=1 bash build-and-run.sh
```

Expected: static `cjalr=0 cincoffset-gp=0 scc-gp>=1`, and
`Called dom (1-th time) retval = 554745961` (`0x2110C069`) →
`__CAPSTONE_GPFREE_DOMAIN_PASSED__`.

## Status (2026-07-22)

Proven end-to-end on QEMU: real app runs correctly gp-free with the monitor
delivering `gp` via cscratch and the fabrication **off**; default domains still
pass with the rebuilt monitor. Remaining: the same monitor change on the FPGA
(caplifive-system) copy + a board run for the cycle number.
