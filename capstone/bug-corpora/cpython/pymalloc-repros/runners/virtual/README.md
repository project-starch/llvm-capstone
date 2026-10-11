# Virtual Capstone

The twenty cases as Capstone applications on the virtual profile (`capstone/runtime/virtual`):
each `defect-NN` is the hosted replay (`ports/cpython/pymalloc/src/native/main.c` around the
case's `PYC_CASE` body), run as a process under `capstone-vexec` on a persistent VM. malloc is
virtual mallocng, exact bounds plus lifetime retirement on free. The physical Sublet heap is not
used.

| arm | configuration (`tools/arms.json`) | build | what bounds a pymalloc block |
|---|---|---|---|
| `virtual-malloc` | `virtual-pymalloc` | `-DPYMALLOC_SUBLET=OFF` | nothing: the arena is one mallocng object |
| `virtual-nested-pools` | `virtual-pymalloc-pools` | `-DPYMALLOC_SUBLET=ON`, mode 1 | the lifetime adapter, one capability per block, retired on free |

In the pools arm the virtual heap lends the arena as one linear capability
(`__capstone_sublet_malloc_linear`), and `ports/common/include/borrow-aligned-block.h` carves the
16 KiB pool alignment the adapter requires out of it.

## Build

    export CAPSTONE_SDK=<virtual application SDK>
    shared/build-cases.sh capstone-application OUT-OFF -DPYMALLOC_SUBLET=OFF
    shared/build-cases.sh capstone-application OUT-ON  -DPYMALLOC_SUBLET=ON
    PYC_CASES=$PWD/controls shared/build-cases.sh capstone-application CTL-OFF -DPYMALLOC_SUBLET=OFF
    PYC_CASES=$PWD/controls shared/build-cases.sh capstone-application CTL-ON  -DPYMALLOC_SUBLET=ON

## Run

    capstone_vm --state VM up --profile virtual --exact-bounds ...
    runners/virtual/run-virtual.py results/<stamp>/virtual-malloc --arm virtual-malloc \
        --state VM --build OUT-OFF --controls-build CTL-OFF --llvm-bin <llvm>/bin --raw RAW
    runners/virtual/run-virtual.py results/<stamp>/virtual-nested-pools --arm virtual-nested-pools \
        --state VM --build OUT-ON --controls-build CTL-ON --llvm-bin <llvm>/bin --raw RAW

The runner refuses a build whose `CMakeCache.txt` does not match the arm, a non-virtual SDK, or a
non-virtual VM. Each case gets a one-event fixture naming it (`struct pym_header` and one
`struct pym_event` whose id is the case number) through the VM share.

## What it observes, and who decides

The runner reports; `tools/verdicts.py` decides. Reached and completed come from
`PYM completed=1`, printed only after the case body has run its `read_probe`. A fault counts as
the defect's only when its pc resolves to `read_probe`, the function that carries the
`pyc_defect_read` label. Before the cases, `controls/00_control_uaf_block` must complete on the
stock arm and fault in `read_probe` on the pools arm. Otherwise every row is NO-READING.

`tools/derive-verdicts.py` writes the verdicts into each `case.json` from the bundles that
`corpus.json` `verdict_bundles` names, and `check-corpus.py` re-judges them.
