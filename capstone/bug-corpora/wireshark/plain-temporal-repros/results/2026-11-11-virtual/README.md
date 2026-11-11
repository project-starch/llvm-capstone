# wireshark/plain-temporal-repros on the virtual profile, 2026-11-11

The `virtual-malloc` arm (`virtual-mallocng`), measured again on the processor model with the
one-operand `CREVOKE` and `CRENEW`, as the regression check of the converted mallocng: `free` ends
the lifetime with `CREVOKE(object)`.

| arm | configuration | controls (bounds-malloc, uaf-malloc) | cases | against `../2026-10-10-virtual` |
|---|---|---|---|---|
| `virtual-malloc` | `virtual-mallocng` | fault, fault, as declared | CAUGHT 10 / 10 | every verdict and site equal; the cause is 25 where that bundle has 24 |

One boot through `tools/run-virtual-cases.py --virtual-kit`, which builds the cases from `case.c`
with the corpus SDK (`-fno-builtin`, `-O0 -DCORPUS_VIRTUAL`). `inputs.json` carries the file
hashes of the kit. Two settings are not in it: the CPU properties
`CAPSTONE_QEMU_CPU_EXTRA=x-capstone-test-csrs=true,x-capstone-exact-bounds=true`, and the module
loaded without parameters (`unpinned=0`, `supervisor_slots=1`). N = 1 per cell.

## The cause code

Every case faults at `wst_read_probe`, as in the 2026-10-10 bundle, with cause 25 (a reference to
an ended lifetime) where that bundle records cause 24. The conversion did not change it:
`dev-control/virtual-malloc` is the same corpus built with the unconverted tree's SDK and run on
that tree's platform (the model with the two-operand `CREVOKE`, `inputs.json`: qemu `35529815bf4b`,
CPU line with `x-capstone-supervisor-slots=32` for its module), and it reads CAUGHT 10 / 10 with
cause 25 at the same probe.
