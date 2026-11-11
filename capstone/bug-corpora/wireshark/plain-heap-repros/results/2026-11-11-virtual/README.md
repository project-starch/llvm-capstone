# wireshark/plain-heap-repros on the virtual profile, 2026-11-11

The `virtual-malloc` arm (`virtual-mallocng`), measured again on the processor model with the
one-operand `CREVOKE` and `CRENEW`, as the regression check of the converted mallocng: `free` ends
the lifetime with `CREVOKE(object)`.

| arm | configuration | controls (bounds-malloc, uaf-malloc) | cases | against `../2026-10-10-virtual` |
|---|---|---|---|---|
| `virtual-malloc` | `virtual-mallocng` | fault, fault, as declared | CAUGHT 12 / 12 | every verdict, cause and site equal |

One boot through `tools/run-virtual-cases.py --virtual-kit`, which builds the cases from `case.c`
with the corpus SDK (`-fno-builtin`, `-O0 -DCORPUS_VIRTUAL`). `inputs.json` carries the file
hashes of the kit. Two settings are not in it: the CPU properties
`CAPSTONE_QEMU_CPU_EXTRA=x-capstone-test-csrs=true,x-capstone-exact-bounds=true`, and the module
loaded without parameters (`unpinned=0`, `supervisor_slots=1`). N = 1 per cell.
