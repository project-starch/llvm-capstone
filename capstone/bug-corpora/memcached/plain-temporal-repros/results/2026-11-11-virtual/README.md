# memcached/plain-temporal-repros on the virtual profile, 2026-11-11

The `virtual-malloc` arm (`virtual-mallocng`), re-measured on the processor model with the
one-operand `CREVOKE` and `CRENEW`. The cases are plain `malloc` and `free`: what changed under them
is the instruction with which the virtual heap's `free` ends an object's lifetime
(`heap_alloc_end` names the object alone).

3 cases, CAUGHT 3: every verdict equals the 2026-10-11 bundle's. `bounds-malloc` and `uaf-malloc`
fault in the same boot.

Not in `inputs.json`: `CAPSTONE_QEMU_CPU_EXTRA=x-capstone-test-csrs=true,x-capstone-exact-bounds=true`; module loaded without parameters (`unpinned=0`, `supervisor_slots=1`). The SDK is a corpus SDK (`-fno-builtin`).
