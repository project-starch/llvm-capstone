# Mapping registry sorted highest address first: gates against the commit it stacks on

Two arms on one host, one kit, one compiler, one QEMU, firmware, kernel `Image`, rootfs and one
module; only the launcher (`capstone-vexec`) differs, because the change is only in `exec.c`.

- control: the launcher of the commit below this one (the address-indexed registry), `fa3e5f06b07c5cfa`
  (the same hash as the launcher of the matched pthread run in `results/pthread-matched-250/`).
- this change: `09d3431ca8fa21cd`.
- shared: `capstone_vm.ko` `22aa0ffef1b56c5a` (flat-table module, `supervisor_slots` 32 in the gate
  copy), QEMU `bd6cd022b07794cf` (`b290f5656600`, `svadu=false`, `x-capstone-supervisor-slots=32`),
  `fw_jump.elf` `4dd4a2ade361a8bd`, `Image` `e833dfb12bccf8a3`, `rootfs.ext2` `62049800ed0fa882`,
  and the images of every gate (SDK, ports, contracts and runners from this tree; the compiler is
  the SPEC toolchain v4 clang, which is dev plus the then-open PRs, not dev alone). The image hashes
  of the two arms are identical.
- launchers cross-built with the buildroot `riscv64-buildroot-linux-gnu-gcc -O2 -static ... -Werror`
  command of `build-adapter.sh`; the control build reproduced the earlier launcher byte for byte.

| gate | control failing checks | this change | difference in either direction |
|---|---|---|---|
| `run-hosted.py` | 0 | 0 | none |
| `run-pthreads.py` | 0 | 0 | none |
| `run-nodes.py` | 10 | 10 | none |
| `run-malloc.py` | 6 | 6 | none |
| `run.py` | 8 | 8 | none |

`run.py`'s `vm-mappings` and `vm_mappings_contract` are `true` in both arms, and both arms print
`VIRTUAL_VM_MAPPINGS_OK 9000`, `VIRTUAL_VM_MAPPINGS_MOVED 8` and `VM_EXIT:vm-mappings:0`.

The nodes, malloc and runtime failures are the ones the commit below also has; they are not examined
here. A comparison of failing sets is blind inside a check that fails on both sides.

Not run: `run-step-hooks.py`, `run-pin-ahead.py`, `run-fp.py`, the memcached gate, the board kernel
build, any run on silicon. Dated 2026-10-21; the per-run logs are on the build host, not in the
repository. Wall time per gate for this change: hosted 20 s, pthreads 29 s, nodes 33 s, malloc 66 s,
`run.py` 336 s (the control ran later alongside other QEMU jobs and took longer).

`test-registry.sh` (host test, no module or hardware) runs the real registry functions from `exec.c`
through 200,000 random insertions, removals, moves and lookups against a model, and through
top-down, near-the-end-removal and bottom-up sweeps and a 6000-registration growth phase. It passes
on this commit and fails on: the previous ascending-order registry, a flipped floor comparison, an
insertion point off by one, a floor that compares only the low 32 bits of an address, an insert that
drops or rewrites payload fields, a removal that moves one entry too few, a reserve that never grows
past 2048 entries, and a find that matches an address inside a mapping.
