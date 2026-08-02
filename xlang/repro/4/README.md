# CVE-2022-1071 — mruby Use-After-Free in OP_GETCONST

This is a minimal, deterministic reproduction of `CVE-2022-1071`, a heap Use-After-Free in mruby's VM execution loop during `OP_GETCONST` handling.

## Vulnerability Overview
When retrieving a constant via `OP_GETCONST`, the VM execution loop evaluates the stack destination address before executing the right-hand side constant-lookup function. If lookup triggers a user-defined `const_missing` callback that reallocates the VM register stack (e.g., via deep recursion), the stack is moved and the old stack's memory is freed. Upon returning, the VM writes the constant value back to the now-stale destination pointer on the old stack.

## Contents
* `target.md` - Pinned versions and commit metadata
* `build_config.rb` - Unified build configuration for host native (ASan) and RISC-V cross-compilation
* `build.sh` - Automated build script to fetch and build mruby at the vulnerable commit
* `trigger.rb` - The minimal Ruby script triggering the UAF via missing constant lookup and deep recursion
* `run.sh` - Runs the trigger natively under ASan and under RISC-V QEMU
* `asan.txt` - Scrubbed ASan crash report proving the heap use-after-free
* `boundary.md` - Language boundary violation analysis

## How to Build and Run
To build the vulnerable mruby and compile targets for both native and RISC-V:
```bash
chmod +x build.sh run.sh
./build.sh
```

To run the reproduction and capture output:
```bash
./run.sh
```

## Expected Outcome
The native execution under AddressSanitizer should abort with a `heap-use-after-free` in `mrb_vm_exec`.
The RISC-V QEMU execution will demonstrate anomalous behavior or crash.

`PASS = the sanitizer/QEMU shows the UAF at mrb_vm_exec`
