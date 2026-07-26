# CVE-2022-1934 — mruby Use-After-Free in hash_new_from_values

This is a minimal, deterministic reproduction of `CVE-2022-1934`, a heap Use-After-Free in mruby's VM execution loop during keyword argument packing.

## Vulnerability Overview
When packing keyword arguments via `hash_new_from_values`, the interpreter iterates through a raw `regs` pointer to retrieve key-value pairs and insert them into a hash. If hash insertion of a key invokes a user-defined `eql?` method on a custom object key, and that method triggers a recursive stack extension, the VM stack is reallocated and moved. The raw `regs` pointer in `hash_new_from_values` becomes stale. Subsequent loop iterations read next key-value pairs from this stale pointer, resulting in a heap-use-after-free.

## Contents
* `target.md` - Pinned versions and commit metadata
* `build_config.rb` - Unified build configuration for host native (ASan) and RISC-V cross-compilation
* `build.sh` - Automated build script to fetch and build mruby at the vulnerable commit
* `trigger.rb` - The minimal Ruby script triggering the UAF via custom `eql?` lookup and deep recursion during keyword packing
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
The native execution under AddressSanitizer should abort with a `heap-use-after-free` in `hash_new_from_values`.
The RISC-V QEMU execution will demonstrate anomalous behavior or crash.

`PASS = the sanitizer/QEMU shows the UAF at hash_new_from_values`
