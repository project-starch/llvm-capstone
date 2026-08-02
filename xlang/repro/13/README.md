# mruby #4927 (Row 13) — Use-After-Free in hash_slice

This is a minimal, deterministic reproduction of `mruby #4927` (Row 13 in `xlang-repro-task.md`), a heap Use-After-Free in `mruby-hash-ext`'s `Hash#slice` due to an argument-copying omission under stack reallocation.

## Vulnerability Overview
When a method receives arguments from the VM stack, they must be copied off the stack to protect against stack reallocation if a callback re-allocates the VM stack. A logical inversion bug in `mrb_get_args()`'s copying condition omitted this protection when arguments were on the stack. When `hash_slice` iterates through its arguments, any hash lookup callback (like calling `#eql?` on a custom key) that triggers a recursive stack extension will invalidate the local `argv` pointer, causing subsequent iterations to read from freed stack memory.

## Contents
* `target.md` - Pinned versions and commit metadata
* `build_config.rb` - Unified build configuration for host native (ASan) and RISC-V cross-compilation
* `build.sh` - Automated build script to fetch, patch, and build mruby at the vulnerable commit
* `trigger.rb` - The minimal Ruby script triggering the UAF via custom `#eql?` lookup and deep recursion during `slice` lookup
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
The native execution under AddressSanitizer should abort with a `heap-use-after-free` in `hash_slice`.
The RISC-V QEMU execution will demonstrate anomalous behavior or crash.

`PASS = the sanitizer/QEMU shows the UAF at hash_slice`
