# mruby #3722 (Row 15) — Use-After-Free in mrb_str_format

This is a minimal, deterministic reproduction of `mruby #3722` (Row 15 in `xlang-repro-task.md`), a heap Use-After-Free in `mrb_str_format` (`sprintf`) due to an argument-copying omission under stack reallocation.

## Vulnerability Overview
When a method receives variable arguments from the VM stack, they are retrieved as a pointer directly pointing into the VM register stack (`ARGV`). If formatting an argument triggers a callback (such as calling custom `#to_s` on an object) that extends and reallocates the VM register stack, the stack is moved, rendering the local argument array pointer (`argv`) in `mrb_str_format` stale. Subsequent iterations of formatting remaining arguments read from this stale pointer, producing a heap Use-After-Free.

## Contents
* `target.md` - Pinned versions and commit metadata
* `build_config.rb` - Unified build configuration for host native (ASan) and RISC-V cross-compilation
* `build.sh` - Automated build script to fetch, patch, and build mruby at the vulnerable commit
* `trigger.rb` - The minimal Ruby script triggering the UAF via custom `#to_s` lookup and deep recursion during `sprintf` formatting
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
The native execution under AddressSanitizer should abort with a `heap-use-after-free` in `mrb_str_format`.
The RISC-V QEMU execution will demonstrate anomalous behavior or crash.

`PASS = the sanitizer/QEMU shows the UAF at mrb_str_format`
