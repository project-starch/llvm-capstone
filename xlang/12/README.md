# mruby #4001 (Row 12) — Use-After-Free in File#initialize_copy

This is a minimal, deterministic reproduction of `mruby #4001` (Row 12 in `xlang-repro-task.md`), a heap Use-After-Free in `mruby-io`'s `File#initialize_copy` due to a dangling pointer left on TypeErrors.

## Vulnerability Overview
When `File#initialize_copy` is called, it first deallocates and frees the existing C `mrb_io` structure (`DATA_PTR`) of the receiver object. It then attempts to resolve the C structure of the source argument using `io_get_open_fptr`. If the source argument is not a valid IO object (e.g., passing `0` instead of a File), `io_get_open_fptr` raises a `TypeError` exception. This raises a longjmp abort before a new C structure can be assigned back to the receiver's `DATA_PTR`, leaving the receiver `File` object carrying a dangling pointer pointing to the forcefully freed C memory. Calling `File#close` on this dangling object triggers a heap-use-after-free.

## Contents
* `target.md` - Pinned versions and commit metadata
* `build_config.rb` - Unified build configuration for host native (ASan) and RISC-V cross-compilation with local `mruby-io` gem enabled
* `build.sh` - Automated build script to clone and build both mruby and mruby-io at their vulnerable commits
* `trigger.rb` - The minimal Ruby script triggering the UAF via invalid copy initialization and `close`
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
The native execution under AddressSanitizer should abort with a `heap-use-after-free` in `io_get_open_fptr` called during `mrb_io_close`.
The RISC-V QEMU execution will demonstrate anomalous behavior or crash.

`PASS = the sanitizer/QEMU shows the UAF at io_get_open_fptr/mrb_io_close`
