# mruby #3829 (Row 9) — Use-After-Free in mrb_gc_mark

This is a minimal, deterministic reproduction of `mruby #3829` (Row 9 in `xlang-repro-task.md`), a heap Use-After-Free in `mrb_gc_mark` during string sweeping of forcefully freed `irep` pool strings.

## Vulnerability Overview
When strings are loaded or evaluated dynamically, they are stored in `irep` literal pools. Substrings taken from these pool strings are optimized as shared strings (`FSHARED`) that point directly to the pool string's raw data buffer on the heap. However, mruby did not track references to pool strings or the `irep` structures from these shared substrings. When the evaluated Proc goes out of scope and is swept, the `irep` is freed, forcefully deallocating all pool strings. This leaves the shared substring pointing to freed heap memory, triggering a heap-use-after-free during the next garbage collection marking or print sweep.

## Contents
* `target.md` - Pinned versions and commit metadata
* `build_config.rb` - Unified build configuration for host native (ASan) and RISC-V cross-compilation
* `build.sh` - Automated build script to fetch, patch, and build mruby at the vulnerable commit
* `trigger.rb` - The minimal Ruby script triggering the UAF via dynamic `eval` compilation, shared substring extraction, and garbage collection
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
The native execution under AddressSanitizer should abort with a `heap-use-after-free` in `mrb_gc_mark`.
The RISC-V QEMU execution will demonstrate anomalous behavior or crash.

`PASS = the sanitizer/QEMU shows the UAF at mrb_gc_mark`
