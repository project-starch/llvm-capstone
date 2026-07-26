# mruby #3596 (Row 14) — Use-After-Free in mark_context_stack

This is a minimal, deterministic reproduction of `mruby #3596` (Row 14 in `xlang-repro-task.md`), a heap Use-After-Free in `mark_context_stack` due to stale pointers left in Uncleared stack regions under GC stress.

## Vulnerability Overview
When a method returns, the active stack pointer shrinks, but the registers in the now-unused region above the stack limit are NOT cleared. These registers continue to hold raw pointers to returned, discarded objects. If those objects are subsequently garbage collected and freed, the inactive stack region holds stale/dangling pointers. If another method call grows the stack over this region again, the stale registers are brought back into the active range. During the subsequent GC marking phase, `mark_context_stack` scans these stale registers, producing a heap Use-After-Free.

## Contents
* `target.md` - Pinned versions and commit metadata
* `build_config.rb` - Unified build configuration with AddressSanitizer and `MRB_GC_STRESS` enabled
* `build.sh` - Automated build script to fetch, patch, and build mruby at the vulnerable commit
* `trigger.rb` - The minimal Ruby script triggering the UAF via a loop that forces multiple allocations and GC stress cycles
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
The native execution under AddressSanitizer should abort with a `heap-use-after-free` in `mark_context_stack` during tests or driver execution.
The RISC-V QEMU execution will demonstrate anomalous behavior or crash.

`PASS = the sanitizer/QEMU shows the UAF at mark_context_stack`
