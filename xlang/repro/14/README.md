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
Native under AddressSanitizer: aborts with `heap-use-after-free` at
`src/gc.c:556` in `mark_context_stack`, freed in `incremental_sweep_phase`.
Deterministic — 10/10 consecutive runs.

RISC-V QEMU (`-O3`, gcc cross-build, no sanitizer): the trigger **runs to
completion and exits 0**. The stale register read lands on memory the allocator
has not yet reused, so the UAF is benign without ASan instrumentation. This is
expected and is not a reproduction failure; the native ASan run is the
memory-safety evidence for this row.

`PASS = native ASan shows the UAF at mark_context_stack`
