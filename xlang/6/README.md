# CVE-2026-1979 (Row 6) — Use-After-Free in mrb_vm_exec

This is the reproduction outline and skip metadata for `CVE-2026-1979` (Row 6 in `xlang-repro-task.md`), a heap Use-After-Free in `mrb_vm_exec` due to pattern-matching bytecode JMPNOT-to-JMPIF optimization corruption.

## Status: SKIPPED
CVE-2026-1979 is the CVE identifier assigned to mruby issue #6701 (Row 7). Compiling and executing mruby 3.4.0 with AddressSanitizer (ASan) triggers severe, boot-time GC stack-scanning failures (prematurely sweeping core classes like `Enumerable`), preventing the ASan-instrumented interpreter from booting successfully and making validation of the bytecode corruption infeasible.

## Contents
* `target.md` - Pinned versions and skip technical rationale
* `build_config.rb` - Unified build configuration
* `build.sh` - Automated build script
* `trigger.rb` - Trigger script outline
* `run.sh` - Verification script outline
* `boundary.md` - Language boundary violation analysis
