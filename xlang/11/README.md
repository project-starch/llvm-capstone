# CVE-2018-10191 (Row 11) — Use-After-Free in OP_GETUPVAR

This is a minimal reproduction outline and skip metadata for `CVE-2018-10191` (Row 11 in `xlang-repro-task.md`), a heap Use-After-Free in `OP_GETUPVAR` due to scope-level integer overflow.

## Status: SKIPPED
In modern sandbox environments, compiling deeply nested scopes (128+ levels) triggers compile-time parser memory limits or stack overflows before the virtual machine's runtime execution and integer overflow can be reached. This prevents a clean native ASan reproduction of the overflow.

## Contents
* `target.md` - Pinned versions and skip technical rationale
* `build_config.rb` - Unified build configuration
* `build.sh` - Automated build script
* `trigger.rb` - Trigger script outline
* `run.sh` - Verification script outline
* `boundary.md` - Language boundary violation analysis
