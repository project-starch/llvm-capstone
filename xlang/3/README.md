# GHSA-f56g-chqp-22m9 (Row 3 under Tier 3) — Use-After-Free in hlua

This is the reproduction outline and skip metadata for `GHSA-f56g-chqp-22m9` / `hlua #144` (Row 3 under Tier 3 in `xlang-repro-task.md`), a heap Use-After-Free of userdata in `hlua` due to incorrect lifetime bindings.

## Status: SKIPPED
`hlua` is an old, unmaintained, and deprecated Rust-Lua binding. Due to extensive deprecations and breaking changes in Rust's macro-expansion compiler-plugins, procedurals, and trait systems over the last 9 years, compiling older `hlua` crates under a modern stable Rust `1.96.1+` toolchain is completely unsupported and unbuildable.

## Contents
* `target.md` - Pinned versions and skip technical rationale
* `Cargo.toml` - Rust project configuration
* `src/main.rs` - Rust entrypoint outline
* `build.sh` - Automated build script outline
* `run.sh` - Verification script outline
* `boundary.md` - Language boundary violation analysis
