# rlua #97 (Row 2 under Tier 3) — Use-After-Free in rlua

This is the reproduction outline and skip metadata for `rlua #97` (Row 2 under Tier 3 in `xlang-repro-task.md`), a heap Use-After-Free of userdata/callbacks in `rlua` via lifetime bypasses.

## Status: SKIPPED
In `rustc 1.96.1+` (2026), `std::mem::uninitialized` is strictly banned and triggers an immediate standard library runtime panic/abort. Because older, vulnerable `rlua` versions (such as rlua pre-v0.12.0) rely on `mem::uninitialized` inside their internal closure value-replacement mechanisms, compiling and executing them on a modern Rust toolchain crashes immediately on start, making verification of the callback lifetime bypass infeasible.

## Contents
* `target.md` - Pinned versions and skip technical rationale
* `Cargo.toml` - Rust project configuration
* `src/main.rs` - Rust entrypoint outline
* `build.sh` - Automated build script outline
* `run.sh` - Verification script outline
* `boundary.md` - Language boundary violation analysis
