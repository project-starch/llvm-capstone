# target.md — rlua #97 (Row 2 under Tier 3)

* **CVE/Issue:** rlua #97
* **Product:** rlua (Lua-in-Rust)
* **Status:** SKIPPED
* **Technical Rationale:** Infeasible to reproduce under modern Rust. In `rustc 1.96.1+` (2026), `std::mem::uninitialized` is strictly banned and triggers an immediate standard library runtime panic/abort. Because older, vulnerable `rlua` versions (such as rlua pre-v0.12.0) rely on `mem::uninitialized` inside their internal closure value-replacement mechanisms, compiling and executing them on a modern Rust toolchain crashes immediately on start, making verification of the callback lifetime bypass infeasible.
