# target.md — GHSA-f56g-chqp-22m9 (Row 3 under Tier 3)

* **CVE/Issue:** GHSA-f56g-chqp-22m9 (hlua #144)
* **Product:** hlua (Lua-in-Rust)
* **Status:** SKIPPED
* **Technical Rationale:** Infeasible to reproduce under modern Rust. `hlua` is an old, unmaintained, and deprecated Rust-Lua binding. Due to extensive deprecations and breaking changes in Rust's macro-expansion compiler-plugins, procedurals, and trait systems over the last 9 years, compiling older `hlua` crates under a modern stable Rust `1.96.1+` toolchain is completely unsupported and unbuildable.
