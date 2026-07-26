# target.md — rlua #19 (Row 1 under Tier 3)

* **CVE/Issue:** rlua #19
* **Product:** rlua (Lua-in-Rust)
* **Status:** SKIPPED
* **Technical Rationale:** Infeasible to reproduce under modern Rust. In `rustc 1.96.1+` (2026), `std::mem::uninitialized` is strictly banned and triggers an immediate standard library runtime panic/abort. Because older, vulnerable `rlua` versions rely on `mem::uninitialized` for destructor value-replacement, the Rust runtime aborts immediately during object initialization, rendering runtime-based Use-After-Free/Double-Free verification infeasible.
