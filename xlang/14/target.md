# target.md — mruby #3596 (Row 14)

* **CVE/Issue:** mruby #3596
* **Product:** mruby
* **Status:** SKIPPED
* **Technical Rationale:** Infeasible to reproduce deterministically under AddressSanitizer (ASan). Stale pointers left in the unused VM stack region under `MRB_GC_STRESS` lead to GC metadata corruption (double-marking or stale-marking reallocated objects). Since this heap corruption does not produce an immediate ASan abort unless precise, non-deterministic fuzzer-driven heap layouts and reallocation alignments are met, reproducing it cleanly in a standalone trigger script is infeasible.
