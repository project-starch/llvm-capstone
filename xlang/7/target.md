# target.md — mruby #6701 (Row 7)

* **CVE/Issue:** mruby #6701
* **Product:** mruby
* **Status:** SKIPPED
* **Technical Rationale:** Infeasible to reproduce under AddressSanitizer (ASan). In mruby 3.3.0, enabling ASan (under both Clang and GCC) causes the conservative GC stack scanner to miss root pointers, resulting in premature sweeping of core boot symbols (such as Enumerable and class/method hooks) and causing startup NameErrors. Because a functional ASan-instrumented interpreter cannot be booted in this version, verifying the bytecode corruption bug via ASan is infeasible.
