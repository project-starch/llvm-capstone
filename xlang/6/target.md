# target.md — CVE-2026-1979 (Row 6)

* **CVE:** CVE-2026-1979
* **Product:** mruby
* **Status:** SKIPPED
* **Technical Rationale:** Infeasible to reproduce deterministically under AddressSanitizer (ASan). CVE-2026-1979 is the official CVE identifier assigned to mruby issue #6701 (which is also listed as Row 7). As detailed in the technical rationale for Row 7, compiling and running mruby 3.4.0 with ASan triggers severe boot-time GC stack-scanning mismatches (prematurely sweeping vital core symbols and classes like Enumerable), preventing the ASan-instrumented interpreter from booting successfully and making validation of the pattern-matching JMPNOT-to-JMPIF bytecode corruption infeasible.
